"""World-conditioning modules attached to the DiT: the per-cell G-buffer embedding (world coordinates,
"has geometry" flag, tier, normals), the memory-frame slots and the optional world RoPE layout. `attach_world(model)` adds them
`scene_gbuf(...)` builds the G-buffer of a window from depth and cameras."""
import os
import math, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
import wizard.models.ref as REF
from wizard.memory.retrieve import chunk_latents
M_MAX = 6; FREQ = [2.0 ** k for k in range(-6, 2)]; NF = 3 * len(FREQ) * 2 + 5; CH = chunk_latents(); HW = 15 * 26     # 48 coordinate + log depth + normal (3) + unknown


class GbufEmbed6(nn.Module):
    """G-buffer [B,8,T,60,104] (world coordinate in player heights 3 / log depth 1 / camera-space normal 3 / unknown 1) -> [B,dim,F,15,26]"""
    def __init__(self, dim):
        super().__init__(); self.proj = nn.Linear(4 * 16 * NF, dim); nn.init.zeros_(self.proj.weight); nn.init.zeros_(self.proj.bias)

    @staticmethod
    def features(g):
        xyz, ld, nrm, unk = g[:, :3], g[:, 3:4], g[:, 4:7], g[:, 7:8]; has = (ld > 0).to(g.dtype)
        f = torch.tensor(FREQ, device=g.device, dtype=g.dtype).view(1, 1, -1, 1, 1, 1) * (2 * math.pi)
        a = xyz[:, :, None] * f; four = torch.cat([a.sin(), a.cos()], 2).flatten(1, 2) * has
        return torch.cat([four, ld, nrm * has, unk], 1)

    def forward(self, g, single=False):
        x = self.features(g.float()); B, C, T, H, W = x.shape
        x = x.repeat_interleave(4, 2) if single else torch.cat([x[:, :, :1].repeat(1, 1, 4, 1, 1), x[:, :, 1:]], 2)
        F_ = x.shape[2] // 4
        x = x.view(B, C, F_, 4, H // 4, 4, W // 4, 4).permute(0, 2, 4, 6, 3, 5, 7, 1).reshape(B, F_, H // 4, W // 4, 4 * 16 * C)
        return self.proj(x).permute(0, 4, 1, 2, 3)


def scene_gbuf(depth, pos, R, K, origin, hp, unknown=None):
    """[T,H,W] planar depth (m, 0 = none) + cameras -> [3+1+3+1, T, H, W]: world coordinate / hp from origin, log depth, CAMERA-space normal, unknown. the normal from the cell's neighbours (central differences of the camera-space points, both neighbours with geometry), facing the
    camera; camera space so it is pure shape -- independent of the world coordinate (identity) and of each source's world handedness. Same at inference."""
    import wizard.data.gbuffer as G
    P = G.world_points(depth, pos, R, K); has = depth > 0
    Pc = torch.einsum("thwk,tjk->thwj", P - pos[:, None, None, :], R)          # camera space (right, up, forward)
    dx = torch.zeros_like(Pc); dy = torch.zeros_like(Pc); dx[:, :, 1:-1] = Pc[:, :, 2:] - Pc[:, :, :-2]; dy[:, 1:-1] = Pc[:, 2:] - Pc[:, :-2]
    okn = torch.zeros_like(has); okn[:, 1:-1, 1:-1] = has[:, 1:-1, 2:] & has[:, 1:-1, :-2] & has[:, 2:, 1:-1] & has[:, :-2, 1:-1] & has[:, 1:-1, 1:-1]
    n = torch.cross(dx, dy, dim=-1); n = n / (n.norm(dim=-1, keepdim=True) + 1e-6); n = torch.where(((n * Pc).sum(-1, keepdim=True) > 0), -n, n)
    n = torch.where(okn[..., None], n, torch.zeros_like(n))
    W = torch.where(has[..., None], (P - origin.view(1, 1, 1, 3)) / hp, torch.zeros_like(P))
    ld = torch.where(has, torch.log1p(depth.float()) / math.log1p(300.0), torch.zeros_like(depth.float()))
    u = (unknown & ~has).float() if unknown is not None else torch.zeros_like(ld)
    return torch.cat([W.permute(3, 0, 1, 2), ld[None], n.permute(3, 0, 1, 2), u[None]], 0)


def attach_world(model):
    """gbuf_embedding + the patch embedding adds it when model._gb_in = (video [B,8,121,60,104], memory [B,M,8,60,104]) is set and the input is the
    memory + video stack (M + 31 frames); the references (1 frame) pass untouched. Also installs forward_world as the reference forward."""
    model.gbuf_embedding = GbufEmbed6(model.dim); pe = model.patch_embedding; prev = pe.forward

    def fwd(x):
        h = prev(x); gi = getattr(model, "_gb_in", None)
        if gi is not None and h.shape[2] == M_MAX + 31:
            gc, gm = gi; e = torch.cat([model.gbuf_embedding(gm.permute(0, 2, 1, 3, 4), single=True), model.gbuf_embedding(gc)], 2)
            h = h + e.to(h.dtype)
        return h
    pe.forward = fwd; REF.forward_ref = forward_world; return model


def world_mask(use, n, hide, allow, R, nr, dev):
    """use [B,8,M] chunk -> memory slot; n [B] filled slots; hide [B,M,HW] memory tokens covered by a referenced character; allow [B,R,Sv] (video part,
    memory rows false) -> [B,1,S,S] bool"""
    B = use.shape[0]; Fl = M_MAX + 31; Sv = Fl * HW; S = Sv + R * nr
    fl = torch.full((31,), -1, dtype=torch.long)
    for c, lats in enumerate(CH):
        for j in lats: fl[j] = c
    tok_f = torch.arange(Sv, device=dev) // HW; is_m = tok_f < M_MAX; ch = fl.to(dev)[(tok_f - M_MAX).clamp(min=0)]
    m = torch.zeros(B, 1, S, S, dtype=torch.bool, device=dev)
    for b in range(B):
        mm = (~is_m)[:, None] & (~is_m)[None, :]
        rd = use[b].to(dev)[ch][:, tok_f.clamp(max=M_MAX - 1)] & is_m[None, :] & (~is_m)[:, None]          # video token of chunk c -> memory frames c picked
        vis = ~hide[b].to(dev).reshape(-1)                                                                    # [M*HW]
        rd[:, :M_MAX * HW] &= vis[None, :]
        mm |= rd; mm |= is_m[:, None] & is_m[None, :] & (tok_f[:, None] == tok_f[None, :])
        m[b, 0, :Sv, :Sv] = mm
        for r in range(R):
            a = Sv + r * nr; m[b, 0, :Sv, a:a + nr] = allow[b, r][:, None]; m[b, 0, a:a + nr, a:a + nr] = True
    return m


def forward_world(model, x, timestep, context, ref):
    from wizard.models.dit import sinusoidal_embedding_1d
    B = x.shape[0]; h = model.patch_embedding(x); Fl, Hl, Wl = h.shape[2:]; hw = Hl * Wl; Sv = Fl * hw; assert Fl == M_MAX + 31 and hw == HW
    h = h.flatten(2).transpose(1, 2)
    lat = ref["lat"].to(h.dtype); R = lat.shape[1]
    hr = model.patch_embedding(lat.flatten(0, 1)); Hr, Wr = hr.shape[3:]; nr = Hr * Wr
    hr = hr.flatten(2).transpose(1, 2).reshape(B, R * nr, -1); h = torch.cat([h, hr], 1); S = h.shape[1]
    ts_tok = torch.cat([timestep.unsqueeze(-1).expand(B, Fl, hw).reshape(B, Sv), timestep.new_zeros(B, R * nr)], 1)
    t = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, ts_tok.reshape(-1)).to(h.dtype)).view(B, S, model.dim)
    t_mod = model.time_projection(t).unflatten(2, (6, model.dim)); ctx = model.text_embedding(context)
    cov = torch.cat([ref["cover"].new_zeros((B, M_MAX) + ref["cover"].shape[2:]), ref["cover"]], 1)       # memory frames are nobody's figure
    allow = REF.gate(cov, ref["slots"].long())
    mask = world_mask(ref["mem_use"], ref["mem_n"], ref["mem_hide"], allow, R, nr, h.device); info = (Sv, R, Hr, Wr, REF.REF_POS)
    WARP["on"] = False; WROPE["on"] = False
    lay = getattr(model, "_mem_wrope", None)
    if lay is not None:                                                                    # world RoPE for reading memory
        nm = M_MAX * hw; install_wrope_split()
        WROPE.update(on=True, nm=nm, ksrc=lay["ksrc"].to(h.device), qf=_freqs_at(model.rope, lay["qpos"].to(h.device), h.device), kf=_freqs_at(model.rope, lay["kpos"].to(h.device), h.device))
        mask = wrope_mask(mask, lay["kchunk"], nm, Sv)
    if getattr(model, "_mem_warp", None) is not None:                                      # memory RoPE warping
        wp = model._mem_warp.to(h.device); nm = M_MAX * hw; valid = torch.isfinite(wp).all(-1)
        install_warp_attention(); WARP.update(on=True, nm=nm, freqs=_freqs_at(model.rope, torch.nan_to_num(wp), h.device))
        mask = warp_mask(mask, valid, nm)
    mb = float(os.environ.get("MEM_BIAS", "0"))
    if mb > 0 and getattr(model, "_gb_in", None) is not None:                          # attention bias by WORLD position: video token -> memory token
        mask = mem_bias_mask(mask, model._gb_in, mb, float(os.environ.get("MEM_BIAS_SIG", "1.5")), h.dtype)
    for blk in model.blocks:
        if model.use_gradient_checkpointing and model.training:
            h = torch_checkpoint(blk, h, t_mod, ctx, model.rope, Fl, Hl, Wl, hw, mask, use_reentrant=False, ref=info)
        else:
            h = blk(h, t_mod, ctx, model.rope, Fl, Hl, Wl, hw, mask, ref=info)
    a = M_MAX * hw
    return model._unpatchify(model.head(h[:, a:Sv], t[:, a:Sv]), 31, Hl, Wl)


def ident_aug(gs, gen=None, kmax=4, tsd=5.0):
    """the world coordinate is an IDENTITY, not a position: the same random piecewise rigid transform on the
    coordinate channels (0:3, cells with geometry) of every G-buffer of one sample (the window and its memory frames). Pieces = Voronoi cells of K
    (1..kmax) seeds drawn from the window's own points, assigned by the ORIGINAL coordinate, so a world point gets the same new coordinate in the window
    and in every memory frame; each piece: a uniform random rotation + a N(0, tsd) translation (player heights). Pixels, depth, normals untouched.
    gs: list of [8,T,H,W] -> new list."""
    g0 = gs[0]; dev = g0.device; has0 = g0[3] > 0; pts = g0[:3].permute(1, 2, 3, 0)[has0]
    if len(pts) == 0: return gs
    K = int(torch.randint(1, kmax + 1, (1,), generator=gen)); seeds = pts[torch.randint(0, len(pts), (K,), generator=gen).to(dev)]
    q = torch.randn(K, 4, generator=gen).to(dev); q = q / q.norm(dim=1, keepdim=True); w, x, y, z = q.unbind(1)
    Rk = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                      2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], 1).view(K, 3, 3)
    tk = (torch.randn(K, 3, generator=gen) * tsd).to(dev); out = []
    for g in gs:
        g = g.clone(); has = g[3] > 0; P = g[:3].permute(1, 2, 3, 0)                                     # [T,H,W,3]
        lab = torch.cdist(P.reshape(-1, 3), seeds).argmin(1).view(P.shape[:-1])
        Pn = torch.einsum("thwij,thwj->thwi", Rk[lab], P) + tk[lab]
        g[:3] = torch.where(has[..., None], Pn, P).permute(3, 0, 1, 2); out.append(g)
    return out

def token_coords(g, frames):
    """g [8,T,60,104] scene_gbuf; frames: list over tokens' latent frames of pixel-frame lists -> coords [L,15,26,3], valid [L,15,26]"""
    W = g[:3]; has = (g[3] > 0).float(); out, val = [], []
    for fs in frames:
        w = W[:, fs] * has[fs][None]; c = has[fs].sum(0)                                   # [3,h,w], [h,w]
        ws = F.avg_pool2d(w.sum(1)[None], 4)[0] * 16; cs = F.avg_pool2d(c[None, None], 4)[0, 0] * 16
        out.append((ws / cs.clamp(min=1)).permute(1, 2, 0)); val.append(cs > 0)
    return torch.stack(out), torch.stack(val)

def mem_bias_mask(mask, gi, beta, sig, dtype):
    """bool mask [B,1,S,S] -> float mask: -inf where not allowed, + beta * exp(-d^2 / 2 sig^2) on allowed video -> memory pairs (d = world distance, player heights)"""
    gc, gm = gi; gc, gm = gc[0], gm[0]                                                     # [8,121,60,104], [M,8,60,104]
    vf = [[0]] + [list(range(4 * j - 3, 4 * j + 1)) for j in range(1, 31)]
    vc, vv = token_coords(gc, vf)                                                          # [31,15,26,3]
    mc, mv = token_coords(gm.permute(1, 0, 2, 3), [[j] for j in range(gm.shape[0])])      # [M,15,26,3]
    vc, vv, mc, mv = vc.reshape(-1, 3), vv.reshape(-1), mc.reshape(-1, 3), mv.reshape(-1)
    d2 = torch.cdist(vc.float(), mc.float()) ** 2
    rel = float(os.environ.get("MEM_BIAS_REL", "0"))
    if rel > 0:                                                                            # sigma grows with distance from the camera (far geometry is estimated loosely)
        rng = vc.float().norm(dim=-1).clamp(min=1.0)[:, None]; sg2 = (sig + rel * rng) ** 2   # coordinates are from the origin near the start: |c| ~ distance
    else: sg2 = sig * sig
    b = beta * torch.exp(-d2 / (2 * sg2)) * (vv[:, None] & mv[None, :]).float()
    nm = mc.shape[0]; S = mask.shape[-1]
    fm = torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(~mask, float("-inf"))
    fm[:, :, nm:nm + vc.shape[0], :nm] += b.to(dtype)[None, None]
    return fm


# ───────────────────────── memory positional-encoding warping (as UCM, arXiv 2602.22960) ─────────────────────────
# A memory token's world point is projected into the camera of the chunk that reads it; video queries see that memory key roped at the
# projected (latent frame of the chunk, row, column) -- the same RoPE position as the video token at the same world point -- so the
# pretrained locality of RoPE ("same place, near in time") becomes "same WORLD place". Memory queries keep reading their own frame at
# the original positions (a second copy of the memory keys). No new parameters. Off unless model._mem_warp is set.
WARP = {}

def mem_warp_pos(mdep, mpos, mR, mK, tpos, tR, tK, use, n):
    """mdep [M,60,104] (characters zeroed), mpos [M,3], mR [M,3,3], mK [M,4] or [4]; target cameras tpos [121,3], tR [121,3,3], tK [4]; use [8,M] bool.
    -> pos [M*HW,3] float (t in the video's RoPE frame, h, w) with NaN where the token has no world point / is not seen by its chunk"""
    import wizard.data.gbuffer as G
    dev = mdep.device; M = mdep.shape[0]; out = torch.full((M, 15, 26, 3), float("nan"), device=dev)
    for j in range(int(n)):
        cs = [c for c in range(len(CH)) if bool(use[c, j])]
        if not cs: continue
        Kj = mK[j] if mK.dim() > 1 else mK
        P = G.world_points(mdep[j:j + 1].float(), mpos[j:j + 1].float(), mR[j:j + 1].float(), Kj)[0]   # [60,104,3]
        has = (mdep[j] > 0).float()
        Ps = F.avg_pool2d((P * has[..., None]).permute(2, 0, 1)[None], 4)[0] * 16; hs = F.avg_pool2d(has[None, None], 4)[0, 0] * 16
        Pt = (Ps / hs.clamp(min=1)).permute(1, 2, 0); okt = hs > 0                                   # [15,26,3] token world points
        fx, fy, cx, cy = [float(v) for v in tK]; best = None
        for c in cs:                                                                                # the most relevant view (UCM): the chunk that sees most of it
            lat = CH[c][len(CH[c]) // 2]; pf = 0 if lat == 0 else 4 * lat - 2
            rel = Pt - tpos[pf].float(); x = rel @ tR[pf, 0].float(); y = rel @ tR[pf, 1].float(); z = rel @ tR[pf, 2].float()
            u = cx + fx * x / z.clamp(min=1e-3); v = cy - fy * y / z.clamp(min=1e-3)
            w_ = (u - 2.0) / 4.0; h_ = (v - 2.0) / 4.0                                              # token index (cell centres at i + 0.5, token t = cells 4t..4t+3)
            good = okt & (z > 0.1) & (w_ >= -0.5) & (w_ <= 25.5) & (h_ >= -0.5) & (h_ <= 14.5)
            if best is None or int(good.sum()) > int(best[0].sum()): best = (good, lat, h_, w_)
        good, lat, h_, w_ = best
        out[j, ..., 0] = torch.where(good, torch.full_like(w_, float(M_MAX + lat)), out[j, ..., 0])
        out[j, ..., 1] = torch.where(good, h_, out[j, ..., 1]); out[j, ..., 2] = torch.where(good, w_, out[j, ..., 2])
    return out.reshape(-1, 3)

def _freqs_at(rope, pos, device):
    """complex RoPE freqs at float positions pos [N,3] (t, h, w) -> [N, head_dim//2]"""
    parts = []
    for tab, p in ((rope.f_t, pos[:, 0]), (rope.f_h, pos[:, 1]), (rope.f_w, pos[:, 2])):
        w = torch.angle(tab[1]).to(device).float()                                                  # per-column angular frequency
        parts.append(torch.polar(torch.ones(p.shape[0], w.shape[0], device=device), p.float()[:, None] * w[None, :]))
    return torch.cat(parts, -1)

def install_warp_attention():
    import wizard.models.dit as GD
    SA = GD.SelfAttention
    if getattr(SA, "_warp_installed", False): return
    orig = SA.forward
    def fwd(self, x, rope, Fl, Hl, Wl, attn_mask=None, kv_cache=None, layer_idx=None, frame_index=None, commit=False, tf=False, ref=None):
        if not WARP.get("on") or kv_cache is not None or tf or ref is None:
            return orig(self, x, rope, Fl, Hl, Wl, attn_mask, kv_cache, layer_idx, frame_index, commit, tf, ref)
        B, S, _ = x.shape; n, d = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(B, S, n, d); k = self.norm_k(self.k(x)).view(B, S, n, d); v = self.v(x).view(B, S, n, d)
        Sv, R, Hr, Wr, pos = ref; nr = Hr * Wr; nm = WARP["nm"]
        kraw = k[:, :nm]
        q = torch.cat([rope(q[:, :Sv], Fl, Hl, Wl)] + [rope(q[:, Sv + r * nr:Sv + (r + 1) * nr], 1, Hr, Wr, f_start=pos + r) for r in range(R)], dim=1)
        k = torch.cat([rope(k[:, :Sv], Fl, Hl, Wl)] + [rope(k[:, Sv + r * nr:Sv + (r + 1) * nr], 1, Hr, Wr, f_start=pos + r) for r in range(R)], dim=1)
        fq = WARP["freqs"]                                                                          # [nm, d//2] (invalid tokens: anything, masked off)
        kc = torch.view_as_complex(kraw.float().reshape(B, nm, n, d // 2, 2)) * fq[None, :, None, :]
        kw = torch.view_as_real(kc).reshape(B, nm, n, d).to(k.dtype)
        k = torch.cat([k, kw], 1); v = torch.cat([v, v[:, :nm]], 1)
        o = GD.sdpa(q, k, v, attn_mask=attn_mask).reshape(B, S, n * d)
        return self.o(o)
    SA.forward = fwd; SA._warp_installed = True

def warp_mask(mask, valid, nm):
    """bool [B,1,S,S] -> [B,1,S,S+nm]: video -> memory reads move to the warped copy for tokens with a valid warp; memory rows keep the original"""
    B, _, S, _ = mask.shape; ext = torch.zeros(B, 1, S, nm, dtype=mask.dtype, device=mask.device)
    vid = slice(nm, S); vm = valid.to(mask.device)
    ext[:, :, vid] = mask[:, :, vid, :nm] & vm[None, None, None, :]
    m = mask.clone(); m[:, :, vid, :nm] = mask[:, :, vid, :nm] & ~vm[None, None, None, :]
    return torch.cat([m, ext], -1)


# ───────────────────────── world RoPE for reading memory (tailored to our known cameras + world points) ─────────────────────────
# Every video query of chunk c and every memory key read by chunk c get a position on the SAME chart: the direction seen from chunk c's camera
# centre, in world-up spherical angles (azimuth relative to the chunk's yaw, elevation), scaled so one screen token = one unit (the pretrained
# RoPE bandwidth), temporal slot = the chunk (neutral: equal for q and k). Queries with geometry use their world point (no parallax error), others
# their viewing ray (the camera is always known). Memory keys: their world point; occluded duplicates (a nearer memory token in the same
# direction) dropped. Per chunk a separate copy of its memory keys. Implemented exactly with ONE attention call by concatenating head dims:
# q = [q_screen | q_world], k = [k_screen | 0] for video / reference / own-frame memory keys and [0 | k_world] for the world copies (scale 1/sqrt(d)).
WROPE = {}

def world_rope_layout(mdep, mpos, mR, mK, n, hide, use, qdep, tpos, tR, tK):
    """mdep [M,60,104] (characters zeroed), mpos [M,3], mR [M,3,3], mK [M,4]/[4], n, hide [M,390] bool, use [8,M]; qdep [121,60,104] window depth
    (0 = unknown / character), tpos [121,3], tR [121,3,3], tK [4]. -> dict(qpos [31*390,3], kpos [N,3], ksrc [N], kchunk [N]) (positions in token units)"""
    import wizard.data.gbuffer as G
    dev = mdep.device; fx, fy, cx, cy = [float(v) for v in tK]; dl = 4.0 / fx                      # radians per screen token (4 grid cells per token)
    gv, gu = torch.meshgrid(torch.arange(60, device=dev, dtype=torch.float32) + 0.5, torch.arange(104, device=dev, dtype=torch.float32) + 0.5, indexing="ij")
    pool4 = lambda a: F.avg_pool2d(a[None], 4)[0]
    def ang(d, yaw, pitch):                                                                     # d [...,3] world directions -> (h, w) token units
        az = torch.atan2(d[..., 0], d[..., 2]) - yaw; az = torch.remainder(az + math.pi, 2 * math.pi) - math.pi
        el = torch.atan2(d[..., 1], torch.sqrt(d[..., 0] ** 2 + d[..., 2] ** 2))
        return (pitch - el) / dl + 7.0, az / dl + 12.5
    cams = []
    for c, lats in enumerate(CH):
        lat = lats[len(lats) // 2]; pf = 0 if lat == 0 else 4 * lat - 2; f_ = tR[pf, 2].float()
        cams.append((tpos[pf].float(), math.atan2(float(f_[0]), float(f_[2])), math.atan2(float(f_[1]), float(torch.sqrt(f_[0] ** 2 + f_[2] ** 2))), float(M_MAX + lat)))
    qpos = torch.zeros(31, 15, 26, 3, device=dev)
    for c, lats in enumerate(CH):
        C, yaw, pitch, tt = cams[c]
        for l in lats:
            pf = 0 if l == 0 else 4 * l - 2; R = tR[pf].float(); z = qdep[pf].float()
            x = (gu - cx) / fx; y = (gv - cy) / fy; ray = x[..., None] * R[0] - y[..., None] * R[1] + R[2]
            P = tpos[pf].float() + z[..., None] * ray; has = (z > 0).float()
            dP = pool4((P - C).permute(2, 0, 1) * has[None]) * 16; hs = pool4(has[None])[0] * 16
            dr = pool4(ray.permute(2, 0, 1))                                                      # mean ray of the token
            d = dr.permute(1, 2, 0) if os.environ.get("WROPE_QDEPTH", "0") != "1" else torch.where((hs > 0)[None], dP / hs.clamp(min=1)[None], dr).permute(1, 2, 0)   # default: the viewing ray only (train = inference)
            h_, w_ = ang(d, yaw, pitch); qpos[l, ..., 0] = tt; qpos[l, ..., 1] = h_; qpos[l, ..., 2] = w_
    M = mdep.shape[0]; Pk = torch.zeros(M, 15, 26, 3, device=dev); okk = torch.zeros(M, 15, 26, dtype=torch.bool, device=dev)
    Rk = torch.zeros(M, 15, 26, 3, device=dev); skyk = torch.zeros(M, 15, 26, dtype=torch.bool, device=dev)
    for j in range(int(n)):
        Kj = mK[j] if mK.dim() > 1 else mK
        P = G.world_points(mdep[j:j + 1].float(), mpos[j:j + 1].float(), mR[j:j + 1].float(), Kj)[0]; has = (mdep[j] > 0).float()
        Ps = pool4(P.permute(2, 0, 1) * has[None]) * 16; hs = pool4(has[None])[0] * 16
        Pk[j] = (Ps / hs.clamp(min=1)[None]).permute(1, 2, 0); okk[j] = (hs > 0) & ~hide[j].reshape(15, 26).to(dev)
        fxj, fyj, cxj, cyj = [float(v) for v in Kj]; Rj = mR[j].float()                              # a token with no depth is sky:
        ray = ((gu - cxj) / fxj)[..., None] * Rj[0] - ((gv - cyj) / fyj)[..., None] * Rj[1] + Rj[2]     # at infinity, its direction is the same from every camera
        Rk[j] = pool4(ray.permute(2, 0, 1)).permute(1, 2, 0); skyk[j] = (hs == 0) & ~hide[j].reshape(15, 26).to(dev)
    kpos, ksrc, kch = [], [], []
    for c in range(len(CH)):
        js = [j for j in range(int(n)) if bool(use[c, j])]
        if not js: continue
        C, yaw, pitch, tt = cams[c]
        sk = skyk[js] if os.environ.get("WROPE_SKY", "1") == "1" else torch.zeros_like(skyk[js])
        d = torch.where(sk[..., None], Rk[js], Pk[js] - C); dist = torch.where(sk, torch.full_like(sk, 1e6, dtype=torch.float32), (Pk[js] - C).norm(dim=-1)); h_, w_ = ang(d, yaw, pitch)
        ok = (okk[js] | sk) & (h_ > -1.5) & (h_ < 15.5) & (w_ > -1.5) & (w_ < 26.5) & (dist > 0.3)
        hb = h_.round().clamp(-2, 16).long() + 2; wb = w_.round().clamp(-2, 27).long() + 2; bin_ = hb * 30 + wb
        best = torch.full((19 * 30,), float("inf"), device=dev)
        best.scatter_reduce_(0, bin_[ok], dist[ok], reduce="amin")
        if os.environ.get("WROPE_OCC", "0") == "1": ok = ok & (dist <= 1.1 * best[bin_] + 0.5)      # occlusion dedupe off by default (noisy depth drops valid keys)
        for i, j in enumerate(js):
            sel = ok[i].reshape(-1)
            if not sel.any(): continue
            idx = torch.nonzero(sel)[:, 0]
            kpos.append(torch.stack([torch.full_like(h_[i].reshape(-1)[idx], tt), h_[i].reshape(-1)[idx], w_[i].reshape(-1)[idx]], -1))
            ksrc.append(j * HW + idx); kch.append(torch.full_like(idx, c))
    if not kpos: return None
    return dict(qpos=qpos.reshape(-1, 3), kpos=torch.cat(kpos), ksrc=torch.cat(ksrc), kchunk=torch.cat(kch))

def install_wrope_attention():
    import wizard.models.dit as GD
    SA = GD.SelfAttention
    if getattr(SA, "_wrope_installed", False): return
    install_warp_attention(); prev = SA.forward
    def fwd(self, x, rope, Fl, Hl, Wl, attn_mask=None, kv_cache=None, layer_idx=None, frame_index=None, commit=False, tf=False, ref=None):
        if not WROPE.get("on") or kv_cache is not None or tf or ref is None:
            return prev(self, x, rope, Fl, Hl, Wl, attn_mask, kv_cache, layer_idx, frame_index, commit, tf, ref)
        B, S, _ = x.shape; n, d = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(B, S, n, d); k = self.norm_k(self.k(x)).view(B, S, n, d); v = self.v(x).view(B, S, n, d)
        Sv, R, Hr, Wr, pos = ref; nr = Hr * Wr; nm = WROPE["nm"]; src = WROPE["ksrc"]
        qv, kraw = q[:, nm:Sv], k[:, src]
        rot = lambda t, fq: torch.view_as_real(torch.view_as_complex(t.float().reshape(B, t.shape[1], n, d // 2, 2)) * fq[None, :, None, :]).reshape(B, t.shape[1], n, d).to(t.dtype)
        qs = torch.cat([rope(q[:, :Sv], Fl, Hl, Wl)] + [rope(q[:, Sv + r * nr:Sv + (r + 1) * nr], 1, Hr, Wr, f_start=pos + r) for r in range(R)], dim=1)
        ks = torch.cat([rope(k[:, :Sv], Fl, Hl, Wl)] + [rope(k[:, Sv + r * nr:Sv + (r + 1) * nr], 1, Hr, Wr, f_start=pos + r) for r in range(R)], dim=1)
        qw = torch.zeros_like(qs); qw[:, nm:Sv] = rot(qv, WROPE["qf"])
        kw = rot(kraw, WROPE["kf"])
        Q = torch.cat([qs, qw], -1); Kk = torch.cat([torch.cat([ks, torch.zeros_like(ks)], -1), torch.cat([torch.zeros_like(kw), kw], -1)], 1)
        V = torch.cat([v, v[:, src]], 1)
        o = torch.nn.functional.scaled_dot_product_attention(Q.transpose(1, 2), Kk.transpose(1, 2), V.transpose(1, 2), attn_mask=attn_mask, scale=d ** -0.5)
        return self.o(o.transpose(1, 2).reshape(B, S, n * d))
    SA.forward = fwd; SA._wrope_installed = True

def wrope_mask(mask, kchunk, nm, Sv):
    """bool [B,1,S,S] -> [B,1,S,S+N]: video rows stop reading the original memory keys; video tokens of chunk c read the world copies of chunk c"""
    B, _, S, _ = mask.shape; dev = mask.device
    fl = torch.full((31,), -1, dtype=torch.long)
    for c, lats in enumerate(CH):
        for j in lats: fl[j] = c
    rows_c = fl.to(dev)[torch.arange(Sv - nm, device=dev) // HW]                                     # chunk of each video token
    m = mask.clone(); m[:, :, nm:Sv, :nm] = False
    ext = torch.zeros(B, 1, S, kchunk.shape[0], dtype=mask.dtype, device=dev)
    ext[:, :, nm:Sv] = (rows_c[:, None] == kchunk.to(dev)[None, :])[None, None]
    return torch.cat([m, ext], -1)


def _eff_attn(q, k, v, mask, scale, kernel=None):
    """q [B,Sq,N,D], k/v [B,Sk,N,D], bool mask [B,1,Sq,Sk] -> (o [B,Sq,N,D], lse [B,N,Sq] fp32); keys padded to a multiple of 16 (bias alignment)"""
    B, Sq, N, D = q.shape; Sk = k.shape[1]; pad = (-Sk) % 16
    if pad:
        k = torch.cat([k, k.new_zeros(B, pad, N, D)], 1); v = torch.cat([v, v.new_zeros(B, pad, N, D)], 1)
        mask = torch.cat([mask, mask.new_zeros(B, 1, Sq, pad)], -1)
    bias = torch.zeros(mask.shape, dtype=q.dtype, device=q.device).masked_fill(~mask, float("-inf")).expand(B, N, Sq, Sk + pad)
    kernel = kernel or os.environ.get("WROPE_K1", "cudnn")
    try:                                                                                        # cuDNN: as fast as the plain masked attention, returns the log-sum-exp
        if kernel != "cudnn": raise RuntimeError("efficient")
        out = torch.ops.aten._scaled_dot_product_cudnn_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), bias, True, 0.0, False, False, scale=scale)
    except RuntimeError:
        out = torch.ops.aten._scaled_dot_product_efficient_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), bias, True, 0.0, False, scale=scale)
    o, lse = out[0], out[1]; lse = lse.reshape(B, N, -1)[..., :Sq].float()
    return o.transpose(1, 2), lse

def install_wrope_split():
    """world RoPE without doubling head dims: attention 1 = the usual one (video rows no longer read the original memory keys),
    attention 2 = video queries (world-roped) x world memory keys; merged exactly by their log-sum-exp (as blockwise flash attention)."""
    import wizard.models.dit as GD
    SA = GD.SelfAttention
    if getattr(SA, "_wrope_split", False): return
    install_wrope_attention(); prev = SA.forward
    def fwd(self, x, rope, Fl, Hl, Wl, attn_mask=None, kv_cache=None, layer_idx=None, frame_index=None, commit=False, tf=False, ref=None):
        if not WROPE.get("on") or os.environ.get("WROPE_SPLIT", "1") != "1" or kv_cache is not None or tf or ref is None:
            return prev(self, x, rope, Fl, Hl, Wl, attn_mask, kv_cache, layer_idx, frame_index, commit, tf, ref)
        B, S, _ = x.shape; n, d = self.num_heads, self.head_dim
        q = self.norm_q(self.q(x)).view(B, S, n, d); k = self.norm_k(self.k(x)).view(B, S, n, d); v = self.v(x).view(B, S, n, d)
        Sv, R, Hr, Wr, pos = ref; nr = Hr * Wr; nm = WROPE["nm"]; src = WROPE["ksrc"]
        rot = lambda t, fq: torch.view_as_real(torch.view_as_complex(t.float().reshape(B, t.shape[1], n, d // 2, 2)) * fq[None, :, None, :]).reshape(B, t.shape[1], n, d).to(t.dtype)
        qw, kw, vw = rot(q[:, nm:Sv], WROPE["qf"]), rot(k[:, src], WROPE["kf"]), v[:, src]
        qs = torch.cat([rope(q[:, :Sv], Fl, Hl, Wl)] + [rope(q[:, Sv + r * nr:Sv + (r + 1) * nr], 1, Hr, Wr, f_start=pos + r) for r in range(R)], dim=1)
        ks = torch.cat([rope(k[:, :Sv], Fl, Hl, Wl)] + [rope(k[:, Sv + r * nr:Sv + (r + 1) * nr], 1, Hr, Wr, f_start=pos + r) for r in range(R)], dim=1)
        sc = d ** -0.5
        o1, l1 = _eff_attn(qs, ks, v, attn_mask[..., :S], sc)                                      # [B,S,n,d], [B,n,S]
        m2 = attn_mask[:, :, nm:Sv, S:]
        rows = torch.nonzero(m2[0, 0].any(-1))[:, 0]                                               # only rows that read >= 1 world key: a fully masked
        if rows.numel() == 0: return self.o(o1.reshape(B, S, n * d))                               # row gives NaN gradients in the cuDNN backward
        o2, l2 = _eff_attn(qw[:, rows], kw, vw, m2[:, :, rows], sc, kernel=os.environ.get("WROPE_K2", "efficient"))
        l1v = l1[:, :, nm:Sv][:, :, rows]
        mx = torch.maximum(l1v, l2); w1 = torch.exp(l1v - mx); w2 = torch.exp(l2 - mx)            # [B,n,r]
        w1, w2 = w1.transpose(1, 2)[..., None], w2.transpose(1, 2)[..., None]
        ov = (o1[:, nm:Sv][:, rows].float() * w1 + o2.float() * w2) / (w1 + w2)
        o = o1.clone(); o[:, nm + rows] = ov.to(o1.dtype)
        if os.environ.get("ATTN_DIAG"):
            import wizard.models.attn_diag as AD
            if AD.ON: AD.record_wrope(rows, w1, w2, m2[:, :, rows], attn_mask[:, :, nm:Sv][:, :, rows][..., :S])
        return self.o(o.reshape(B, S, n * d))
    SA.forward = fwd; SA._wrope_split = True
