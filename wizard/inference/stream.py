"""KV-cache streaming of the causal model (wizard.train.causal.forward_tf), one window of 31 latent frames, equivalent to the
teacher-forcing forward:
  * positions as training: memory slot j at temporal position j, video latent frame f at M_MAX + f (absolute; post-RoPE K cached)
  * memory frames and reference images read only themselves -> their K/V per layer are captured once per window (and per text: the
    unconditional pass of guidance needs its own capture)
  * frame f (one latent frame, 390 tokens) reads: the cached clean frames inside the history window (> f - W) or the sink (< S), itself,
    the memory slots its chunk retrieved minus character-hidden tokens, reference r inside its figure's box
  * commit: the clean frame is run at t = 0 and its K/V kept (exactly the clean half of training)
Installed by patching SelfAttention.forward; the original path runs whenever kv_cache is not this module's dict."""
import torch, torch.nn.functional as F
from wizard.models.dit import SelfAttention, sdpa, sinusoidal_embedding_1d
import wizard.models.ref as REF
from wizard.models.world_cond import M_MAX, CH, HW

_ORIG = SelfAttention.forward
FL = torch.full((31,), -1, dtype=torch.long)
for _c, _lats in enumerate(CH):
    for _j in _lats: FL[_j] = _c


def _fwd(self, x, rope, Fl, Hl, Wl, attn_mask=None, kv_cache=None, layer_idx=None, frame_index=None, commit=False, tf=False, ref=None):
    if not (isinstance(kv_cache, dict) and kv_cache.get("kv_stream")):
        return _ORIG(self, x, rope, Fl, Hl, Wl, attn_mask, kv_cache, layer_idx, frame_index, commit, tf, ref)
    B, S, _ = x.shape; n, d = self.num_heads, self.head_dim; li = layer_idx; mode = kv_cache["mode"]
    q = self.norm_q(self.q(x)).view(B, S, n, d); k = self.norm_k(self.k(x)).view(B, S, n, d); v = self.v(x).view(B, S, n, d)
    if mode == "mem":                                                                     # M memory slots, each alone at position j
        hw = Hl * Wl; M = S // hw
        q = torch.cat([rope(q[:, j * hw:(j + 1) * hw], 1, Hl, Wl, f_start=j) for j in range(M)], 1)
        k = torch.cat([rope(k[:, j * hw:(j + 1) * hw], 1, Hl, Wl, f_start=j) for j in range(M)], 1)
        own = torch.arange(S, device=x.device) // hw
        o = sdpa(q, k, v, attn_mask=own[:, None] == own[None, :]).reshape(B, S, n * d)
        kv_cache["mk"][li] = k; kv_cache["mv"][li] = v; return self.o(o)
    if mode == "ref":                                                                     # R references, each alone at REF_POS + r
        R, Hr, Wr = kv_cache["ref_shape"]; nr = Hr * Wr
        q = torch.cat([rope(q[:, r * nr:(r + 1) * nr], 1, Hr, Wr, f_start=REF.REF_POS + r) for r in range(R)], 1)
        k = torch.cat([rope(k[:, r * nr:(r + 1) * nr], 1, Hr, Wr, f_start=REF.REF_POS + r) for r in range(R)], 1)
        own = torch.arange(S, device=x.device) // nr
        o = sdpa(q, k, v, attn_mask=own[:, None] == own[None, :]).reshape(B, S, n * d)
        kv_cache["rk"][li] = k; kv_cache["rv"][li] = v; return self.o(o)
    f = frame_index; pos = M_MAX + f                                                       # one video latent frame
    q = rope(q, 1, Hl, Wl, f_start=pos); k = rope(k, 1, Hl, Wl, f_start=pos)
    W, sink = kv_cache["W"], kv_cache["sink"]; cache = kv_cache["vk"][li]
    hist = [g for g in sorted(cache) if g < f and ((W and g > f - W) or not W or g < sink)]
    ks = [cache[g][0] for g in hist] + [k]; vs = [cache[g][1] for g in hist] + [v]
    hh = kv_cache.get("hist_hide")
    if hh is None: allow = [torch.ones(B, S, (len(hist) + 1) * S, dtype=torch.bool, device=x.device)]
    else:
        memo = kv_cache.setdefault("hist_hide_memo", {})
        for g in hist:
            if (f, g) not in memo: memo[(f, g)] = (~hh(f, g)).to(x.device)
        allow = [memo[(f, g)][None, None, :].expand(B, S, S) for g in hist] + [torch.ones(B, S, S, dtype=torch.bool, device=x.device)]
    mk = kv_cache["mk"][li]
    if mk is not None:                                                                    # memory: the slots chunk(f) retrieved, minus hidden tokens
        ma = kv_cache["mem_allow"][f]                                                     # [B, M*hw]
        ks.append(mk); vs.append(kv_cache["mv"][li]); allow.append(ma[:, None, :].expand(B, S, ma.shape[1]))
    rk = kv_cache["rk"][li]
    if rk is not None:
        ks.append(rk); vs.append(kv_cache["rv"][li]); allow.append(kv_cache["ref_allow"][f])   # [B, hw, R*nr]
    o = sdpa(q, torch.cat(ks, 1), torch.cat(vs, 1), attn_mask=torch.cat(allow, 2)[:, None]).reshape(B, S, n * d)
    if commit: cache[f] = (k.detach(), v.detach())
    return self.o(o)


SelfAttention.forward = _fwd


class Stream:
    """One window, one text. model: causal WizardDiT with attach_cond + attach_world (bf16, eval). gc [1,8,121,60,104] / gm [1,M,8,60,104]: the
    coordinate inputs exactly as given to model._gb_in; mem_lat [1,48,M,30,52]; ref = dict(lat [1,R,48,1,H,W], slots [1,R], cover [1,31,15,26,3],
    mem_use [8,M] or [1,8,M], mem_hide [1,M,HW], mem_n)."""

    @torch.no_grad()
    def __init__(self, model, ctx, gc, gm, mem_lat, ref, W=20, sink=1):
        self.m = model; dev = mem_lat.device; dt = next(model.parameters()).dtype; n = len(model.blocks)
        self.kv = dict(kv_stream=True, mode=None, W=W, sink=sink, vk=[{} for _ in range(n)], mk=[None] * n, mv=[None] * n, rk=[None] * n, rv=[None] * n)
        self.ctx = model.text_embedding(ctx.to(dt))
        self.ev = model.gbuf_embedding(gc.to(dev)).to(dt)                                  # [1,dim,31,15,26] video coordinate embedding
        use = ref["mem_use"].to(dev); use = use[0] if use.dim() == 3 else use; hide = ref["mem_hide"].to(dev)
        nmem = int(ref["mem_n"]) if ref.get("mem_n") is not None else M_MAX
        # memory capture (every slot, like training: empty slots exist but no chunk reads them)
        em = model.gbuf_embedding(gm.permute(0, 2, 1, 3, 4).to(dev), single=True).to(dt)     # [1,dim,M,15,26]
        from wizard.models.cond import cat_cond
        hm = self._pe(cat_cond(mem_lat.to(dt), torch.zeros_like(mem_lat).to(dt))) + em    # [1,dim,M,15,26]
        h = hm.flatten(2).transpose(1, 2); self._run(h, torch.zeros(1, h.shape[1], device=dev), "mem", Fl=M_MAX)
        self.kv["mem_allow"] = {}
        vis = (~hide[0]).reshape(-1)                                                        # [M*HW]
        slot = torch.arange(M_MAX * HW, device=dev) // HW
        for f in range(31):
            self.kv["mem_allow"][f] = (use[FL[f]][slot] & vis)[None]                       # [1, M*HW]
        # reference capture
        lat = ref["lat"].to(dev, dt); R = lat.shape[1]
        hr = self._pe(lat.flatten(0, 1)); Hr, Wr = hr.shape[3:]; nr = Hr * Wr; self.kv["ref_shape"] = (R, Hr, Wr)
        hr = hr.flatten(2).transpose(1, 2).reshape(1, R * nr, -1); self._run(hr, torch.zeros(1, R * nr, device=dev), "ref", Fl=1, Hl=Hr, Wl=Wr)
        cov = torch.cat([ref["cover"].new_zeros((1, M_MAX) + ref["cover"].shape[2:]), ref["cover"]], 1).to(dev)
        allow = REF.gate(cov, ref["slots"].long().to(dev))                                  # [1,R,(M+31)*HW]
        self.kv["ref_allow"] = {f: allow[:, :, (M_MAX + f) * HW:(M_MAX + f + 1) * HW].transpose(1, 2)[:, :, :, None].expand(1, HW, R, nr).reshape(1, HW, R * nr) for f in range(31)}

    def _pe(self, x):
        """the patch embedding WITHOUT attach_world's coordinate hook (it only fires for M+31 frames; added explicitly here)"""
        return self.m.patch_embedding(x)

    def _run(self, h, ts_tok, mode, Fl=1, Hl=15, Wl=26, frame=None, commit=False):
        m = self.m; dt = h.dtype; B, S = h.shape[:2]
        t = m.time_embedding(sinusoidal_embedding_1d(m.freq_dim, ts_tok.reshape(-1)).to(dt)).view(B, S, m.dim)
        t_mod = m.time_projection(t).unflatten(2, (6, m.dim)); self.kv["mode"] = mode
        info = (0, 0, 0, 0, 0)                                                              # ref not None -> shared-text cross attention, as training
        for li, blk in enumerate(m.blocks):
            h = blk(h, t_mod, self.ctx, m.rope, Fl, Hl, Wl, Hl * Wl, attn_mask=None, kv_cache=self.kv, layer_idx=li, frame_index=frame, commit=commit, ref=info)
        return h, t

    @torch.no_grad()
    def frame(self, x, cond, f, t, commit=False):
        """x [1,48,1,30,52] latent of video frame f (noisy at t, or clean with t = 0 to commit); cond [1,48,1,30,52] its condition latent
        -> velocity [1,48,1,30,52]"""
        from wizard.models.cond import cat_cond
        dt = self.ev.dtype; h = self._pe(cat_cond(x.to(dt), cond.to(dt))) + self.ev[:, :, f:f + 1]
        h = h.flatten(2).transpose(1, 2)
        h, tt = self._run(h, torch.full((1, h.shape[1]), float(t), device=h.device), "frame", frame=f, commit=commit)
        return self.m._unpatchify(self.m.head(h, tt), 1, 15, 26)
