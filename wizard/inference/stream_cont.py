"""continuous KV streaming of the causal model over any number of latent frames.
Same attention as wizard.inference.stream (history window W + sink, memory slots, references), with two changes:
  * K is cached BEFORE RoPE and re-rotated each call: up to frame 30 the positions are training's (M_MAX + f); past it the current frame sits at
    M_MAX + 30, history frame g at M_MAX + 30 - (f - g), the sink at M_MAX + g -- every distance stays inside what training saw
  * the coordinate embedding, the memory slots and their per-frame allow masks can be replaced mid-stream (world maintenance, new retrieval)"""
import torch
from wizard.models.dit import SelfAttention, sdpa, sinusoidal_embedding_1d
import wizard.models.ref as REF
from wizard.models.world_cond import M_MAX, HW

_ORIG = SelfAttention.forward
TRAIN_L = 31


def vpos(g, f, sink):
    if f < TRAIN_L or g < sink: return M_MAX + g
    return M_MAX + TRAIN_L - 1 - (f - g)


def _fwd(self, x, rope, Fl, Hl, Wl, attn_mask=None, kv_cache=None, layer_idx=None, frame_index=None, commit=False, tf=False, ref=None):
    if not (isinstance(kv_cache, dict) and kv_cache.get("kv_cont")):
        return _ORIG(self, x, rope, Fl, Hl, Wl, attn_mask, kv_cache, layer_idx, frame_index, commit, tf, ref)
    B, S, _ = x.shape; n, d = self.num_heads, self.head_dim; li = layer_idx; mode = kv_cache["mode"]
    q = self.norm_q(self.q(x)).view(B, S, n, d); k = self.norm_k(self.k(x)).view(B, S, n, d); v = self.v(x).view(B, S, n, d)
    if mode == "mem":
        hw = Hl * Wl; M = S // hw
        q = torch.cat([rope(q[:, j * hw:(j + 1) * hw], 1, Hl, Wl, f_start=j) for j in range(M)], 1)
        k = torch.cat([rope(k[:, j * hw:(j + 1) * hw], 1, Hl, Wl, f_start=j) for j in range(M)], 1)
        own = torch.arange(S, device=x.device) // hw
        o = sdpa(q, k, v, attn_mask=own[:, None] == own[None, :]).reshape(B, S, n * d)
        kv_cache["mk"][li] = k; kv_cache["mv"][li] = v; return self.o(o)
    if mode == "ref":
        R, Hr, Wr = kv_cache["ref_shape"]; nr = Hr * Wr
        q = torch.cat([rope(q[:, r * nr:(r + 1) * nr], 1, Hr, Wr, f_start=REF.REF_POS + r) for r in range(R)], 1)
        k = torch.cat([rope(k[:, r * nr:(r + 1) * nr], 1, Hr, Wr, f_start=REF.REF_POS + r) for r in range(R)], 1)
        own = torch.arange(S, device=x.device) // nr
        o = sdpa(q, k, v, attn_mask=own[:, None] == own[None, :]).reshape(B, S, n * d)
        kv_cache["rk"][li] = k; kv_cache["rv"][li] = v; return self.o(o)
    f = frame_index; W, sink = kv_cache["W"], kv_cache["sink"]; cache = kv_cache["vk"][li]
    k_raw = k
    q = rope(q, 1, Hl, Wl, f_start=vpos(f, f, sink)); k = rope(k, 1, Hl, Wl, f_start=vpos(f, f, sink))
    hist = [g for g in sorted(cache) if g < f and (g > f - W or g < sink)]
    ks = [rope(cache[g][0], 1, Hl, Wl, f_start=vpos(g, f, sink)) for g in hist] + [k]; vs = [cache[g][1] for g in hist] + [v]
    allow = [torch.ones(B, S, (len(hist) + 1) * S, dtype=torch.bool, device=x.device)]
    mk = kv_cache["mk"][li]
    if mk is not None:
        ma = kv_cache["mem_allow"].get(f)
        if ma is not None:
            ks.append(mk); vs.append(kv_cache["mv"][li]); allow.append(ma[:, None, :].expand(B, S, ma.shape[1]))
    rk = kv_cache["rk"][li]
    if rk is not None:
        ks.append(rk); vs.append(kv_cache["rv"][li]); allow.append(kv_cache["ref_allow"][f])
    o = sdpa(q, torch.cat(ks, 1), torch.cat(vs, 1), attn_mask=torch.cat(allow, 2)[:, None]).reshape(B, S, n * d)
    if commit:
        cache[f] = (k_raw.detach(), v.detach())
        for g in [g for g in cache if sink <= g <= f - W]: del cache[g]                     # out of the window for good
    return self.o(o)


SelfAttention.forward = _fwd


class ContStream:
    """model: causal WizardDiT with attach_cond + attach_world. gc [1,8,NF,60,104] coordinate inputs of the whole video; ref = dict(lat [1,R,48,1,H,W],
    slots [1,R], cover [1,L,15,26,3]). Memory is set with set_memory()."""

    @torch.no_grad()
    def __init__(self, model, ctx, gc, ref, W=20, sink=1):
        self.m = model; dev = gc.device; dt = next(model.parameters()).dtype; n = len(model.blocks); self.dt = dt
        self.kv = dict(kv_cont=True, mode=None, W=W, sink=sink, vk=[{} for _ in range(n)], mk=[None] * n, mv=[None] * n, rk=[None] * n, rv=[None] * n, mem_allow={})
        self.ctx = model.text_embedding(ctx.to(dt)); self.set_coords(gc)
        lat = ref["lat"].to(dev, dt); R = lat.shape[1]
        hr = self.m.patch_embedding(lat.flatten(0, 1)); Hr, Wr = hr.shape[3:]; nr = Hr * Wr; self.kv["ref_shape"] = (R, Hr, Wr)
        hr = hr.flatten(2).transpose(1, 2).reshape(1, R * nr, -1); self._run(hr, torch.zeros(1, R * nr, device=dev), "ref", Fl=1, Hl=Hr, Wl=Wr)
        allow = REF.gate(ref["cover"].to(dev), ref["slots"].long().to(dev)); L = ref["cover"].shape[1]   # [1,R,L*HW]
        self.kv["ref_allow"] = {f: allow[:, :, f * HW:(f + 1) * HW].transpose(1, 2)[:, :, :, None].expand(1, HW, R, nr).reshape(1, HW, R * nr) for f in range(L)}

    @torch.no_grad()
    def set_coords(self, gc):
        with torch.autocast("cuda", dtype=torch.bfloat16): self.ev = self.m.gbuf_embedding(gc).to(self.dt)                                   # [1,dim,L,15,26]

    @torch.no_grad()
    def set_memory(self, mem_lat, gm, allow):
        """mem_lat [1,48,M,30,52], gm [1,M,8,60,104]; allow {latent frame f: [1, M*HW] bool} for the frames about to be made"""
        from wizard.models.cond import cat_cond
        dev = mem_lat.device; dt = self.dt
        em = self.m.gbuf_embedding(gm.permute(0, 2, 1, 3, 4).to(dev), single=True).to(dt)
        hm = self.m.patch_embedding(cat_cond(mem_lat.to(dt), torch.zeros_like(mem_lat).to(dt))) + em
        h = hm.flatten(2).transpose(1, 2); self._run(h, torch.zeros(1, h.shape[1], device=dev), "mem", Fl=M_MAX)
        self.kv["mem_allow"] = allow

    def _run(self, h, ts_tok, mode, Fl=1, Hl=15, Wl=26, frame=None, commit=False):
        m = self.m; dt = h.dtype; B, S = h.shape[:2]
        t = m.time_embedding(sinusoidal_embedding_1d(m.freq_dim, ts_tok.reshape(-1)).to(dt)).view(B, S, m.dim)
        t_mod = m.time_projection(t).unflatten(2, (6, m.dim)); self.kv["mode"] = mode
        for li, blk in enumerate(m.blocks):
            h = blk(h, t_mod, self.ctx, m.rope, Fl, Hl, Wl, Hl * Wl, attn_mask=None, kv_cache=self.kv, layer_idx=li, frame_index=frame, commit=commit, ref=(0, 0, 0, 0, 0))
        return h, t

    @torch.no_grad()
    def frame(self, x, cond, f, t, commit=False):
        from wizard.models.cond import cat_cond
        dt = self.dt; h = self.m.patch_embedding(cat_cond(x.to(dt), cond.to(dt))) + self.ev[:, :, f:f + 1]
        h = h.flatten(2).transpose(1, 2)
        h, tt = self._run(h, torch.full((1, h.shape[1]), float(t), device=h.device), "frame", frame=f, commit=commit)
        return self.m._unpatchify(self.m.head(h, tt), 1, 15, 26)
