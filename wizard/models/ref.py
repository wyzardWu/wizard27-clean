"""Reference images bound to characters by a slot gate.

Sequence: [video tokens | reference 0 | reference 1]; each reference is one clean latent frame through the model's own
patch embedding, at timestep 0 and temporal RoPE position REF_POS + r. Attention (True = attend):
  video -> video            everything
  video -> reference r      only for video tokens inside the dilated box of the character that owns reference r
                            (slots: 1 player, 2 boss A, 3 boss B; 0 = nobody), in that token's own latent frame
  reference r -> itself     only, so its keys / values can be computed once and cached when the model runs causally
The head reads the video tokens only.

ref = dict(lat [B,R,48,1,H,W], slots [B,R] long, cover [B,Fl,Hl,Wl,K] (token cover of slot k+1))"""
import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

REF_POS = 900          # temporal RoPE position of the first reference (video frames use 0..Fl-1; the sliding window caps them at 16)
COVER_THR = 0.02       # a slot covers a token above this share
DILATE = 0.25          # box dilation per side


def gate(cover, slots, thr=COVER_THR, dilate=DILATE):
    """cover [B,F,H,W,K], slots [B,R] -> allow [B,R,F*H*W] bool: video tokens that may read reference r."""
    B, F_, H_, W_, K = cover.shape; dev = cover.device; occ = cover.float() > thr
    hh = torch.arange(H_, device=dev); ww = torch.arange(W_, device=dev); big = 10 ** 6
    allow = torch.zeros(B, K, F_, H_, W_, dtype=torch.bool, device=dev)
    for k in range(K):
        o = occ[..., k]; rows = o.any(3); cols = o.any(2)                                    # [B,F,H], [B,F,W]
        r0 = torch.where(rows, hh.expand_as(rows), big).min(-1).values; r1 = torch.where(rows, hh.expand_as(rows), -big).max(-1).values
        c0 = torch.where(cols, ww.expand_as(cols), big).min(-1).values; c1 = torch.where(cols, ww.expand_as(cols), -big).max(-1).values
        dh = ((r1 - r0 + 1).clamp(min=0) * dilate).long() + 1; dw = ((c1 - c0 + 1).clamp(min=0) * dilate).long() + 1
        r0, r1, c0, c1 = r0 - dh, r1 + dh, c0 - dw, c1 + dw                                    # a frame without the figure: r0 > r1, stays empty
        allow[:, k] = ((hh.view(1, 1, H_, 1) >= r0[..., None, None]) & (hh.view(1, 1, H_, 1) <= r1[..., None, None])
                       & (ww.view(1, 1, 1, W_) >= c0[..., None, None]) & (ww.view(1, 1, 1, W_) <= c1[..., None, None]))
    allow = allow.reshape(B, K, -1); bi = torch.arange(B, device=dev)
    return torch.stack([allow[bi, (slots[:, r] - 1).clamp(min=0)] & (slots[:, r] > 0)[:, None] for r in range(slots.shape[1])], 1)


def ref_mask(allow, nr):
    """allow [B,R,Sv] -> [B,1,S,S] bool for the sequence [video Sv | R x nr]."""
    B, R, Sv = allow.shape; S = Sv + R * nr; m = torch.zeros(B, 1, S, S, dtype=torch.bool, device=allow.device)
    m[:, :, :Sv, :Sv] = True
    for r in range(R):
        a = Sv + r * nr; m[:, 0, :Sv, a:a + nr] = allow[:, r][:, :, None]; m[:, :, a:a + nr, a:a + nr] = True
    return m


def forward_ref(model, x, timestep, context, ref):
    from wizard.models.dit import sinusoidal_embedding_1d
    assert context.dim() == 3, "reference path: one shared text per clip"
    B = x.shape[0]; h = model.patch_embedding(x); Fl, Hl, Wl = h.shape[2:]; hw = Hl * Wl; Sv = Fl * hw
    h = h.flatten(2).transpose(1, 2)
    lat = ref["lat"].to(h.dtype); R = lat.shape[1]
    hr = model.patch_embedding(lat.flatten(0, 1)); Hr, Wr = hr.shape[3:]; nr = Hr * Wr              # 48 channels: the official path, no condition
    hr = hr.flatten(2).transpose(1, 2).reshape(B, R * nr, -1)
    h = torch.cat([h, hr], 1); S = h.shape[1]
    ts = timestep if timestep.ndim == 2 else timestep[:, None].expand(B, Fl)
    ts_tok = torch.cat([ts.unsqueeze(-1).expand(B, Fl, hw).reshape(B, Sv), ts.new_zeros(B, R * nr)], 1)        # references are clean
    t = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, ts_tok.reshape(-1)).to(h.dtype)).view(B, S, model.dim)
    t_mod = model.time_projection(t).unflatten(2, (6, model.dim))
    ctx = model.text_embedding(context)
    mask = ref_mask(gate(ref["cover"], ref["slots"].long()), nr); info = (Sv, R, Hr, Wr, REF_POS)
    for blk in model.blocks:
        if model.use_gradient_checkpointing and model.training:
            h = torch_checkpoint(blk, h, t_mod, ctx, model.rope, Fl, Hl, Wl, hw, mask, use_reentrant=False, ref=info)
        else:
            h = blk(h, t_mod, ctx, model.rope, Fl, Hl, Wl, hw, mask, ref=info)
    return model._unpatchify(model.head(h[:, :Sv], t[:, :Sv]), Fl, Hl, Wl)


def tf_mask(Fl, hw, dev, W=None, sink=0):
    """The teacher-forcing mask of WizardDiT.forward over the stream [clean F | noisy F] (the same rules, kept in step with it): a clean frame
    reads the clean frames up to itself; a noisy frame reads the clean frames strictly before it (inside the training window W + sink, if set) and itself."""
    S = Fl * hw; tok = torch.arange(2 * S, device=dev); cln = tok < S; fr = torch.where(cln, tok // hw, (tok - S) // hw)
    cq, ck = cln[:, None], cln[None, :]; fq, fk = fr[:, None], fr[None, :]
    nctx = (~cq) & ck & (fk < fq)
    if W:
        win = fk > (fq - W)
        if sink: win = win | (fk < sink)
        nctx = nctx & win
    return (tok[:, None] == tok[None, :]) | (cq & ck & (fk <= fq)) | ((~cq) & (~ck) & (fq == fk)) | nctx


def forward_ref_tf(model, x, timestep, context, clean_x, ref, aug_t=None):
    """Causal TEACHER FORCING with gated reference images: the stream is [clean F | noisy F | R references]. The video halves keep the
    teacher-forcing mask; a video token of EITHER half reads reference r only inside its figure's box in its own latent frame (the gate of the
    bidirectional stage); a reference reads itself only -- so at rollout its keys / values are computed once and every frame pass reads them."""
    from wizard.models.dit import sinusoidal_embedding_1d
    assert context.dim() == 3, "reference path: one shared text per clip"
    B = x.shape[0]; h = model.patch_embedding(x); Fl, Hl, Wl = h.shape[2:]; hw = Hl * Wl; S = Fl * hw
    h = h.flatten(2).transpose(1, 2); hc = model.patch_embedding(clean_x).flatten(2).transpose(1, 2); assert hc.shape == h.shape
    lat = ref["lat"].to(h.dtype); R = lat.shape[1]
    hr = model.patch_embedding(lat.flatten(0, 1)); Hr, Wr = hr.shape[3:]; nr = Hr * Wr
    hr = hr.flatten(2).transpose(1, 2).reshape(B, R * nr, -1)
    h = torch.cat([hc, h, hr], 1); St = h.shape[1]
    tsn = timestep if timestep.ndim == 2 else timestep[:, None].expand(B, Fl)
    tsc = torch.zeros_like(tsn) if aug_t is None else (aug_t if aug_t.ndim == 2 else aug_t[:, None].expand(B, Fl))
    tok = lambda ts: ts.unsqueeze(-1).expand(B, Fl, hw).reshape(B, S)
    ts_tok = torch.cat([tok(tsc), tok(tsn), tsn.new_zeros(B, R * nr)], 1)                                      # clean half and references: t = 0
    t = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, ts_tok.reshape(-1)).to(h.dtype)).view(B, St, model.dim)
    t_mod = model.time_projection(t).unflatten(2, (6, model.dim))
    ctx = model.text_embedding(context)
    allow = gate(ref["cover"], ref["slots"].long())                                                            # [B, R, S]
    mask = torch.zeros(B, 1, St, St, dtype=torch.bool, device=h.device)
    mask[:, 0, :2 * S, :2 * S] = tf_mask(Fl, hw, h.device, getattr(model, "train_local_attn_size", None), getattr(model, "train_sink_size", 0) or 0)
    for r in range(R):
        a = 2 * S + r * nr; mask[:, 0, :S, a:a + nr] = allow[:, r][:, :, None]; mask[:, 0, S:2 * S, a:a + nr] = allow[:, r][:, :, None]; mask[:, 0, a:a + nr, a:a + nr] = True
    info = (2 * S, R, Hr, Wr, REF_POS)
    for blk in model.blocks:
        if model.use_gradient_checkpointing and model.training:
            h = torch_checkpoint(blk, h, t_mod, ctx, model.rope, Fl, Hl, Wl, hw, mask, use_reentrant=False, tf=True, ref=info)
        else:
            h = blk(h, t_mod, ctx, model.rope, Fl, Hl, Wl, hw, mask, tf=True, ref=info)
    return model._unpatchify(model.head(h[:, S:2 * S], t[:, S:2 * S]), Fl, Hl, Wl)


@torch.no_grad()
def ref_cache(model, kv_cache, ref_lat, context):
    """Streaming rollout with reference images: run the R references ALONE through the blocks once (timestep 0, their own positions,
    the clip's text) and keep their keys / values per layer in the KV cache -- exactly what they are in forward_ref_tf, where a reference reads
    itself only. ref_lat [B,R,48,1,H,W]; context = the text this cache rolls out with (one cache per text when guidance is used)."""
    from wizard.models.dit import sinusoidal_embedding_1d
    B, R = ref_lat.shape[:2]; hr = model.patch_embedding(ref_lat.flatten(0, 1)); Hr, Wr = hr.shape[3:]; nr = Hr * Wr
    h = hr.flatten(2).transpose(1, 2).reshape(B, R * nr, -1)
    t = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, torch.zeros(B, device=h.device)).to(h.dtype)); t_mod = model.time_projection(t).unflatten(1, (6, model.dim))
    ctx = model.text_embedding(context); n = len(model.blocks); kv_cache.update(ref_k=[None] * n, ref_v=[None] * n, ref_capture=True, ref_n=(R, nr))
    for li, blk in enumerate(model.blocks):
        h = blk(h, t_mod, ctx, model.rope, 1, Hr, Wr, nr, attn_mask=None, kv_cache=kv_cache, layer_idx=li, frame_index=0, commit=False, ref=(0, R, Hr, Wr, REF_POS))
    kv_cache["ref_capture"] = False


def ref_allow(kv_cache, cover_k, slots):
    """Set the gate of ONE latent frame before its forward_frame passes: cover_k [B,1,Hl,Wl,K] (that frame's token cover), slots [B,R]."""
    a = gate(cover_k, slots.long()); B, R, hw = a.shape; nr = kv_cache["ref_n"][1]                # [B, R, hw]
    kv_cache["ref_allow"] = a.transpose(1, 2)[:, :, :, None].expand(B, hw, R, nr).reshape(B, hw, R * nr)
