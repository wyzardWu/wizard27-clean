"""diag: how much attention the video tokens put on the memory tokens. Wraps wizard.models.dit.sdpa; while ON, for every self-attention
call (q and k the same length) samples queries per chunk and records, per layer, the softmax mass on memory keys (and the share of memory
among the keys the mask allows = what uniform attention would give)."""
import os, json, torch
import wizard.models.dit as GD
ORIG = GD.sdpa; ON = False; LAYER = 0; REC = []; CTX = {}

def wrap(q, k, v, attn_mask=None):
    global LAYER
    o = ORIG(q, k, v, attn_mask)
    ext = ON and attn_mask is not None and k.shape[1] == q.shape[1] + CTX.get('M', 0) * CTX.get('HW', 0)   # warped memory keys appended
    if ON and attn_mask is not None and (q.shape[1] == k.shape[1] or ext):
        M, HW, CH = CTX["M"], CTX["HW"], CTX["CH"]; nm = M * HW; g = torch.Generator(device="cpu").manual_seed(0)
        for c, lats in enumerate(CH):
            toks = torch.cat([torch.arange((M + j) * HW, (M + j + 1) * HW) for j in lats]); idx = toks[torch.randperm(len(toks), generator=g)[:96]].to(q.device)
            am = attn_mask[0, 0, idx]; bias = am.float() if am.dtype != torch.bool else None; mk = torch.isfinite(bias) if bias is not None else am   # float mask = bias
            memcol = torch.zeros(mk.shape[1], dtype=torch.bool, device=mk.device); memcol[:nm] = True
            if mk.shape[1] > q.shape[1]: memcol[q.shape[1]:] = True
            if not mk[:, memcol].any(): continue
            qs = q[0, idx].float(); ks = k[0].float(); D = qs.shape[-1]; mass = 0.0
            for h in range(qs.shape[1]):
                sc = (qs[:, h] @ ks[:, h].T) * D ** -0.5; sc = (sc + bias) if bias is not None else sc.masked_fill(~mk, float("-inf")); p = sc.softmax(-1)
                mass += p[:, memcol].sum(-1).mean().item()
            per_slot = [mk[:, j * HW:(j + 1) * HW].any().item() for j in range(M)]
            if os.environ.get("ATTN_ALIGN"):                                                     # does a video token look at the SAME place in a memory frame?
                W_ = int(os.environ.get("ATTN_W", "26")); qp = ((idx - nm) % HW).cpu(); qy, qx = qp // W_, qp % W_; Hh = HW // W_
                al = {}
                for j in range(M):
                    if not per_slot[j]: continue
                    nb = torch.stack([j * HW + (qy + dy).clamp(0, Hh - 1) * W_ + (qx + dx).clamp(0, W_ - 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1)], 1).to(q.device)   # [nq, 9]
                    slot_m = 0.0; al_m = 0.0
                    for h in range(qs.shape[1]):
                        sc = (qs[:, h] @ ks[:, h].T) * D ** -0.5; sc = (sc + bias) if bias is not None else sc.masked_fill(~mk, float("-inf")); pr = sc.softmax(-1)
                        sm_ = pr[:, j * HW:(j + 1) * HW].sum(-1); am_ = pr.gather(1, nb).sum(-1)
                        ok_ = mk[:, j * HW:(j + 1) * HW].any(-1)
                        slot_m += sm_[ok_].mean().item() if ok_.any() else 0.0; al_m += (am_[ok_] / sm_[ok_].clamp(min=1e-9)).mean().item() if ok_.any() else 0.0
                    al[j] = (slot_m / qs.shape[1], al_m / qs.shape[1])
                CTX.setdefault("align", []).append(dict(step=CTX["step"], layer=LAYER, chunk=c, slots={str(k_): v_ for k_, v_ in al.items()}))
            REC.append(dict(step=CTX["step"], layer=LAYER, chunk=c, mem_mass=mass / qs.shape[1], uniform=(mk[:, memcol].sum(-1).float() / mk.sum(-1).float()).mean().item(), slots=per_slot))
        LAYER += 1
    return o

def install(): GD.sdpa = wrap
def start(step, M, HW, CH):
    global ON, LAYER; ON = True; LAYER = 0; CTX.update(step=step, M=M, HW=HW, CH=CH)
def stop(): 
    global ON; ON = False
def dump(path): json.dump(dict(rec=REC, align=CTX.get("align", [])) if os.environ.get("ATTN_ALIGN") else REC, open(path, "w"))


def record_wrope(rows, w1, w2, m2, m1):
    """world-RoPE split attention (wizard.models.world_cond.install_wrope_split): a video row's softmax mass on the world memory keys = w2 / (w1 + w2)
    exactly (attention 1 no longer lets video rows read memory). rows [r] video-token indices (0 = first video token); w1, w2 [B,r,n,1]
    m2 [B,1,r,Kw] allowed world keys, m1 [B,1,r,S] allowed ordinary keys. Recorded per chunk like wrap()."""
    global LAYER
    if not ON: return
    HW, CH = CTX["HW"], CTX["CH"]
    mass = (w2 / (w1 + w2))[0, :, :, 0].float().mean(-1)                                   # [r] mean over heads
    uni = m2[0, 0].float().sum(-1) / (m2[0, 0].float().sum(-1) + m1[0, 0].float().sum(-1)).clamp(min=1)
    fr = (rows // HW).cpu()
    for c, lats in enumerate(CH):
        sel = torch.isin(fr, torch.as_tensor(lats))
        if sel.any():
            s = sel.to(mass.device)
            REC.append(dict(step=CTX["step"], layer=LAYER, chunk=c, mem_mass=mass[s].mean().item(), uniform=uni[s].mean().item(), slots=[], wrope=True))
    LAYER += 1
