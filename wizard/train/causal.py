"""Stage 2: the causal (autoregressive) teacher, trained with teacher forcing from the bidirectional checkpoint.

The sequence is the bidirectional layout twice: [memory | clean history] and [memory | noisy frames]; every noisy latent frame
attends to the clean frames before it inside a sliding history window plus an attention sink (frame 0), to its chunk's memory and
to the reference images. `forward_tf` is the forward that inference (`wizard.inference.stream`) reproduces with a KV cache.

    accelerate launch --config_file configs/accelerate/fsdp.yaml --num_processes 8 -m wizard.train.causal \
        --data $WIZARD_DATA/wan_v7 --init <bidirectional dit> --out runs/causal [--grad_accum N]"""
import os, sys, time, json, random, argparse, collections
import torch

import wizard.train.bidirectional as T
from wizard.train.bidirectional import ClipDataset, collate, lt, DS, SFLY
from wizard.models.world_cond import attach_world, scene_gbuf, ident_aug, M_MAX, CH, HW
from wizard.models.dit import WizardDiT, ti2v_5b_config, tiny_config, sinusoidal_embedding_1d
from wizard.models.flow_match import FlowMatchScheduler
from wizard.models.cond import attach_cond, cat_cond
import wizard.models.ref as REF
from torch.utils.checkpoint import checkpoint as torch_checkpoint

GAP_F = 90                                                                                  # wizard.memory.retrieve.GAP = 3 s at 30 fps


def tf_mask(use, hide, allow, R, nr, W, sink, dev):
    """use [B,8,M] chunk -> memory slot; hide [B,M,HW]; allow [B,R,S1] (gate over one half: memory rows False) -> [B,1,St,St] bool"""
    B = use.shape[0]; S1 = (M_MAX + 31) * HW; S2 = 2 * S1; St = S2 + R * nr
    fl = torch.full((31,), -1, dtype=torch.long)
    for c, lats in enumerate(CH):
        for j in lats: fl[j] = c
    i = torch.arange(S2, device=dev); half = i // S1; ft = (i % S1) // HW; is_m = ft < M_MAX; vf = (ft - M_MAX).clamp(min=0); ch = fl.to(dev)[vf]
    hq, hk = half[:, None], half[None, :]; mq, mk = is_m[:, None], is_m[None, :]; fq, fk = vf[:, None], vf[None, :]
    mem_self = mq & mk & (hq == hk) & (ft[:, None] == ft[None, :])
    win = (fk > fq - W) if W else torch.ones(S2, S2, dtype=torch.bool, device=dev)
    if sink: win = win | (fk < sink)
    clean = (hq == 0) & (hk == 0) & ~mq & ~mk & (fk <= fq) & win                          # the clean history is windowed too: the rollout's cached frames were
    nself = (hq == 1) & (hk == 1) & ~mq & ~mk & (fk == fq)                                 # made reading their own window only
    nctx = (hq == 1) & (hk == 0) & ~mq & ~mk & (fk < fq) & win
    base = mem_self | clean | nself | nctx                                                 # [S2,S2], same for every sample
    m = torch.zeros(B, 1, St, St, dtype=torch.bool, device=dev)
    kslot = ft.clamp(max=M_MAX - 1)                                                        # memory slot of an A-half memory key
    for b in range(B):
        rd = use[b].to(dev)[ch][:, kslot]                                                  # [S2,S2]: query chunk -> memory slot of the key
        vis = (~hide[b].to(dev)).reshape(-1)                                               # [M*HW]
        kvis = torch.zeros(S2, dtype=torch.bool, device=dev); kvis[:M_MAX * HW] = vis      # only the A copy is readable
        rd = rd & (~mq) & mk & (hk == 0) & kvis[None, :]
        m[b, 0, :S2, :S2] = base | rd
        for r in range(R):
            a = S2 + r * nr; al = allow[b, r]
            m[b, 0, :S1, a:a + nr] = al[:, None]; m[b, 0, S1:S2, a:a + nr] = al[:, None]; m[b, 0, a:a + nr, a:a + nr] = True
    return m


def forward_tf(model, x, timestep, context, clean_x, ref, aug_t=None):
    """x / clean_x: cat_cond of [memory | noisy video] / [memory | clean video] (M_MAX + 31 frames); timestep [B,31] (noisy video).
    Installed as wizard.models.ref.forward_ref_tf, so model.forward(..., clean_x=, ref=) runs it under FSDP. -> velocity [B,C,31,H,W]"""
    B = x.shape[0]
    ha = model.patch_embedding(clean_x); Fl, Hl, Wl = ha.shape[2:]; hw = Hl * Wl; S1 = Fl * hw; assert Fl == M_MAX + 31 and hw == HW
    hb = model.patch_embedding(x)
    ha = ha.flatten(2).transpose(1, 2); hb = hb.flatten(2).transpose(1, 2)
    lat = ref["lat"].to(ha.dtype); R = lat.shape[1]
    hr = model.patch_embedding(lat.flatten(0, 1)); Hr, Wr = hr.shape[3:]; nr = Hr * Wr
    hr = hr.flatten(2).transpose(1, 2).reshape(B, R * nr, -1)
    h = torch.cat([ha, hb, hr], 1); St = h.shape[1]
    tsn = timestep if timestep.ndim == 2 else timestep[:, None].expand(B, 31)
    tok = lambda ts: ts.unsqueeze(-1).expand(B, ts.shape[1], hw).reshape(B, -1)
    zm = tsn.new_zeros(B, M_MAX * hw)
    ts_tok = torch.cat([zm, tsn.new_zeros(B, 31 * hw), zm, tok(tsn), tsn.new_zeros(B, R * nr)], 1)    # memory / clean / references: t = 0
    t = model.time_embedding(sinusoidal_embedding_1d(model.freq_dim, ts_tok.reshape(-1)).to(h.dtype)).view(B, St, model.dim)
    t_mod = model.time_projection(t).unflatten(2, (6, model.dim)); ctx = model.text_embedding(context)
    cov = torch.cat([ref["cover"].new_zeros((B, M_MAX) + ref["cover"].shape[2:]), ref["cover"]], 1)       # memory frames are nobody's figure
    allow = REF.gate(cov, ref["slots"].long())                                                              # [B,R,S1]
    W = getattr(model, "train_local_attn_size", 0) or 0; sink = getattr(model, "train_sink_size", 0) or 0
    mask = tf_mask(ref["mem_use"], ref["mem_hide"], allow, R, nr, W, sink, h.device)
    info = (2 * S1, R, Hr, Wr, REF.REF_POS)
    for blk in model.blocks:
        if model.use_gradient_checkpointing and model.training:
            h = torch_checkpoint(blk, h, t_mod, ctx, model.rope, Fl, Hl, Wl, hw, mask, use_reentrant=False, tf=True, ref=info)
        else:
            h = blk(h, t_mod, ctx, model.rope, Fl, Hl, Wl, hw, mask, tf=True, ref=info)
    a = S1 + M_MAX * hw
    return model._unpatchify(model.head(h[:, a:2 * S1], t[:, a:2 * S1]), 31, Hl, Wl)


class ARData(ClipDataset):
    """wizard.train.bidirectional's data with PAST-only memory candidates (the rollout only has frames it already made)."""
    def __init__(self, root, rows, a):
        super().__init__(root, rows, a)
        info = json.load(open(f"{SFLY}/index.json")) if os.path.exists(f"{SFLY}/index.json") else {}
        sf = lt(f"{SFLY}/mem_index.pt") if os.path.exists(f"{SFLY}/mem_index.pt") else {}
        n0 = n1 = 0
        for name in sf:                                                                   # SceneFly forward windows: drop picks from after the window
            if name not in info or name not in self.mem_index: continue
            lim = int(info[name]["f0"]) - GAP_F; e = dict(self.mem_index[name])
            ch = [[(f, g) for f, g in c if f <= lim] for c in e["chunks"]]
            n0 += sum(len(c) for c in e["chunks"]); n1 += sum(len(c) for c in ch); e["chunks"] = ch; self.mem_index[name] = e
        self.past_info = (n0, n1)


def loss_fn(model, core, sched, b, neg, a, dev):
    lat = b["latent"].to(dev).float(); act = b["actor"].to(dev).float(); mem = b["mem_lat"].to(dev).float()[:, :, :, 0].permute(0, 2, 1, 3, 4)
    if random.random() < a.p_memnoise: am = random.uniform(0, a.memnoise); mem = (1 - am) * mem + am * torch.randn_like(mem)
    B, C, Fl = lat.shape[:3]; assert B == 1
    idx = sched.sample_pf_timestep_ids(B, Fl, dev, ar_order_bias=0.0); noise = torch.randn_like(lat)
    noisy = sched.add_noise_pf(lat, noise, idx); ts = sched.timesteps.to(dev, torch.float32)[idx].clone()
    noisy[:, :, 0] = lat[:, :, 0]; ts[:, 0] = 0.0                                          # first frame always given
    gc = scene_gbuf(b["depth"][0].to(dev), b["pos"][0].to(dev), b["R"][0].to(dev), b["K"][0], b["origin"][0].to(dev), float(b["hp"][0]), None)
    tier_c = b["unknown"][0].to(dev).float()
    n = int(b["mem_n"][0]); gm = torch.zeros(M_MAX, 8, 60, 104, device=dev)
    for j in range(n):
        gm[j] = scene_gbuf(b["mem_depth"][0, j:j + 1].to(dev), b["mem_pos"][0, j:j + 1].to(dev), b["mem_R"][0, j:j + 1].to(dev), b["mem_K"][0, j], b["origin"][0].to(dev), float(b["hp"][0]))[:, 0]
    if random.random() < a.p_ident:
        aug = ident_aug([gc] + [gm[j:j + 1].permute(1, 0, 2, 3) for j in range(n)]); gc = aug[0]
        for j in range(n): gm[j] = aug[1 + j][:, 0]
    tier_m = b["mem_tier"][0].to(dev).float() / 3.0
    gc = torch.cat([gc[:3], (gc[3:4] > 0).to(gc.dtype), torch.zeros_like(gc[4:7]), tier_c[None].to(gc.dtype)], 0)
    gm = torch.cat([gm[:, :3], (gm[:, 3:4] > 0).to(gm.dtype), torch.zeros_like(gm[:, 4:7]), tier_m[:, None].to(gm.dtype)], 1)
    core._gb_in = (gc[None], gm[None]); core._mem_warp = None; core._mem_wrope = None
    ctx = b["context"].to(dev).to(torch.bfloat16)
    if random.random() < a.p_text_drop: ctx = neg.to(dev).to(torch.bfloat16)
    cond = torch.cat([torch.zeros_like(mem), act], 2)
    xn = cat_cond(torch.cat([mem, noisy], 2), cond); xc = cat_cond(torch.cat([mem, lat], 2), cond)
    ref = dict(lat=b["ref_lat"].to(dev), slots=b["ref_slots"].to(dev), cover=b["cover"].to(dev), mem_use=b["mem_use"], mem_n=b["mem_n"], mem_hide=b["mem_hide"])
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        v = model(xn, ts, ctx, clean_x=xc, ref=ref)
    core._gb_in = None
    se = ((v.float() - (noise - lat)).pow(2) * b["memw"].to(dev).float()[:, None]).mean(dim=(1, 3, 4)); keep = torch.ones_like(se); keep[:, 0] = 0.0
    w = sched.training_weight(idx).float(); return (w * se * keep).sum() / keep.sum().clamp(min=1.0)


def build_model(a, dry=False):
    model = WizardDiT(**(tiny_config() if dry else ti2v_5b_config()), causal=True, zero_init_head=False, train_local_attn_size=a.window, train_sink_size=a.sink)
    attach_cond(model); attach_world(model); REF.forward_ref_tf = forward_tf; model.use_gradient_checkpointing = True
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--init", required=True); ap.add_argument("--step0", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--lr_cond", type=float, default=1e-5)
    ap.add_argument("--max_steps", type=int, default=10000); ap.add_argument("--warmup", type=int, default=200); ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=20); ap.add_argument("--num_workers", type=int, default=6); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--window", type=int, default=20); ap.add_argument("--sink", type=int, default=1)
    ap.add_argument("--p_irrel", type=float, default=0.2); ap.add_argument("--p_nomem", type=float, default=0.1); ap.add_argument("--p_text_drop", type=float, default=0.1)
    ap.add_argument("--p_ident", type=float, default=0.3); ap.add_argument("--p_memnoise", type=float, default=0.5); ap.add_argument("--memnoise", type=float, default=0.15)
    ap.add_argument("--group_rep", default="e1o:2,e2o:2,e1n:2,e2n:2,e1x:2,turn:2,roam:2"); ap.add_argument("--rep_mem", type=int, default=0); ap.add_argument("--p_memunk", type=float, default=0.0)
    ap.add_argument("--mem_loss_w", type=float, default=1.0); ap.add_argument("--extra_rows", default=""); ap.add_argument("--mem_index", default="mem_index.pt"); ap.add_argument("--rep_keys", default="")
    ap.add_argument("--p_trueunk", type=float, default=0.0); ap.add_argument("--p_unk2", type=float, default=0.0); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--grad_accum", type=int, default=1)
    a = ap.parse_args(); a.mem_warp = False; a.mem_wrope = False; a.no_coords = False; a.empty_actor = False; a.skeleton_depth_cond = False; a.coords_only = False
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    accel = Accelerator(); set_seed(a.seed, device_specific=True); random.seed(a.seed + accel.process_index)
    main_ = accel.is_main_process; os.makedirs(a.out, exist_ok=True)
    log = (lambda *x: (print(*x, flush=True), open(os.path.join(a.out, "train.log"), "a").write(" ".join(map(str, x)) + "\n"))) if main_ else (lambda *x: None)
    model = build_model(a); core = model
    import safetensors.torch as st
    info = model.load_state_dict(st.load_file(a.init), strict=False)
    bad = [k for k in info.missing_keys if "rope" not in k]; assert not bad and not info.unexpected_keys, (bad[:5], info.unexpected_keys[:5])
    log(f"AR teacher forcing <- {a.init} (step {a.step0}); history window {a.window} + sink {a.sink}; memory = past only")
    rows = [json.loads(l) for l in open(os.path.join(a.data, "v7_rows.jsonl"))]
    for er in [x for x in a.extra_rows.split(",") if x]: rows += [json.loads(l) for l in open(er)]
    if a.limit: rows = rows[:a.limit] + [r for r in rows if r["group"] == "roam"][:a.limit]
    rep = {k: int(v) for k, v in (x.split(":") for x in a.group_rep.split(","))}
    rk = set(json.load(open(a.rep_keys))) if a.rep_keys else None
    nrep = lambda r: a.rep_mem if (a.rep_mem and r["group"] == "roam" and rk is not None and r["key"] in rk) else rep.get(r["group"], 1)
    train = [r for r in rows for _ in range(nrep(r))]
    c = collections.Counter(r["group"] for r in train); log(f"rows {len(rows)} -> {len(train)}:", {k: f"{v} ({100 * v / len(train):.1f}%)" for k, v in sorted(c.items())})
    ds = ARData(a.data, train, a); log(f"SceneFly memory picks kept past-only: {ds.past_info[1]} of {ds.past_info[0]}")
    if main_: json.dump({"args": vars(a), "rows": len(rows)}, open(os.path.join(a.out, f"args_{a.step0}.json"), "w"), indent=1)
    neg = lt(f"{DS}/wan_mix5/neg_emb.pt")["neg_emb"]; neg = torch.cat([neg, neg.new_zeros(512 - neg.shape[0], neg.shape[1])])[None]
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=a.num_workers, collate_fn=collate, drop_last=True, pin_memory=True, persistent_workers=True)
    hi = ("cond_embedding", "gbuf_embedding")
    opt = torch.optim.AdamW([dict(params=[p for n_, p in model.named_parameters() if not n_.startswith(hi)], lr=a.lr),
                             dict(params=[p for n_, p in model.named_parameters() if n_.startswith(hi)], lr=a.lr_cond)], weight_decay=0.01, betas=(0.9, 0.999))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))
    model, opt, dl, sch = accel.prepare(model, opt, dl, sch)
    sched = FlowMatchScheduler(shift=5.0).to(accel.device)
    step, t0, it = a.step0, time.time(), iter(dl); model.train()
    while step < a.max_steps:
        try: b = next(it)
        except StopIteration: it = iter(dl); b = next(it)
        loss = loss_fn(model, core, sched, b, neg, a, accel.device)
        if a.grad_accum > 1:
            accel.backward(loss / a.grad_accum)
            for _ in range(a.grad_accum - 2):
                try: b = next(it)
                except StopIteration: it = iter(dl); b = next(it)
                accel.backward(loss_fn(model, core, sched, b, neg, a, accel.device) / a.grad_accum)
            try: b = next(it)
            except StopIteration: it = iter(dl); b = next(it)
            loss = loss_fn(model, core, sched, b, neg, a, accel.device); accel.backward(loss / a.grad_accum)
        else: accel.backward(loss)
        gn = accel.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):
            print(f"NONFINITE_GRAD rank {accel.process_index} step {step} loss {float(loss):.4f}", flush=True); opt.zero_grad(); step += 1; sch.step(); continue
        opt.step(); opt.zero_grad(); sch.step(); step += 1
        if step % a.log_every == 0:
            l = accel.gather(loss.detach()).mean().item(); sps = a.log_every / (time.time() - t0); t0 = time.time()
            log(f"step {step:6d}/{a.max_steps} loss {l:.4f} gnorm {float(gn):.3f} {1 / max(sps, 1e-9):.2f} s/step")
        if step % a.save_every == 0 or step == a.max_steps:
            sd = accel.get_state_dict(model)
            if main_:
                os.makedirs(os.path.join(a.out, "dits"), exist_ok=True)
                st.save_file({k: v.contiguous().to(torch.bfloat16) for k, v in sd.items()}, os.path.join(a.out, "dits", f"dit_step{step}.safetensors")); log(f"saved step {step}")
                old = step - 2 * a.save_every
                if old > 0 and old % 2000: [os.remove(f) for f in [os.path.join(a.out, "dits", f"dit_step{old}.safetensors")] if os.path.exists(f)]
            accel.wait_for_everyone()
    log("AR_TF_DONE")


if __name__ == "__main__":
    main()
