"""Stage 3: causal consistency distillation of the causal teacher into a 2-step student.

Discrete consistency axis (N levels); the teacher takes one Euler step t -> t_next with CFG; the student's x0(x_t) is regressed onto
the EMA's x0(x_t_next). Inputs (memory, G-buffer, references, history) are the teacher's.

    accelerate launch --config_file configs/accelerate/fsdp_dmd.yaml --num_processes 8 -m wizard.distill.consistency \
        --data $WIZARD_DATA/wan_v7 --ckpt <causal teacher dit> --out runs/cd [data arguments]"""
import os, sys, time, json, random, argparse, collections
import torch, torch.nn.functional as F

import wizard.train.causal as A
from wizard.train.bidirectional import collate, lt, DS
from wizard.models.world_cond import scene_gbuf, M_MAX
from wizard.models.flow_match import FlowMatchScheduler
from wizard.models.cond import cat_cond
from wizard.distill.common import cd_axis, noise_clip, _per_frame_t, _keep_frame0, ema_update
import safetensors.torch as st


def prep(b, dev):
    """wizard.train.causal.loss_fn's inputs, no augmentation -> (lat, cond, mem, gc, gm, ctx, ref)"""
    lat = b["latent"].to(dev).float(); act = b["actor"].to(dev).float(); mem = b["mem_lat"].to(dev).float()[:, :, :, 0].permute(0, 2, 1, 3, 4)
    gc = scene_gbuf(b["depth"][0].to(dev), b["pos"][0].to(dev), b["R"][0].to(dev), b["K"][0], b["origin"][0].to(dev), float(b["hp"][0]), None)
    n = int(b["mem_n"][0]); gm = torch.zeros(M_MAX, 8, 60, 104, device=dev)
    for j in range(n):
        gm[j] = scene_gbuf(b["mem_depth"][0, j:j + 1].to(dev), b["mem_pos"][0, j:j + 1].to(dev), b["mem_R"][0, j:j + 1].to(dev), b["mem_K"][0, j], b["origin"][0].to(dev), float(b["hp"][0]))[:, 0]
    tier_c = b["unknown"][0].to(dev).float(); tier_m = b["mem_tier"][0].to(dev).float() / 3.0
    gc = torch.cat([gc[:3], (gc[3:4] > 0).to(gc.dtype), torch.zeros_like(gc[4:7]), tier_c[None].to(gc.dtype)], 0)
    gm = torch.cat([gm[:, :3], (gm[:, 3:4] > 0).to(gm.dtype), torch.zeros_like(gm[:, 4:7]), tier_m[:, None].to(gm.dtype)], 1)
    cond = torch.cat([torch.zeros_like(mem), act], 2)
    ref = dict(lat=b["ref_lat"].to(dev), slots=b["ref_slots"].to(dev), cover=b["cover"].to(dev), mem_use=b["mem_use"], mem_n=b["mem_n"], mem_hide=b["mem_hide"])
    return lat, cond, mem, (gc[None], gm[None]), b["context"].to(dev).to(torch.bfloat16), ref


def cd_loss(models, cores, b, neg, sigmas, tsteps, idx, cfg, dev):
    student, ema, teacher = models
    lat, cond, mem, gb, ctx, ref = prep(b, dev)
    for c in cores: c._gb_in = gb; c._mem_warp = None; c._mem_wrope = None
    B, C, Fl = lat.shape[:3]
    run = lambda m, x, ts, cx_: m(cat_cond(torch.cat([mem, x], 2), cond).to(torch.bfloat16), ts, cx_, clean_x=xc, ref=ref).float()
    xc = cat_cond(torch.cat([mem, lat], 2), cond).to(torch.bfloat16)
    s_t, s_tn = sigmas[idx].view(1, 1, 1, 1, 1), sigmas[idx + 1].view(1, 1, 1, 1, 1)
    ts_t = _per_frame_t(float(tsteps[idx]), B, Fl, dev); ts_tn = _per_frame_t(float(tsteps[idx + 1]), B, Fl, dev)
    latent_t = noise_clip(lat, s_t, torch.randn_like(lat))
    with torch.no_grad():
        v = run(teacher, latent_t, ts_t, ctx)
        if abs(cfg - 1.0) > 1e-6: v_u = run(teacher, latent_t, ts_t, neg); v = v_u + cfg * (v - v_u)
        latent_tn = _keep_frame0(latent_t + (s_tn - s_t) * v, lat)
    cm_t = latent_t - s_t * run(student, latent_t, ts_t, ctx)
    with torch.no_grad(): cm_tn = latent_tn - s_tn * run(ema, latent_tn, ts_tn, ctx)
    for c in cores: c._gb_in = None
    return F.mse_loss(cm_t[:, :, 1:], cm_tn[:, :, 1:].detach()), float(sigmas[idx]), float(sigmas[idx + 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=48); ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-6); ap.add_argument("--max_steps", type=int, default=10000); ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--ema_decay", type=float, default=0.99); ap.add_argument("--ema_start", type=int, default=200)
    ap.add_argument("--window", type=int, default=20); ap.add_argument("--sink", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=500); ap.add_argument("--log_every", type=int, default=10); ap.add_argument("--num_workers", type=int, default=6); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p_irrel", type=float, default=0.2); ap.add_argument("--p_nomem", type=float, default=0.1)
    ap.add_argument("--group_rep", default="e1o:2,e2o:2,e1n:2,e2n:2,e1x:2,turn:2,roam:2"); ap.add_argument("--rep_mem", type=int, default=0); ap.add_argument("--rep_keys", default="")
    ap.add_argument("--extra_rows", default=""); ap.add_argument("--mem_index", default="mem_index.pt"); ap.add_argument("--p_trueunk", type=float, default=0.0); ap.add_argument("--p_unk2", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    a.mem_warp = a.mem_wrope = a.no_coords = a.empty_actor = a.skeleton_depth_cond = a.coords_only = False
    a.p_memunk = 0.0; a.mem_loss_w = 1.0; a.p_text_drop = 0.0; a.p_memnoise = 0.0; a.memnoise = 0.0; a.p_ident = 0.0
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    accel = Accelerator(step_scheduler_with_optimizer=False); set_seed(a.seed, device_specific=True); random.seed(a.seed + accel.process_index)
    main_ = accel.is_main_process; os.makedirs(a.out, exist_ok=True)
    log = (lambda *x: (print(*x, flush=True), open(os.path.join(a.out, "train.log"), "a").write(" ".join(map(str, x)) + "\n"))) if main_ else (lambda *x: None)
    models = []
    for _ in range(3):
        m = A.build_model(a); info = m.load_state_dict(st.load_file(a.ckpt), strict=False)
        bad = [k for k in info.missing_keys if "rope" not in k]; assert not bad and not info.unexpected_keys, (bad[:5], info.unexpected_keys[:5]); models.append(m)
    cores = list(models); student, ema, teacher = models; student.use_gradient_checkpointing = True; ema.use_gradient_checkpointing = False; teacher.use_gradient_checkpointing = False
    rows = [json.loads(l) for l in open(os.path.join(a.data, "v7_rows.jsonl"))]
    for er in [x for x in a.extra_rows.split(",") if x]: rows += [json.loads(l) for l in open(er)]
    if a.limit: rows = rows[:a.limit] + [r for r in rows if r["group"] == "roam"][:a.limit]
    rep = {k: int(v) for k, v in (x.split(":") for x in a.group_rep.split(","))}
    rk = set(json.load(open(a.rep_keys))) if a.rep_keys else None
    nrep = lambda r: a.rep_mem if (a.rep_mem and r["group"] == "roam" and rk is not None and r["key"] in rk) else rep.get(r["group"], 1)
    train = [r for r in rows for _ in range(nrep(r))]
    c = collections.Counter(r["group"] for r in train)
    log(f"CD v7 (teacher cfg {a.cfg}, all window {a.window}+{a.sink}) <- {a.ckpt}; world {accel.num_processes}; N {a.N} lr {a.lr}")
    log(f"rows {len(rows)} -> {len(train)}:", {k: f"{v} ({100 * v / len(train):.1f}%)" for k, v in sorted(c.items())})
    ds = A.ARData(a.data, train, a); log(f"SceneFly memory picks kept past-only: {ds.past_info[1]} of {ds.past_info[0]}")
    if main_: json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    neg = lt(f"{DS}/wan_mix5/neg_emb.pt")["neg_emb"]; neg = torch.cat([neg, neg.new_zeros(512 - neg.shape[0], neg.shape[1])])[None].to(accel.device, torch.bfloat16)
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=a.num_workers, collate_fn=collate, drop_last=True, pin_memory=True, persistent_workers=True)
    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(student.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.0, 0.999))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / a.warmup if s < a.warmup else 1.0)
    teacher.requires_grad_(False).eval(); ema.requires_grad_(False).eval()
    student, ema, teacher, opt, dl, sch = accel.prepare(student, ema, teacher, opt, dl, sch)
    sched = FlowMatchScheduler(shift=5.0).to(accel.device); sigmas, tsteps = cd_axis(sched, a.N, accel.device)
    idx_rng = torch.Generator(device="cpu").manual_seed(1234 + a.seed)                 # same consistency level on every rank
    step, t0, it = 0, time.time(), iter(dl); student.train()
    while step < a.max_steps:
        try: b = next(it)
        except StopIteration: it = iter(dl); b = next(it)
        idx = int(torch.randint(0, a.N - 1, (1,), generator=idx_rng).item())
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss, s0, s1 = cd_loss((student, ema, teacher), cores, b, neg, sigmas, tsteps, idx, a.cfg, accel.device)
        accel.backward(loss)
        gn = student.clip_grad_norm_(10.0) if hasattr(student, "clip_grad_norm_") else torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0)
        if not torch.isfinite(gn):
            print(f"NONFINITE_GRAD rank {accel.process_index} step {step}", flush=True); opt.zero_grad(); step += 1; sch.step(); continue
        opt.step(); opt.zero_grad(); sch.step()
        ema_update(ema, student, decay=(0.0 if step < a.ema_start else a.ema_decay)); step += 1
        if step % a.log_every == 0:
            l = accel.gather(loss.detach()).nanmean().item(); sps = a.log_every / (time.time() - t0); t0 = time.time()
            log(f"step {step:6d}/{a.max_steps} cd {l:.5f} sig {s0:.3f}->{s1:.3f} gnorm {float(gn):.2f} {1 / max(sps, 1e-9):.2f} s/step")
        if step % a.save_every == 0 or step == a.max_steps:
            accel.wait_for_everyone(); sd_e = accel.get_state_dict(ema)
            if main_:
                os.makedirs(os.path.join(a.out, "ema"), exist_ok=True)
                st.save_file({k: v.contiguous().to(torch.bfloat16) for k, v in sd_e.items()}, os.path.join(a.out, "ema", f"ema_step{step}.safetensors"))
                log(f"saved ema step {step}")
            accel.wait_for_everyone()
    log("CD_DONE")


if __name__ == "__main__":
    main()
