"""Stage 4: distribution matching distillation (DMD) of the 2-step causal student.

The student is rolled out autoregressively exactly as `wizard.inference.causal` deploys it (KV-cached stream, memory slots, world
G-buffer, gated references); the real score and the fake score (critic) are the bidirectional teacher on [memory | clip] with the same
memory, real score with CFG. Critic : student updates = 5 : 1, gradient accumulation, gradient only on a random window of latent frames.

    accelerate launch --config_file configs/accelerate/fsdp_dmd.yaml --num_processes 8 -m wizard.distill.dmd \
        --data $WIZARD_DATA/wan_v7 --student <cd ema> --real <bidirectional dit> --out runs/dmd --accum 4 --cfg 3
    resume: --student <dmd ema> --fake <runs/dmd/fake_last.safetensors> --step0 <step>"""
import os, sys, glob, json, time, argparse, random, collections
import torch
import torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__)); 
from wizard.models.dit import WizardDiT, ti2v_5b_config
from wizard.models.flow_match import FlowMatchScheduler
from wizard.models.cond import attach_cond, cat_cond
import wizard.train.causal as A
import wizard.inference.stream as AS
from wizard.train.bidirectional import collate, lt, DS
from wizard.models.world_cond import attach_world, M_MAX
from wizard.distill.consistency import prep
from wizard.distill.common import _per_frame_t, _keep_frame0, ema_update
import safetensors.torch as st


def load(model, path):
    info = model.load_state_dict(st.load_file(path), strict=False)
    bad = [k for k in info.missing_keys if "rope" not in k]
    assert not bad and not info.unexpected_keys, (path, bad[:5], info.unexpected_keys[:5])


_fwd0 = WizardDiT.forward
def _fwd(self, x=None, timestep=None, context=None, *a_, rollout=None, **kw):
    return rollout(self) if rollout is not None else _fwd0(self, x, timestep, context, *a_, **kw)
WizardDiT.forward = _fwd


class GradStream(AS.Stream):
    """Stream whose frame can carry the gradient: memory / reference K/V captured without it (as the old ref_cache), committed history
    detached; the text and video coordinate embeddings recomputed with the gradient so they train as before."""
    def __init__(self, model, ctx, gc, gm, mem_lat, ref, W=20, sink=1):
        super().__init__(model, ctx, gc, gm, mem_lat, ref, W, sink)
        dt = self.ev.dtype; self.ctx = model.text_embedding(ctx.to(dt)); self.ev = model.gbuf_embedding(gc.to(mem_lat.device)).to(dt)

    def _run(self, h, ts_tok, mode, Fl=1, Hl=15, Wl=26, frame=None, commit=False):
        """Stream._run with per-block gradient checkpointing when the frame carries the gradient; the recompute reads the same cache entries (frames < f)."""
        if not torch.is_grad_enabled(): return super()._run(h, ts_tok, mode, Fl, Hl, Wl, frame, commit)
        from torch.utils.checkpoint import checkpoint
        m = self.m; dt = h.dtype; B, S = h.shape[:2]
        t = m.time_embedding(AS.sinusoidal_embedding_1d(m.freq_dim, ts_tok.reshape(-1)).to(dt)).view(B, S, m.dim)
        t_mod = m.time_projection(t).unflatten(2, (6, m.dim)); self.kv["mode"] = mode
        info = (0, 0, 0, 0, 0)
        for li, blk in enumerate(m.blocks):
            h = checkpoint(blk, h, t_mod, self.ctx, m.rope, Fl, Hl, Wl, Hl * Wl, use_reentrant=False, attn_mask=None, kv_cache=self.kv, layer_idx=li, frame_index=frame, commit=commit, ref=info)
        return h, t

    def frame_g(self, x, cond, f, t, commit=False):
        dt = self.ev.dtype; h = self._pe(cat_cond(x.to(dt), cond.to(dt))) + self.ev[:, :, f:f + 1]
        h = h.flatten(2).transpose(1, 2)
        h, tt = self._run(h, torch.full((1, h.shape[1]), float(t), device=h.device), "frame", frame=f, commit=commit)
        return self.m._unpatchify(self.m.head(h, tt), 1, 15, 26)


def rollout_fn(clip, cond, context, ref, levels, exit_at, grad, kv_window, sink, gwin, mem, gb):
    """-> a function of the BARE student that rolls the clip out frame by frame on its own committed history and returns [B,C,F,H,W] (frame 0 = the
    given first frame). The last level (exit_at) of the frames gwin[0] <= k < gwin[1] carries the gradient when `grad`: a frame's graph holds its own
    concatenated history keys / values in all 30 blocks, so only a window does. cond = the video condition (31 frames), mem / gb = the memory latents and (video, memory) G-buffer; the stream = wizard.inference.causal's."""
    def run(m):
        B, C, Fl = clip.shape[:3]
        with torch.set_grad_enabled(grad):
            s = GradStream(m, context, gb[0], gb[1], mem, ref, W=kv_window, sink=sink)
        with torch.no_grad(): s.frame_g(clip[:, :, :1], cond[:, :, :1], 0, 0.0, commit=True)
        if os.environ.get("DMD_MEMDBG") and torch.distributed.get_rank() == 0: print(f"MEMDBG after stream init + frame 0: alloc {torch.cuda.memory_allocated() / 2 ** 30:.1f} GB", flush=True)
        out = [clip[:, :, :1]]
        for k in range(1, Fl):
            ck = cond[:, :, k:k + 1]; x = torch.randn_like(clip[:, :, :1]); x0 = None
            for i, t in enumerate(levels[:exit_at + 1]):
                last = i == exit_at
                with torch.set_grad_enabled(grad and last and gwin[0] <= k < gwin[1]):
                    x0 = x - (t / 1000.0) * s.frame_g(x, ck, k, t).float()
                if not last: ns = levels[i + 1] / 1000.0; x = (1 - ns) * x0.detach() + ns * torch.randn_like(x0)
            with torch.no_grad(): s.frame_g(x0.detach(), ck, k, 0.0, commit=True)        # its own frame, clean, becomes history
            if os.environ.get("DMD_MEMDBG") and torch.distributed.get_rank() == 0: print(f"MEMDBG frame {k} alloc {torch.cuda.memory_allocated() / 2 ** 30:.1f} GB max {torch.cuda.max_memory_allocated() / 2 ** 30:.1f}", flush=True)
            out.append(x0)
        return torch.cat(out, 2)
    return run


def bidir(m, core, x, mem, cond, gb, ctx, ref, ts):
    """the bidirectional model on [memory | clip] (wizard.train.bidirectional.loss_fn's call) -> velocity of the clip"""
    core._gb_in = gb; core._mem_warp = None; core._mem_wrope = None
    tsa = torch.cat([torch.zeros(ts.shape[0], M_MAX, device=ts.device), ts], 1)
    v = m(cat_cond(torch.cat([mem, x], 2), cond).to(torch.bfloat16), tsa, ctx, ref=ref).float()
    core._gb_in = None; return v


def ti2v_noise(clip, sched, idx):
    """the bidirectional model's own training noising: frame 0 clean (t = 0), frames 1.. at one shared level"""
    B, C, Fl = clip.shape[:3]; dev = clip.device; noise = torch.randn_like(clip); noisy = sched.add_noise(clip, noise, idx); noisy[:, :, 0] = clip[:, :, 0]
    ts = sched.timesteps.to(dev, torch.float32)[idx][:, None].expand(B, Fl).clone(); ts[:, 0] = 0.0
    return noisy, ts, sched.sigmas.to(dev, torch.float32)[idx].view(B, 1, 1, 1, 1), noise


def dmd_loss(real, fake, gen, cond, context, uncond, ref, sched, idx, cfg, gwin, cores, mem, gb):
    with torch.no_grad():
        noisy, ts, sig, _ = ti2v_noise(gen.detach(), sched, idx)
        sc = lambda m, c_: bidir(m, cores[id(m)], noisy, mem, cond, gb, c_, ref, ts)      # [memory | clip], aligned inputs
        x_c = noisy - sig * sc(real, context)
        if abs(cfg - 1.0) > 1e-6: x_u = noisy - sig * sc(real, uncond); pred_real = x_u + cfg * (x_c - x_u)   # as the bidirectional model samples (condition + memory + references in both)
        else: pred_real = x_c
        pred_fake = noisy - sig * sc(fake, context)
        g0, g1 = gwin; norm = (gen.detach() - pred_real)[:, :, g0:g1].abs().mean(dim=[1, 2, 3, 4], keepdim=True).clamp(min=1e-2)
        grad = torch.nan_to_num((pred_fake - pred_real) / norm)
    loss = 0.5 * F.mse_loss(gen[:, :, g0:g1].float(), (gen - grad).detach()[:, :, g0:g1].float())     # the frames that carry the student's gradient
    return loss, float(grad[:, :, 1:].float().pow(2).mean().sqrt()), float((pred_real[:, :, 1:].float().std())), float(pred_fake[:, :, 1:].float().std())


def critic_loss(fake, gen, cond, context, ref, sched, idx, cores, mem, gb):
    clip = gen.detach(); noisy, ts, sig, noise = ti2v_noise(clip, sched, idx)
    v = bidir(fake, cores[id(fake)], noisy, mem, cond, gb, context, ref, ts)
    return F.mse_loss(v[:, :, 1:].float(), (noise - clip)[:, :, 1:].float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--group_rep", default="e1o:2,e2o:2,e1n:2,e2n:2,e1x:2,turn:2,roam:2")
    ap.add_argument("--rep_mem", type=int, default=0); ap.add_argument("--rep_keys", default=""); ap.add_argument("--extra_rows", default=""); ap.add_argument("--mem_index", default="mem_index.pt")
    ap.add_argument("--p_irrel", type=float, default=0.2); ap.add_argument("--p_nomem", type=float, default=0.1); ap.add_argument("--p_trueunk", type=float, default=0.0); ap.add_argument("--p_unk2", type=float, default=0.0); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--accum", type=int, default=8, help="micro-batches per GPU per update, critic and student (8 GPUs x 8 = 64)")
    ap.add_argument("--student", required=True, help="the 2-step causal student to start from (wizard.distill.consistency ema)"); ap.add_argument("--real", required=True, help="the bidirectional model: real score, and the fake score's start")
    ap.add_argument("--out", required=True); ap.add_argument("--levels", default="1000,500", help="the student's sampling levels, e.g. 1000,500")
    ap.add_argument("--cfg", type=float, default=3.0, help="guidance of the real score (the bidirectional model is sampled at 3)")
    ap.add_argument("--lr", type=float, default=2e-6); ap.add_argument("--lr_critic", type=float, default=4e-7); ap.add_argument("--critic_steps", type=int, default=5, help="critic updates per student update")
    ap.add_argument("--max_steps", type=int, default=400, help="STUDENT updates"); ap.add_argument("--step0", type=int, default=0, help="step count to continue from"); ap.add_argument("--fake", default="", help="fake score init, default = --real"); ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--ema_decay", type=float, default=0.99); ap.add_argument("--ema_start", type=int, default=20)
    ap.add_argument("--windowed_train", type=int, default=20); ap.add_argument("--windowed_sink", type=int, default=1)
    ap.add_argument("--save_every", type=int, default=25); ap.add_argument("--log_every", type=int, default=1); ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--opt8bit", type=int, default=1); ap.add_argument("--rope_cap", type=int, default=21); ap.add_argument("--grad_frames", type=int, default=10, help="latent frames of the rollout that keep the student's gradient (a random window per clip)")
    ap.add_argument("--check", type=int, default=0, help="N > 0: run N student updates' worth of steps, print the numbers, save nothing")
    a = ap.parse_args()
    a.mem_warp = a.mem_wrope = a.no_coords = a.empty_actor = a.skeleton_depth_cond = a.coords_only = False                # ARData's switches, as wizard.distill.consistency
    a.p_memunk = 0.0; a.mem_loss_w = 1.0; a.p_text_drop = 0.0; a.p_memnoise = 0.0; a.memnoise = 0.0; a.p_ident = 0.0; a.window = a.windowed_train; a.sink = a.windowed_sink
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    accel = Accelerator(step_scheduler_with_optimizer=False); set_seed(a.seed, device_specific=True); random.seed(a.seed + accel.process_index); main_ = accel.is_main_process; os.makedirs(a.out, exist_ok=True)
    log = (lambda *x: (print(*x, flush=True), open(os.path.join(a.out, "train.log"), "a").write(" ".join(map(str, x)) + "\n"))) if main_ else (lambda *x: None)
    levels = [float(v) for v in a.levels.split(",")]
    student = A.build_model(a); load(student, a.student); ema = A.build_model(a); load(ema, a.student)           # causal + proxy + world tokens, window / sink
    real = WizardDiT(**ti2v_5b_config(), causal=False, zero_init_head=False); attach_cond(real); attach_world(real); load(real, a.real)
    fake = WizardDiT(**ti2v_5b_config(), causal=False, zero_init_head=False); attach_cond(fake); attach_world(fake); load(fake, a.fake or a.real)
    real_core, fake_core = real, fake                                                    # _gb_in lives on the bare modules
    student.use_gradient_checkpointing = True; fake.use_gradient_checkpointing = True
    rows = [json.loads(l) for l in open(os.path.join(a.data, "v7_rows.jsonl"))]            # data = wizard.distill.consistency's
    for er in [x for x in a.extra_rows.split(",") if x]: rows += [json.loads(l) for l in open(er)]
    if a.limit: rows = rows[:a.limit] + [r for r in rows if r["group"] == "roam"][:a.limit]
    rep = {k: int(v) for k, v in (x.split(":") for x in a.group_rep.split(","))}
    rk = set(json.load(open(a.rep_keys))) if a.rep_keys else None
    nrep = lambda r: a.rep_mem if (a.rep_mem and r["group"] == "roam" and rk is not None and r["key"] in rk) else rep.get(r["group"], 1)
    files = [r for r in rows for _ in range(nrep(r))]
    c = collections.Counter(r["group"] for r in files); log("group_rep", rep, "->", {k: f"{v} ({100 * v / len(files):.1f}%)" for k, v in sorted(c.items())})
    log(f"DMD v7 student <- {a.student} | real / fake score <- {a.real} (bidirectional, cfg {a.cfg}) | levels {levels} | autoregressive rollout, window {a.windowed_train} + sink {a.windowed_sink} | world {accel.num_processes} x accum {a.accum} = batch {accel.num_processes * a.accum} | rows {len(files)} | lr {a.lr} / critic {a.lr_critic} x{a.critic_steps}")
    if main_: json.dump(vars(a), open(os.path.join(a.out, "args.json"), "w"), indent=1)
    ds = A.ARData(a.data, files, a)
    neg = lt(f"{DS}/wan_mix5/neg_emb.pt")["neg_emb"]; neg = torch.cat([neg, neg.new_zeros(512 - neg.shape[0], neg.shape[1])])     # empty prompt zero-padded to 512 tokens
    assert neg.shape[0] == 512, neg.shape
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True, num_workers=a.num_workers, collate_fn=collate, drop_last=True, pin_memory=True, persistent_workers=True)
    def adam(p, lr):
        if a.opt8bit:
            import bitsandbytes as bnb
            return bnb.optim.AdamW8bit(p, lr=lr, weight_decay=0.0, betas=(0.0, 0.999))
        return torch.optim.AdamW(p, lr=lr, weight_decay=0.0, betas=(0.0, 0.999))
    opt_g, opt_c = adam(student.parameters(), a.lr), adam(fake.parameters(), a.lr_critic)
    sch = torch.optim.lr_scheduler.LambdaLR(opt_g, lambda s: (s + 1) / a.warmup if s < a.warmup else 1.0)
    real.requires_grad_(False).eval(); ema.requires_grad_(False).eval()
    student, ema, real, fake, opt_g, opt_c, dl, sch = accel.prepare(student, ema, real, fake, opt_g, opt_c, dl, sch)
    cores = {id(real): real_core, id(fake): fake_core}                                   # wrapped -> bare module (for _gb_in)
    sched = FlowMatchScheduler(shift=5.0); n_t = sched.timesteps.shape[0]; lo, hi = int(0.02 * n_t), int(0.98 * n_t)
    rng = torch.Generator(device="cpu").manual_seed(4321 + a.seed)                  # the same exit step / score level on every rank
    it = iter(dl); g_step = a.step0; k = 0; t0 = time.time(); student.train(); fake.train(); stats = {}
    clip_ = lambda m: m.clip_grad_norm_(10.0) if hasattr(m, "clip_grad_norm_") else torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
    total = a.check or a.max_steps
    while g_step < total:
        try: b = next(it)
        except StopIteration: it = iter(dl); b = next(it)
        dev = accel.device; clip, cond, mem, gb, ctx, ref = prep(b, dev); act = cond[:, :, M_MAX:]   # cond = [memory zeros | video condition]
        unc = neg[None].to(dev, ctx.dtype)
        exit_at = int(torch.randint(0, len(levels), (1,), generator=rng).item()); idx = torch.randint(lo, hi, (1,), generator=rng).to(dev).expand(clip.shape[0])
        gen_turn = ((k // a.accum) % (a.critic_steps + 1)) == a.critic_steps; last_micro = (k + 1) % a.accum == 0; k += 1   # accum micro-batches per update
        Fl_ = clip.shape[2]; g0 = int(torch.randint(1, max(2, Fl_ - a.grad_frames + 1), (1,), generator=rng).item()); gwin = (g0, min(Fl_, g0 + a.grad_frames))
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
            gen = student(rollout=rollout_fn(clip, act, ctx, ref, levels, exit_at, gen_turn, a.windowed_train, a.windowed_sink, gwin, mem, gb)).to(clip.dtype)
            if gen_turn:
                loss, gnm, sd_r, sd_f = dmd_loss(real, fake, gen, cond, ctx, unc, ref, sched, idx, a.cfg, gwin, cores, mem, gb)
                for k_, v_ in dict(dmd=loss.item(), grad=gnm, sd_real=sd_r, sd_fake=sd_f, sd_gen=float(gen[:, :, 1:].float().std()), sd_gt=float(clip[:, :, 1:].float().std())).items(): stats[k_] = stats.get(k_, 0.0) + v_ / a.accum
            else:
                loss = critic_loss(fake, gen, cond, ctx, ref, sched, idx, cores, mem, gb); stats["critic"] = stats.get("critic", 0.0) + loss.item() / (a.accum * a.critic_steps)
        accel.backward(loss / a.accum)
        if not last_micro: continue                                                      # accumulate
        if gen_turn:
            gn = clip_(student); opt_g.step(); opt_g.zero_grad(); opt_c.zero_grad(); sch.step(); g_step += 1
            ema_update(ema, student, decay=(0.0 if g_step < a.ema_start else a.ema_decay)); stats["gnorm"] = float(gn)
            if g_step % a.log_every == 0 or a.check:
                log(f"step {g_step:5d}/{total} dmd {stats.get('dmd', 0):.5f} grad_rms {stats.get('grad', 0):.3f} critic {stats.get('critic', 0):.4f} gnorm {stats['gnorm']:.2f} | std gen {stats.get('sd_gen', 0):.3f} gt {stats.get('sd_gt', 0):.3f} real {stats.get('sd_real', 0):.3f} fake {stats.get('sd_fake', 0):.3f} | exit {exit_at} grad frames {gwin[0]}-{gwin[1] - 1} | mem {torch.cuda.max_memory_allocated() / 2 ** 30:.0f} GB | {(time.time() - t0) / max(g_step - a.step0, 1):.1f} s/step")
            stats = {}                                                                   # per-update means over the accumulated micro-batches
            if not a.check and (g_step % a.save_every == 0 or g_step == a.max_steps):
                accel.wait_for_everyone(); sd_e = accel.get_state_dict(ema); sd_s = accel.get_state_dict(student); sd_f = accel.get_state_dict(fake)
                if main_:
                    os.makedirs(os.path.join(a.out, "ema"), exist_ok=True)
                    st.save_file({k_: v.contiguous().to(torch.bfloat16) for k_, v in sd_e.items()}, os.path.join(a.out, "ema", f"ema_step{g_step}.safetensors")); log(f"saved ema step {g_step}")
                    for nm_, sd_ in (("student", sd_s), ("fake", sd_f)): st.save_file({k_: v.contiguous().to(torch.bfloat16) for k_, v in sd_.items()}, os.path.join(a.out, f"{nm_}_last.safetensors.tmp")); os.replace(os.path.join(a.out, f"{nm_}_last.safetensors.tmp"), os.path.join(a.out, f"{nm_}_last.safetensors"))
                    log(f"saved student / fake state at step {g_step}")
                accel.wait_for_everyone()
        else:
            clip_(fake); opt_c.step(); opt_c.zero_grad(); opt_g.zero_grad()
    log("DMD_CHECK_DONE" if a.check else "DMD_DONE")


if __name__ == "__main__":
    main()
