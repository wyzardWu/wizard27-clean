"""Stage 3 (used): consistency distillation with history self-augmentation.

As `wizard.distill.consistency`, but with probability SA_P the history read by the student and the EMA is replaced by the teacher's
one-step x0 estimate of a noised copy, while the teacher keeps reading the clean ground-truth history: the student must match the
teacher's trajectory from a drifting history, using memory. Env SA_P, SA_MAX, SA_GROUPS as wizard.train.causal_selfaug."""
import os, random, torch, torch.nn.functional as F
import wizard.distill.consistency as CD
import wizard.train.causal as A
from wizard.models.flow_match import FlowMatchScheduler
from wizard.models.cond import cat_cond
from wizard.distill.common import noise_clip, _per_frame_t, _keep_frame0
SA_P = float(os.environ.get("SA_P", "0.7")); SA_MAX = float(os.environ.get("SA_MAX", "0.5")); CALL = [0]; STAT = [0, 0]; SCH = [None]
SA_GROUPS = set(os.environ.get("SA_GROUPS", "lookback,lookback0,sfly,sfly_replay").split(","))
_get = A.ClipDataset.__getitem__
def _getitem(self, i):
    d = _get(self, i); d["sa_ok"] = torch.tensor(float(self.rows[i]["group"] in SA_GROUPS)); return d
A.ClipDataset.__getitem__ = _getitem


def cd_loss(models, cores, b, neg, sigmas, tsteps, idx, cfg, dev):
    student, ema, teacher = models
    lat, cond, mem, gb, ctx, ref = CD.prep(b, dev)
    for c in cores: c._gb_in = gb; c._mem_warp = None; c._mem_wrope = None
    B, C, Fl = lat.shape[:3]
    xc = cat_cond(torch.cat([mem, lat], 2), cond).to(torch.bfloat16); xs = xc                   # xc: clean GT history (teacher); xs: the student's / EMA's
    run = lambda m, x, ts, cx_, hist_: m(cat_cond(torch.cat([mem, x], 2), cond).to(torch.bfloat16), ts, cx_, clean_x=hist_, ref=ref).float()
    CALL[0] += 1
    if random.Random(1000003 * CALL[0] + 17).random() < SA_P:                             # same decision on every rank (same number of forwards)
        if SCH[0] is None: SCH[0] = FlowMatchScheduler(shift=5.0).to(dev)                # the training grid (cd_axis re-set the CD scheduler to N levels)
        sch = SCH[0]
        with torch.no_grad():
            sig = torch.rand(B, Fl, device=dev) * SA_MAX; ia = sch.sigma_to_idx(sig.reshape(-1)).reshape(B, Fl)
            xa = sch.add_noise_pf(lat, torch.randn_like(lat), ia); tsa = sch.timesteps.to(dev, torch.float32)[ia].clone()
            xa[:, :, 0] = lat[:, :, 0]; tsa[:, 0] = 0.0
            va = run(teacher, xa, tsa, ctx, xc)                                           # the teacher's estimate, clean GT history as its context
            s_ = sch.sigmas.to(dev, torch.float32)[ia][:, None, :, None, None]
            hist = xa - s_ * va; hist[:, :, 0] = lat[:, :, 0]                              # one-step x0 estimate; the sink stays clean
        if float(b["sa_ok"][0]) > 0.5: xs = cat_cond(torch.cat([mem, hist], 2), cond).to(torch.bfloat16); STAT[0] += 1
    STAT[1] += 1
    s_t, s_tn = sigmas[idx].view(1, 1, 1, 1, 1), sigmas[idx + 1].view(1, 1, 1, 1, 1)
    ts_t = _per_frame_t(float(tsteps[idx]), B, Fl, dev); ts_tn = _per_frame_t(float(tsteps[idx + 1]), B, Fl, dev)
    latent_t = noise_clip(lat, s_t, torch.randn_like(lat))
    with torch.no_grad():
        v = run(teacher, latent_t, ts_t, ctx, xc)                                         # teacher: clean history
        if abs(cfg - 1.0) > 1e-6: v_u = run(teacher, latent_t, ts_t, neg, xc); v = v_u + cfg * (v - v_u)
        latent_tn = _keep_frame0(latent_t + (s_tn - s_t) * v, lat)
    cm_t = latent_t - s_t * run(student, latent_t, ts_t, ctx, xs)                           # student / EMA: the corrupted history
    with torch.no_grad(): cm_tn = latent_tn - s_tn * run(ema, latent_tn, ts_tn, ctx, xs)
    for c in cores: c._gb_in = None
    if STAT[1] % int(os.environ.get("SA_LOG_EVERY", "500")) == 0: print(f"SELF_AUG {STAT[0]}/{STAT[1]} samples (p {SA_P} on {sorted(SA_GROUPS)}, sigma <= {SA_MAX})", flush=True)
    return F.mse_loss(cm_t[:, :, 1:], cm_tn[:, :, 1:].detach()), float(sigmas[idx]), float(sigmas[idx + 1])


CD.cd_loss = cd_loss
if __name__ == "__main__": CD.main()
