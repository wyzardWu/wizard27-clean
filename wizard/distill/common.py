"""Small helpers shared by the distillation stages."""
import torch


def cd_axis(sched, n, dev):
    """Discrete consistency axis: N sigmas/timesteps (descending, shift-warped). Returns
    (sigmas[N], timesteps[N]) on dev. sigmas[0]~1 (high noise) -> sigmas[N-1]~0.1 (low)."""
    sched.set_timesteps(num_inference_steps=n, denoising_strength=1.0, device=dev)
    return sched.infer_sigmas.to(dev, torch.float32), sched.infer_timesteps.to(dev, torch.float32)


def _per_frame_t(t_val, B, Fl, dev):
    """[B,Fl] per-frame timestep, frame0 clean (t=0), frames 1..F-1 at t_val."""
    ts = torch.full((B, Fl), float(t_val), device=dev, dtype=torch.float32)
    ts[:, 0] = 0.0
    return ts


def noise_clip(clip, sigma, noise):
    """latent_t = (1-sigma)*clip + sigma*noise, frame0 forced clean. sigma: scalar tensor."""
    latent = (1 - sigma) * clip + sigma * noise
    f0 = (torch.arange(clip.shape[2], device=clip.device) == 0).view(1, 1, -1, 1, 1)
    return torch.where(f0, clip, latent)


def _keep_frame0(x, clip):
    f0 = (torch.arange(clip.shape[2], device=clip.device) == 0).view(1, 1, -1, 1, 1)
    return torch.where(f0, clip, x)


def ema_update(ema, student, decay):
    """Per-shard EMA. With fsdp_use_orig_params=true both models expose the SAME original
    Parameters in identical order, sharded identically -> zip is exact. decay<=0 => hard copy."""
    for pe, p in zip(ema.parameters(), student.parameters()):
        if decay <= 0.0:
            pe.data.copy_(p.data)
        else:
            pe.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
