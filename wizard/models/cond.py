"""Visual-proxy injection by channel concatenation.

The proxy video is encoded with the same Wan2.2 VAE, so its latent frames align 1:1 with the video's. Callers pass
``torch.cat([latent, proxy_latent], dim=1)`` (96 channels); ``attach_cond`` adds a zero-initialised ``cond_embedding``
(the same Conv3d patch as ``patch_embedding``) and makes ``patch_embedding`` compute
    patch_embedding(x[:, :48]) + cond_embedding(x[:, 48:]).
The official checkpoint keys stay untouched, and a 48-channel input takes the original path bit for bit."""
import torch
import torch.nn as nn
import safetensors.torch as st
import json, os

LAT_C = 48


def attach_cond(model):
    pe = model.patch_embedding
    if not hasattr(model, "cond_embedding"):
        model.cond_embedding = nn.Conv3d(LAT_C, pe.out_channels, kernel_size=pe.kernel_size, stride=pe.stride).to(pe.weight.device, pe.weight.dtype)
        nn.init.zeros_(model.cond_embedding.weight); nn.init.zeros_(model.cond_embedding.bias)
    orig = nn.Conv3d.forward
    ce = model.cond_embedding

    def fwd(x):
        if x.shape[1] == LAT_C:
            return orig(pe, x)
        assert x.shape[1] == 2 * LAT_C, f"expected 48 or 96 channels, got {x.shape[1]}"
        return orig(pe, x[:, :LAT_C]) + ce(x[:, LAT_C:])
    pe.forward = fwd
    return model


def load_official(model, ckpt_dir):
    """Official Wan2.2-TI2V-5B sharded safetensors -> WizardDiT (key names match DiffSynth / official WanModel)."""
    idx = json.load(open(os.path.join(ckpt_dir, "diffusion_pytorch_model.safetensors.index.json")))["weight_map"]
    sd = {}
    for shard in sorted(set(idx.values())):
        sd.update(st.load_file(os.path.join(ckpt_dir, shard)))
    info = model.load_state_dict(sd, strict=False, assign=True)
    if any(b.is_meta for b in model.rope.buffers()) or any(getattr(model.rope, a).is_meta for a in dir(model.rope)
                                                            if isinstance(getattr(model.rope, a, None), torch.Tensor)):
        model.rope = type(model.rope)(model.dim // model.num_heads, max_pos=1024)    # built on meta: rebuild the tables
    return info


def cat_cond(x, cond):
    """x [B,48,F,H,W] (or one frame), cond aligned frames -> [B,96,...]."""
    return torch.cat([x, cond.to(x.dtype)], dim=1)
