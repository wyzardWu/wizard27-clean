"""MoGe-2 depth of generated frames (run under venv_triposg, which has moge): wizard.world.workers.moge_frames <in.npy [n,H,W,3] uint8> <out.npz>
writes z [n,H,W] float32 (MoGe planar depth, its own scale) and ok [n,H,W] bool (MoGe's valid mask; invalid = sky / undefined)."""
import sys, numpy as np, torch
from moge.model.v2 import MoGeModel
fr = np.load(sys.argv[1]); m = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to("cuda").eval(); Z, OK = [], []
for im in fr:
    with torch.no_grad(): o = m.infer(torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float().div(255).cuda())
    Z.append(o["points"][..., 2].cpu().numpy().astype(np.float32)); OK.append(o["mask"].cpu().numpy().astype(bool))
np.savez(sys.argv[2], z=np.stack(Z), ok=np.stack(OK)); print("MOGE_DONE", len(Z), flush=True)
