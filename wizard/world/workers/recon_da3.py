"""the same job as wizard.world.workers.recon_mapany with Depth Anything 3 (DA3NESTED-GIANT-LARGE-1.1, pose-conditioned; run under venv_da3):
   wizard.world.workers.recon_da3 in.npz out.npz     in: img [N,H,W,3] u8, K [3,3], c2w [N,4,4] OpenCV cam2world (proper rotation)
   out: ptscam [N,h,w,3] camera-frame points on DA3's grid, mask [N,h,w] (finite, not sky), conf, Km [N,3,3] (the grid's intrinsics)"""
import os, sys, time, numpy as np, torch
from depth_anything_3.api import DepthAnything3
d = np.load(sys.argv[1]); img, K, c2w = d["img"], d["K"].astype(np.float32), d["c2w"].astype(np.float64)
N = len(img); model = DepthAnything3.from_pretrained(os.environ.get("DA3", "depth-anything/DA3NESTED-GIANT-LARGE-1.1")).to("cuda").eval()
w2c = np.linalg.inv(c2w).astype(np.float32); Ks = np.repeat(K[None], N, 0)
for _ in range(2 if os.environ.get("WARM") else 1):
    torch.cuda.synchronize(); t0 = time.time()
    run = lambda al, ext=True: model.inference([im for im in img], extrinsics=w2c if ext else None, intrinsics=Ks if ext else None, align_to_input_ext_scale=al, process_res=int(os.environ.get("DA3_RES", "504")))
    with torch.no_grad():
        try: pr = run(N >= 3)
        except Exception as e:                       # near-collinear cameras: Umeyama degenerate, scale is fixed later by the world guide anyway
            print("da3 align fallback:", type(e).__name__, flush=True); pr = run(False, False)
    torch.cuda.synchronize(); dt = time.time() - t0
dep = pr.depth.astype(np.float32); Km = pr.intrinsics.astype(np.float32); h, w = dep.shape[1:]
gv, gu = np.mgrid[0:h, 0:w] + 0.5
pts = np.stack([np.stack([(gu - k[0, 2]) / k[0, 0] * z, (gv - k[1, 2]) / k[1, 1] * z, z], -1) for z, k in zip(dep, Km)]).astype(np.float32)
sky = pr.sky.astype(bool) if getattr(pr, "sky", None) is not None else np.zeros_like(dep, bool)
mask = np.isfinite(dep) & (dep > 0) & ~sky
print(f"MAPANY_TIME {dt:.3f}s for {N} views (DA3), grid {(h, w)}, sky {sky.mean():.3f}, depth p50/p95/p99 {np.percentile(dep[mask], [50, 95, 99]).round(1)}", flush=True)
np.savez_compressed(sys.argv[2], ptscam=pts, mask=mask, conf=pr.conf.astype(np.float32), Km=Km, sky=sky)
print("MAPANY_DONE", flush=True)
