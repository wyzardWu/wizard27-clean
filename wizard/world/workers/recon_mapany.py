"""joint multi-view reconstruction of generated keyframes with MapAnything (run under venv_mapany):
   wizard.world.workers.recon_mapany in.npz out.npz      in: img [N,H,W,3] u8, K [3,3] (pixels of img), c2w [N,4,4] OpenCV cam2world (our world), depth0 [H,W] (view 0 metric depth, 0 = none)
   out: pts [N,H,W,3] world points, mask [N,H,W], conf [N,H,W], depth [N,H,W] -- at the input resolution (nearest upsampling of the model grid) wizard.world.workers.recon_mapany --serve: the model stays loaded; one "in.npz out.npz" per stdin line, MAPANY_DONE per request (no 20 s reload per maintenance)"""
import os, sys, time, numpy as np, torch, torch.nn.functional as F
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from mapanything.models import MapAnything
from mapanything.utils.image import preprocess_inputs
dev = "cuda"
model = MapAnything.from_pretrained(os.environ.get("MAPANY", "facebook/map-anything")).to(dev).eval()


def run(inp, outp):
    d = np.load(inp); img, K, c2w, d0 = d["img"], d["K"].astype(np.float32), d["c2w"].astype(np.float32), d["depth0"].astype(np.float32)
    N, H, W = img.shape[:3]
    views = []
    for i in range(N):
        v = dict(img=torch.from_numpy(img[i]), intrinsics=torch.from_numpy(K), camera_poses=torch.from_numpy(c2w[i]), is_metric_scale=torch.tensor([True]))
        di = d["depthN"][i].astype(np.float32) if "depthN" in d.files else (d0 if i == 0 else np.zeros_like(d0))     # known world depth of every view anchors the reconstruction
        if (di > 0).any(): v["depth_z"] = torch.from_numpy(di)
        views.append(v)
    pv = preprocess_inputs(views)
    for _ in range(2 if os.environ.get("WARM") else 1):
        torch.cuda.synchronize(); t0 = time.time()
        with torch.no_grad(): pr = model.infer(pv, memory_efficient_inference=False, use_amp=True, amp_dtype="bf16", apply_mask=True, mask_edges=bool(os.environ.get("MAPANY_EDGES")))
        torch.cuda.synchronize(); dt = time.time() - t0
    print(f"MAPANY_TIME {dt:.3f}s for {N} views, model grid {tuple(pr[0]['pts3d'].shape[1:3])}", flush=True)
    g = lambda k: np.stack([p[k][0].float().cpu().numpy() for p in pr])
    C0 = c2w[0].astype(np.float64)                                                           # the outputs are in view 0's frame: back to the input world
    def to_world(X): return (X.astype(np.float64) @ C0[:3, :3].T + C0[:3, 3]).astype(np.float32)
    pts, msk, conf, dep = to_world(g("pts3d")), g("mask")[..., 0] > 0.5, g("conf"), g("depth_z")[..., 0]          # the model grid (the preprocessing may crop); world points need no pixel mapping
    np.savez(outp, pts=pts.astype(np.float32), mask=msk, conf=conf.astype(np.float32), depth=dep.astype(np.float32), Km=g("intrinsics"), poses=np.einsum("ij,njk->nik", C0, g("camera_poses").astype(np.float64)), ptscam=g("pts3d_cam")); print("MAPANY_DONE", pts.shape, flush=True)


if sys.argv[1] == "--serve":
    print("MAPANY_READY", flush=True)
    for line in sys.stdin:
        if line.strip(): run(*line.split())
else:
    run(sys.argv[1], sys.argv[2])
