"""EGL mesh rendering in its own process (wizard.world.mesh_depth.render with RENDER_PROC=1 talks to it): on a shared server the EGL rasterizer aborts the whole
process now and then when other jobs start on the machine; isolated here, a crash costs one retry instead of the generation run.
Protocol: one request path per stdin line (npz: V, F, pos, R, cam=[f,cx,cy,H,W]); writes <path>.out.npy, answers 'OK' (or 'ERR msg')."""
import os, sys, numpy as np
os.environ.pop("RENDER_PROC", None)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))); import wizard.world.mesh_depth as MD
for line in sys.stdin:
    p = line.strip()
    if not p: continue
    try:
        d = np.load(p); f, cx, cy, H, W = d["cam"].tolist()
        out = MD.render([(d["V"], d["F"])], d["pos"], d["R"], f, cx, cy, int(H), int(W)); np.save(p + ".out.npy", out.astype(np.float32)); print("OK", flush=True)
    except Exception as e: print("ERR", repr(e)[:300], flush=True)
