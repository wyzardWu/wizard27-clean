"""the maintained world as a triangle MESH (not point squares): each source frame's MoGe point grid -> quads between neighbours on the same
surface (depth ratio < ratio), z-buffered rasterization (moderngl, EGL) into every target camera -> planar depth [T,H,W] (0 = nothing)."""
import os, numpy as np, moderngl
_CTX = _PROG = _FBO = None
VS = """#version 330
uniform vec3 pos; uniform mat3 Rm; uniform float f, cx, cy, W, H
in vec3 P; out float zc
void main(){ vec3 r = P - pos; float x = dot(r, Rm[0]), y = dot(r, Rm[1]), z = dot(r, Rm[2]); zc = z
  float n = 0.05, fa = 5000.0
  gl_Position = vec4((2.0*cx/W - 1.0)*z + 2.0*f/W*x, (1.0 - 2.0*cy/H)*z + 2.0*f/H*y, z*(fa+n)/(fa-n) - 2.0*fa*n/(fa-n), z); }"""
FS = """#version 330
in float zc; out vec4 o; void main(){ o = vec4(zc, 0, 0, 1); }"""

def grid_mesh(P, ok, z, ratio=1.12, step=2, loose=None, loose_ratio=6.0):
    """P [h,w,3] world points, ok [h,w], z [h,w] depth -> vertices [n,3], triangles [m,3]"""
    Q = P[::step, ::step]; m = ok[::step, ::step]; d = z[::step, ::step]; h, w = m.shape; idx = np.arange(h * w).reshape(h, w)
    m = m & np.isfinite(Q).all(-1) & np.isfinite(d) & (d > 0); Q = np.where(np.isfinite(Q), Q, 0.0); d = np.where(np.isfinite(d), d, 0.0)   # no NaN / inf vertex ever reaches the rasterizer
    a, b, c, e = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    D = np.stack([d[:-1, :-1], d[:-1, 1:], d[1:, :-1], d[1:, 1:]]); ok4 = m[:-1, :-1] & m[:-1, 1:] & m[1:, :-1] & m[1:, 1:]
    g = ok4 & (D.max(0) < ratio * D.min(0))
    if loose is not None:                                                               # ground plane / backdrop: may stretch (rows at the horizon), never across to other surfaces
        L = loose[::step, ::step]; L4 = L[:-1, :-1] & L[:-1, 1:] & L[1:, :-1] & L[1:, 1:]; g = g | (ok4 & L4 & (D.max(0) < loose_ratio * D.min(0)))
    tri = np.concatenate([np.stack([a[g], b[g], c[g]], 1), np.stack([b[g], e[g], c[g]], 1)])
    return Q.reshape(-1, 3).astype("f4"), tri.astype("i4")

_WK = {}
def _render_remote(meshes, pos, R, f, cx, cy, H, W):
    """the same render in the worker process (wizard.world.workers.mesh_render); restarted and retried when it dies"""
    import subprocess, tempfile, sys as _s
    V = np.concatenate([m[0] for m in meshes]).astype("f4"); off = np.cumsum([0] + [len(m[0]) for m in meshes[:-1]]); Fc = np.concatenate([m[1] + o for m, o in zip(meshes, off)]).astype("i4")
    d = tempfile.mkdtemp(dir="/dev/shm"); p = f"{d}/req.npz"; np.savez(p, V=V, F=Fc, pos=np.asarray(pos, np.float32), R=np.asarray(R, np.float32), cam=np.array([f, cx, cy, H, W], np.float64))
    try:
        for attempt in range(4):
            w = _WK.get("p")
            if w is None or w.poll() is not None:
                if w is not None: print(f"RENDER_WORKER died (rc {w.returncode}), restart #{attempt}", flush=True)
                _WK["p"] = w = subprocess.Popen([_s.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "workers", "mesh_render.py")], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
            try:
                w.stdin.write(p + "\n"); w.stdin.flush(); ans = w.stdout.readline().strip()
            except (BrokenPipeError, OSError): ans = ""
            if ans == "OK": return np.load(p + ".out.npy")
            if ans.startswith("ERR"): raise RuntimeError("render worker: " + ans)
        raise RuntimeError("render worker died 4 times")
    finally:
        import shutil; shutil.rmtree(d, ignore_errors=True)

def render(meshes, pos, R, f, cx, cy, H, W):
    """meshes: list of (V, F); pos [T,3], R [T,3,3] rows right/up/forward -> depth [T,H,W]"""
    global _CTX, _PROG, _FBO
    if os.environ.get("RENDER_PROC") == "1": return _render_remote(meshes, pos, R, f, cx, cy, H, W)
    if _CTX is None:
        dev = int(os.environ.get("EGL_DEVICE", (os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")[0]))   # the job's own card (EGL on another, busy card aborts)
        _CTX = moderngl.create_standalone_context(backend="egl", device_index=dev); _PROG = _CTX.program(vertex_shader=VS, fragment_shader=FS); _FBO = {}
    ctx = _CTX; prog = _PROG
    V = np.concatenate([m[0] for m in meshes]); off = np.cumsum([0] + [len(m[0]) for m in meshes[:-1]]); Fc = np.concatenate([m[1] + o for m, o in zip(meshes, off)])
    vbo = ctx.buffer(V.tobytes()); ibo = ctx.buffer(Fc.astype("i4").tobytes()); vao = ctx.vertex_array(prog, [(vbo, "3f", "P")], ibo)
    if (W, H) not in _FBO: _FBO[(W, H)] = ctx.framebuffer(color_attachments=[ctx.texture((W, H), 1, dtype="f4")], depth_attachment=ctx.depth_renderbuffer((W, H)))
    fbo = _FBO[(W, H)]
    for k, v in dict(f=f, cx=cx, cy=cy, W=float(W), H=float(H)).items(): prog[k].value = float(v)
    out = np.zeros((len(pos), H, W), np.float32); fbo.use(); ctx.enable(moderngl.DEPTH_TEST)
    for t in range(len(pos)):
        prog["pos"].value = tuple(float(x) for x in pos[t]); prog["Rm"].value = tuple(float(x) for x in np.asarray(R[t], float).reshape(-1))   # mat3 column-major: columns = rows of R
        fbo.clear(0.0, 0.0, 0.0, 1.0, depth=1.0); vao.render(moderngl.TRIANGLES)
        out[t] = np.frombuffer(fbo.color_attachments[0].read(), np.float32).reshape(H, W)[::-1]
    for o in (vao, vbo, ibo): o.release()
    return out
