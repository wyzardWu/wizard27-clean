"""inference world maintenance, ONE world:
  world = the case's known geometry (per pixel, <case>/world_px.npz from depth_edit_v7 DUMP_Z=1; the first frame's world as EXACT, V7_SEEN_EXACT=1)
          + meshes grown from the generated frames.
  After a window: its keyframes in order; each one's MoGe-2 depth is scale-aligned to the CURRENT world rendered at that camera, and only the part the
  world does not have yet is added (still-unknown pixels, not sky, not characters), stitched to the surfaces it touches (a ring with the world's own
  depth), small fragments dropped. Sky = MoGe invalid, or sky-coloured, above the horizon, far, and connected to the top of the picture.
  The later frames are then re-rendered from the whole world in ONE encoding: exact tier + normal shading (as the case's own exact geometry),
  R = height code fitted on the case's exact pixels, B = depth; never-seen stays unknown, sky directions black. Tokens follow (8x8 majority)."""
import os, numpy as np, torch, cv2
from wizard.paths import PY_MAPANY, PY_DA3
import wizard.world.maintainer as MV
import wizard.world.mesh_depth as MD
import sys; import wizard.data.cond_encode as E7


class _MapAnyServer:
    """MapAnything kept loaded in its own venv: requests over stdin, replies on stdout (MAPANY_DONE ends one)"""
    def __init__(self):
        import subprocess
        env = {**os.environ, **({"CUDA_VISIBLE_DEVICES": os.environ["MAINT_CVD"]} if os.environ.get("MAINT_CVD") else {})}
        self.p = subprocess.Popen([PY_MAPANY, "-u", os.path.dirname(os.path.abspath(__file__)) + "/workers/recon_mapany.py", "--serve"],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, text=True, env=env, bufsize=1)
        self._wait("MAPANY_READY")

    def _wait(self, tag):
        lines = []
        while True:
            l = self.p.stdout.readline()
            if not l: raise RuntimeError("MapAnything server died: " + "".join(lines[-20:]))
            lines.append(l.rstrip())
            if l.startswith(tag): return lines

    def run(self, inp, outp):
        self.p.stdin.write(f"{inp} {outp}\n"); self.p.stdin.flush(); return None, self._wait("MAPANY_DONE")


_MAS = []
def _mapany_server():
    if not _MAS: _MAS.append(_MapAnyServer())
    return _MAS[0]


def sky_colour(im):
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV); h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    return ((h >= 85) & (h <= 130) & (s >= 35) & (v >= 110)) | ((s <= 45) & (v >= 165))          # blue sky | white / pale clouds


def top_connected(m):
    n, lab = cv2.connectedComponents(m.astype(np.uint8)); top = np.unique(lab[0][m[0]]); return np.isin(lab, top[top > 0])


SEG_ON = os.environ.get("SKY_SEG", "1") == "1"
WATER_IDS = [21, 26, 60, 109, 128]
GROUND_IDS = [3, 6, 9, 11, 13, 28, 29, 46, 52, 54, 91, 94]                                     # ADE20K floor road grass sidewalk earth rug field sand path runway dirt-track land
VERT_IDS = [0, 1, 4, 16, 17, 25, 34, 42, 48, 68, 72, 84]                                       # wall building tree mountain plant house rock column skyscraper hill palm tower
GROUND_SCALE = os.environ.get("GROUND_SCALE", "1") == "1"
                                                             # ADE20K water, sea, river, pool, lake
_SEG = {}
def seg_labels(img):
    """SegFormer-b2 ADE20K labels at the picture size (~9 ms / frame on H200, fp16); cached per image"""
    import torch
    key = (img.ctypes.data, img.shape)
    if key in _SEG: return _SEG[key]
    if "m" not in _SEG:
        from transformers import SegformerForSemanticSegmentation
        _SEG["m"] = SegformerForSemanticSegmentation.from_pretrained(os.environ.get("SEG_MODEL", "nvidia/segformer-b2-finetuned-ade-512-512")).cuda().half().eval()
    H, W = img.shape[:2]; x = torch.from_numpy(np.ascontiguousarray(img)).cuda().permute(2, 0, 1)[None].float() / 255
    x = torch.nn.functional.interpolate(x, size=(288, 512), mode="bilinear")
    x = (x - torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)) / torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
    with torch.no_grad(): lg = _SEG["m"](pixel_values=x.half()).logits
    lab = torch.nn.functional.interpolate(lg.float(), size=(H, W), mode="bilinear").argmax(1)[0].cpu().numpy()
    _SEG[key] = lab; return lab

class WorldMaintainer:
    def __init__(self, case, v7c, cond, t7, dep_all, pos_all, R_all, K, every=10, min_frag=600):
        w = np.load(f"{case}/world_px.npz"); self.Zc = w["z"].astype(np.float32); self.Tc = w["tier"]
        self.v7c, self.cond, self.t7, self.dep, self.pos, self.R = v7c, cond, t7, dep_all, pos_all, R_all
        k = [float(x) for x in K]; self.f, self.cx, self.cy = k[0] * 8, k[2] * 8, k[3] * 8; self.H, self.W = v7c.shape[1:3]
        self.every, self.min_frag = every, min_frag; self.cells = {}; self.meshes = []; self.fmeshes = []; self.hpts = []; self.coarse = os.environ.get("COARSE_WORLD") == "1"; self.case_seen = self.coarse and os.environ.get("SELF_FIRST", "0") != "1"; self.scales = []; self.one = bool(os.environ.get("ONE_RECON")); self.sky = np.zeros((360, 720), bool); self.log = []
        pb = np.load(f"{case}/slots.npz")["pixel_bits"][:len(cond)].repeat(4, 1).repeat(4, 2)      # the TRUE character pixels (the lossy cond.mp4 lifts G >= 20 along terrain edges)
        ys = (np.arange(self.H) * 576 / self.H).astype(int); xs = ((np.arange(self.W) + (853 - self.W) / 2) * 1024 / 853).astype(int)
        self.chr = [(b[ys][:, xs] > 0) for b in pb]
        self.ch = [cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) for m in self.chr]   # characters + their outline
        # height code R = a * world height + b, fitted on the case's exact environment pixels of frame 0
        Pw = self._world(0, self.Zc[0]); ok = (self.Tc[0] >= 2) & ~self.ch[0] & (self.Zc[0] > 0); Rr = v7c[0][..., 0].astype(float) / 255
        ok &= (Rr > 0.51) & (Rr < 0.99)
        A = np.stack([Pw[..., 1][ok], np.ones(ok.sum())], 1); self.ab = np.linalg.lstsq(A, Rr[ok], rcond=None)[0] if ok.sum() > 500 else np.array([0.5 / 56, 0.5])

    def _world(self, t, z):
        v, u = np.mgrid[0:z.shape[0], 0:z.shape[1]] + 0.5; R = self.R[t].numpy().astype(float); pos = self.pos[t].numpy().astype(float)
        return pos + ((u - self.cx) / self.f * z)[..., None] * R[0] - ((v - self.cy) / self.f * z)[..., None] * R[1] + z[..., None] * R[2]

    def _hfield(self):
        """the world as a HEIGHT FIELD over a log-polar grid round the player: per cell the mean height of the
        reconstructed points; cells within a few metres of the player's ground (sea, plains) are flattened to it and, where an azimuth ends flat,
        continued flat to the horizon; only cells clearly above it (islands, hills) stand up. -> one mesh"""
        P = np.concatenate(self.hpts); c0 = self.pos[0].numpy().astype(np.float64); d = P[:, [0, 2]] - c0[[0, 2]]
        r = np.linalg.norm(d, axis=1); az = np.arctan2(d[:, 1], d[:, 0])
        NR, NA, R0, R1 = 150, 720, 1.0, 4000.0
        ir = np.clip((np.log(np.maximum(r, R0)) - np.log(R0)) / (np.log(R1) - np.log(R0)) * NR, 0, NR - 1).astype(int); ia = ((az + np.pi) / (2 * np.pi) * NA).astype(int) % NA
        near = r < 8; g0 = float(np.median(P[near, 1])) if near.sum() > 50 else float(np.median(P[:, 1]))
        S = np.zeros((NR, NA)); N = np.zeros((NR, NA)); np.add.at(S, (ir, ia), P[:, 1]); np.add.at(N, (ir, ia), 1)
        H = np.where(N > 0, S / np.maximum(N, 1), np.nan); rc = R0 * np.exp((np.arange(NR) + 0.5) / NR * (np.log(R1) - np.log(R0)))
        thr = np.maximum(2.0, 0.07 * rc)[:, None]                                              # "clearly above the ground" grows with distance (depth noise)
        flat = np.isfinite(H) & (H - g0 < thr); H = np.where(flat, g0, H)
        from scipy import ndimage as ndi
        for a in range(NA):                                                                     # along each azimuth: fill gaps inside the observed range,
            col = H[:, a]; ok = np.isfinite(col)
            if ok.sum() < 3: continue
            idx = np.nonzero(ok)[0]; col = np.interp(np.arange(NR), idx, col[idx]); last = idx[-1]
            if np.all(np.abs(H[idx[-3:], a] - g0) < 1e-6): col[last:] = g0; last = NR - 1        # ... and if it ENDS flat (sea / plain): flat to the horizon
            col[last + 1:] = np.nan; col[:idx[0]] = np.nan; H[:, a] = col
        Hs = ndi.median_filter(np.nan_to_num(H, nan=-1e4), size=3, mode="wrap"); H = np.where(np.isfinite(H), np.where(Hs > -1e3, Hs, H), np.nan)
        A = (np.arange(NA) + 0.5) / NA * 2 * np.pi - np.pi
        X = c0[0] + rc[:, None] * np.cos(A)[None]; Z = c0[2] + rc[:, None] * np.sin(A)[None]
        V = np.stack([X, np.nan_to_num(H, nan=g0), Z], -1).astype(np.float32); m = np.isfinite(H)
        idx = np.arange(NR * NA).reshape(NR, NA); a_, b_ = idx[:-1], np.roll(idx, -1, 1)[:-1]; c_, d_ = idx[1:], np.roll(idx, -1, 1)[1:]
        mm = m[:-1] & np.roll(m, -1, 1)[:-1] & m[1:] & np.roll(m, -1, 1)[1:]
        F = np.concatenate([np.stack([a_[mm], b_[mm], c_[mm]], 1), np.stack([b_[mm], d_[mm], c_[mm]], 1)]).astype("i4")
        self.log.append(("hfield", f"g0 {g0:.2f} cells {int(m.sum())} flat {int(flat.sum())} raised {int((np.isfinite(H) & (H - g0 > 0.5)).sum())}"))
        return V.reshape(-1, 3), F

    def _grown(self, ts, fused=False):
        if not self.meshes: return np.zeros((len(ts), self.H, self.W), np.float32)
        if fused and self.hpts and os.environ.get("HFIELD", "1") == "1":
            if getattr(self, "_hf", None) is None or self._hf_n != len(self.hpts): self._hf = self._hfield(); self._hf_n = len(self.hpts)
            return MD.render([self._hf], self.pos[ts].numpy(), self.R[ts].numpy(), self.f, self.cx, self.cy, self.H, self.W)
        if fused and self.fmeshes and os.environ.get("FUSE", "1") == "1":                 # every keyframe's surface rendered on its own,
            Z = np.stack([MD.render([m], self.pos[ts].numpy(), self.R[ts].numpy(), self.f, self.cx, self.cy, self.H, self.W) for m in self.fmeshes])   # [K,T,H,W]
            Zm = np.where(Z > 0, Z, np.inf); z0 = Zm.min(0)                                 # the front-most surface, then the median of the views that agree with it
            agree = (Z > 0) & (Z < 1.3 * z0[None]); Za = np.where(agree, Z, np.nan)
            with np.errstate(all="ignore"): zf = np.nanmedian(Za, 0)
            return np.where(np.isfinite(zf), zf, 0.0).astype(np.float32)
        return MD.render(self.meshes, self.pos[ts].numpy(), self.R[ts].numpy(), self.f, self.cx, self.cy, self.H, self.W)

    def _cur(self, t, grown):
        if self.one: return grown
        zc = np.where(self.Tc[t] >= 2, self.Zc[t], 0.0)
        return np.where((grown > 0) & ((zc == 0) | (grown < zc)), grown, zc)

    # SELF_FIRST=1: the first frame's world is estimated by this code too (same rules as the later parts), coarse tier kept
    def _recon(self, made, fs):
        """MapAnything on the keyframes jointly, our cameras (and view 0's world depth) given -> per view camera-frame points on its model grid"""
        import tempfile, subprocess
        z0 = (np.where(self.Tc[fs[0]] >= 2, self.Zc[fs[0]], 0.0) if ((self.one and not self.meshes) or self.coarse) else self._cur(fs[0], self._grown([fs[0]])[0])).astype(np.float32)   # metric anchor
        def c2w(t):
            R = self.R[t].numpy().astype(float); M = np.eye(4); M[:3, 0] = R[0]; M[:3, 1] = -R[1]; M[:3, 2] = R[2]; M[:3, 3] = self.pos[t].numpy(); return np.diag([1, 1, -1, 1]) @ M
        Kp = np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1]], np.float32)
        with tempfile.TemporaryDirectory() as d:
            dN = np.stack([np.where(self.ch[t], 0.0, self._cur(t, self._grown([t])[0])).astype(np.float32) for t in fs]) if os.environ.get("ANCHOR_ALL", "0") == "1" else None
            np.savez(f"{d}/in.npz", img=np.stack([made[t] for t in fs]), K=Kp, c2w=np.stack([c2w(t) for t in fs]), depth0=z0, **({"depthN": dN} if dN is not None else {}))
            da3 = os.environ.get("RECON_MODEL") == "da3"                                     # Depth Anything 3 instead of MapAnything (same interface)
            if not da3 and os.environ.get("RECON_RESIDENT") == "1":                          # one resident MapAnything (wizard.world.workers.recon_mapany --serve) on MAINT_CVD
                o_, lines = _mapany_server().run(f"{d}/in.npz", f"{d}/out.npz"); o = np.load(f"{d}/out.npz")
                self.log.append(("recon", [l for l in lines if "TIME" in l])); self._Km = o["Km"]; return o["ptscam"], o["mask"], o["conf"]
            r = subprocess.run([PY_DA3 if da3 else PY_MAPANY, os.path.dirname(os.path.abspath(__file__)) + ("/workers/recon_da3.py" if da3 else "/workers/recon_mapany.py"), f"{d}/in.npz", f"{d}/out.npz"], capture_output=True, text=True, env={**os.environ, **({"CUDA_VISIBLE_DEVICES": os.environ["MAINT_CVD"]} if os.environ.get("MAINT_CVD") else {})})
            assert "MAPANY_DONE" in r.stdout, r.stderr[-3000:]
            o = np.load(f"{d}/out.npz"); self.log.append(("recon", [l for l in r.stdout.splitlines() if "TIME" in l]))
            self._Km = o["Km"]; return o["ptscam"], o["mask"], o["conf"]

    def update_recon(self, made, s0, s1, t_from):
        """one consistent reconstruction of the window's keyframes (MapAnything), then the same ordered growth: each view adds what the world lacks"""
        fs = [s0] + [f for f in range(s0 + self.every, s1 + 1, self.every) if f < len(made)]
        PC, MK, CF = self._recon(made, fs)
        for i, f in enumerate(fs):
            x, y, z = PC[i][..., 0].astype(np.float64), PC[i][..., 1].astype(np.float64), PC[i][..., 2].astype(np.float64)
            ok = (z > 0.05) & (z < 1000)                                                    # MapAnything's mask alone drops far hazy mountains; sky is decided below
            km = self._Km[i]; gv, gu = np.mgrid[0:z.shape[0], 0:z.shape[1]] + 0.5                 # the grid's own rays -> our pixel (valid for sky / infinite depth too)
            u = self.cx + self.f * (gu - km[0, 2]) / km[0, 0]; v = self.cy + self.f * (gv - km[1, 2]) / km[1, 1]
            fin = np.isfinite(x) & np.isfinite(y) & np.isfinite(z); x, y, z = [np.where(fin, a_, 0.0) for a_ in (x, y, z)]
            ins = ok & (u >= -1e9) & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H); ui = np.clip(u, 0, self.W - 1).astype(int); vi = np.clip(v, 0, self.H - 1).astype(int)
            act = cv2.dilate((self.cond[f][..., 1] >= 20).astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
            p, yw = MV.dir_grid(self.R[f].numpy().astype(float), self.H, self.W, self.f, self.cx, self.cy); up = p >= 180
            skypx = up & sky_colour(made[f]); skypx = top_connected(skypx)                   # sky in the picture: sky-coloured, above the horizon, from the top edge
            cur = self._cur(f, self._grown([f])[0])
            tok = (self.Tc[f][vi, ui] != 0) if self.one else (self.Tc[f][vi, ui] == 1)         # one world: every surface comes from the reconstruction
            kn_g = (self.Tc[f] >= 2)[vi, ui]                                                # the case knows geometry there: never sky
            sky_g = skypx[vi, ui] & (~MK[i] | (z >= 1000) | (z <= 0.05)) & ~kn_g                                           # sky only where the colour AND MapAnything agree (hazy blue mountains are not sky)
            if os.environ.get("SKYLINE", "1") == "1":                                         # one clean skyline per keyframe
                from scipy import ndimage as ndi
                hg, wg = z.shape; geo = (ok | kn_g) & ~sky_g
                geo = cv2.morphologyEx(geo.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
                n_, lab = cv2.connectedComponents(geo.astype(np.uint8)); grounded = np.unique(lab[int(0.75 * hg):][geo[int(0.75 * hg):]])   # floating pieces (clouds) are not the ground
                geo = np.isin(lab, grounded[grounded > 0]); geo = ndi.binary_fill_holes(geo)
                top = np.where(geo.any(0), geo.argmax(0), hg).astype(np.float32)               # the skyline row per column, smoothed along the horizon
                top_s = ndi.median_filter(top, size=15, mode="nearest"); top_s = cv2.GaussianBlur(top_s[None], (0, 0), 3)[0]
                geo = (np.arange(hg)[:, None] >= top_s[None, :]) & (np.arange(hg)[:, None] >= 0) & (top_s[None, :] < hg)
                # depth inside the clean silhouette: the reconstruction where valid, else filled from the neighbours (disparity); far range smoothed against noise
                zv = ok & ~sky_g; disp = np.where(zv, 1.0 / np.maximum(z, 1e-3), 0.0).astype(np.float32)
                _, (iy, ix) = ndi.distance_transform_edt(~zv, return_indices=True); dfill = np.where(zv, disp, disp[iy, ix])
                sm = cv2.bilateralFilter(dfill, 9, float(np.percentile(dfill[geo], 50) * 0.15 + 1e-4) if geo.any() else 1e-3, 5)
                far = np.clip((z - 60.0) / 140.0, 0, 1); dfin = np.where(geo, far * sm + (1 - far) * dfill, 0.0)
                zn = np.where(dfin > 0, 1.0 / np.maximum(dfin, 1e-6), 0.0); rr = np.where(z > 0.05, zn / np.maximum(z, 1e-3), 0.0)
                rays = np.stack([(gu - km[0, 2]) / km[0, 0], (gv - km[1, 2]) / km[1, 1]], -1)
                x, y, z = rays[..., 0] * zn, rays[..., 1] * zn, zn
                ok = geo & (z > 0.05) & (z < 1000); ins = ok & (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
                sky_g = ~geo & (v < self.H) & (u >= 0) & (u < self.W) & up[vi, ui]
            new = ins & ~act[vi, ui] & ~sky_g & tok & (cur[vi, ui] == 0)
            cand = ins & ~act[vi, ui] & ~sky_g; new = new | (cv2.morphologyEx(new.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool) & cand & tok & (cur[vi, ui] == 0))   # close pin holes inside a surface
            n_, lab, st, _ = cv2.connectedComponentsWithStats(new.astype(np.uint8)); keep = np.zeros(n_, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= self.min_frag // 6; new = keep[lab]
            if new.sum() < 50: self.log.append((f, "nothing new")); continue
            # the reconstruction vs the world it grows from: log depth ratio where both see the same surface,
            # spread smoothly over this view (normalized convolution); new surfaces are corrected by it, so they meet the known world without a step
            kd = cur[vi, ui]; both = ins & (kd > 0) & ~act[vi, ui] & ~skypx[vi, ui] & (z > 0.05)
            if both.sum() > 200 and (not self.one or os.environ.get("STITCH_ONE", "0") == "1"):   # one world too: each new strip meets the strips before it
                lr = np.where(both, np.log(np.maximum(kd, 1e-3) / np.maximum(z, 1e-3)), 0.0); lr = np.where(both, np.clip(lr, -1.5, 1.5), 0.0); glob = float(np.median(lr[both]))
                sig = 0.06 * max(z.shape); num = cv2.GaussianBlur((lr - glob) * both, (0, 0), sig); den = cv2.GaussianBlur(both.astype(np.float64), (0, 0), sig)
                corr = np.exp(glob + num / np.maximum(den, 1e-6) * np.clip(den / 0.05, 0, 1))         # local near known surfaces, the global ratio far from them
                x, y, z = x * corr, y * corr, z * corr; self.log.append((f, f"stitch ratio median {np.exp(glob):.3f}"))
            R = self.R[f].numpy().astype(float); Pw = self.pos[f].numpy()[None, None] + x[..., None] * R[0] - y[..., None] * R[1] + z[..., None] * R[2]
            ring = cv2.dilate(new.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~new & ins & (cur[vi, ui] > 0)
            zr = np.where(ring, cur[vi, ui], z); Pw = np.where(ring[..., None], self.pos[f].numpy()[None, None] + (x / np.maximum(z, 1e-3) * zr)[..., None] * R[0] - (y / np.maximum(z, 1e-3) * zr)[..., None] * R[1] + zr[..., None] * R[2], Pw)
            self.meshes.append(MD.grid_mesh(Pw, new | ring, zr, ratio=1.12, step=1))
            if self.one: self.hpts.append(Pw[ins & ~act[vi, ui]].reshape(-1, 3)[::2].astype(np.float32))
            if self.one: self.fmeshes.append(MD.grid_mesh(Pw, ins & ~act[vi, ui], z, ratio=1.12, step=1))   # the whole clean surface of this keyframe, fused at render time
            sa = np.zeros((self.H, self.W), np.uint8); sa[vi[sky_g], ui[sky_g]] = 1; sa = (cv2.dilate(sa, np.ones((5, 5), np.uint8)) > 0) & skypx   # the agreed sky, in pixels
            g = np.zeros((360, 720), np.uint8); sk = sa & (self.Tc[f] == 1); g[p[sk], yw[sk]] = 1; self.sky |= cv2.dilate(g, np.ones((3, 3), np.uint8)).astype(bool)
            self.log.append((f, f"new {int(new.sum())} sky {int(sa.sum())}"))
        if os.environ.get("MW_DUMP"): np.savez(os.environ["MW_DUMP"], **{f"v{i}": m[0] for i, m in enumerate(self.meshes)}, **{f"f{i}": m[1] for i, m in enumerate(self.meshes)})
        return self._paint(t_from)

    # the world = FROZEN meshes. Near field (<= NEAR m along the view) = the joint MapAnything reconstruction; ground / water = the ray hitting the
    # player's ground plane (flat to the horizon); far raised content (hills, islands) = a backdrop ring at BD m keeping its outline; sky = black.
    # A window's keyframes are fused (median) first, then each keyframe adds only what the world does not have yet; nothing seen is ever changed.
    NEAR, BD = 80.0, 300.0

    def _trim(self, mesh, f):
        """no long slivers: a triangle whose longest edge is > TRIM_REL of its distance from the keyframe camera is dropped, unless all three
        corners lie on the ground plane (the plane may stretch to the horizon) -- the slabs across the sky were such slivers seen from elsewhere"""
        V, F_ = mesh; rel = float(os.environ.get("TRIM_REL", "0.15"))
        if rel <= 0 or len(F_) == 0: return mesh
        P = V[F_]; e = np.linalg.norm(P - P[:, [1, 2, 0]], axis=-1).max(1); d = np.linalg.norm(P.mean(1) - self.pos[f].numpy()[None], axis=-1)
        flat = (np.abs(P[..., 1] - self.g0) < 0.5).all(1)
        return V, F_[(e <= rel * np.maximum(d, 1e-3)) | flat]

    def _frozen(self, ts):
        if not self.wmeshes: return np.zeros((len(ts), self.H, self.W), np.float32)
        return MD.render(self.wmeshes, self.pos[ts].numpy(), self.R[ts].numpy(), self.f, self.cx, self.cy, self.H, self.W)

    def _layered(self, PC, MK, i, f, made, km):
        """one keyframe's grid -> (ray grid u, v; final depth zf with near / ground / backdrop; masks)"""
        from scipy import ndimage as ndi
        x, y, z = [PC[i][..., k].astype(np.float64) for k in range(3)]; h, w = z.shape
        gv, gu = np.mgrid[0:h, 0:w] + 0.5; rx = (gu - km[0, 2]) / km[0, 0]; ry = (gv - km[1, 2]) / km[1, 1]
        u = self.cx + self.f * rx; v = self.cy + self.f * ry; fin = np.isfinite(z) & (z > 0.05); z = np.where(fin, z, 0.0)
        inb = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H); ui = np.clip(u, 0, self.W - 1).astype(int); vi = np.clip(v, 0, self.H - 1).astype(int)
        R = self.R[f].numpy().astype(np.float64); c = self.pos[f].numpy().astype(np.float64)
        dw = rx[..., None] * R[0] - ry[..., None] * R[1] + R[2]                              # world ray with unit forward component: point = c + t * dw, t = view depth
        p, yw = MV.dir_grid(R, self.H, self.W, self.f, self.cx, self.cy); up = p >= 180
        kn = (self.Tc[f] >= 2)[vi, ui] & (self.Zc[f][vi, ui] > 0) if (f == 0 and (self.case_seen or not self.coarse)) else np.zeros_like(fin)
        if SEG_ON:                                                                             # semantic sky (SegFormer ADE20K): pale hazy mountains are not sky
            lab = seg_labels(made[f]); skypx = top_connected(lab == 2); water = np.isin(lab, WATER_IDS); seglab = lab
            sky = inb & skypx[vi, ui] & ~kn
        else:
            skypx = top_connected(up & sky_colour(made[f])); water = np.zeros((self.H, self.W), bool)
            sky = inb & skypx[vi, ui] & (~MK[i] | (z >= float(os.environ.get("SKY_MIN_M", "30"))) | ~fin) & ~kn                    # sky-coloured + from the top + not near: sky (hazy skies get finite depth from the model)
        geo = inb & fin & ~sky                                                                 # clean skyline: grounded, no floating pieces, smoothed per column
        geo = cv2.morphologyEx(geo.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
        n_, lab = cv2.connectedComponents(geo.astype(np.uint8)); gr = np.unique(lab[int(0.75 * h):][geo[int(0.75 * h):]]); geo = np.isin(lab, gr[gr > 0]); geo = ndi.binary_fill_holes(geo)
        top = np.where(geo.any(0), geo.argmax(0), h).astype(np.float32); top = cv2.GaussianBlur(ndi.median_filter(top, size=15, mode="nearest")[None], (0, 0), 3)[0]
        geo = (np.arange(h)[:, None] >= top[None, :]) & inb & (top[None, :] < h); sky = inb & ~geo & up[vi, ui]
        zv = fin & geo; disp = np.where(zv, 1.0 / np.maximum(z, 1e-3), 0.0)                     # depth inside the silhouette: reconstruction, holes from neighbours
        _, (iy, ix) = ndi.distance_transform_edt(~zv, return_indices=True); z = np.where(zv, z, np.where(geo, 1.0 / np.maximum(disp[iy, ix], 1e-6), 0.0))
        tg = np.where(dw[..., 1] < -1e-6, (self.g0 - c[1]) / np.minimum(dw[..., 1], -1e-6), np.inf)    # the ray meets the ground plane
        gsem = vsem = np.zeros_like(geo)
        if SEG_ON: L_ = seglab[vi, ui]; gsem = geo & (np.isin(L_, GROUND_IDS) | water[vi, ui]); vsem = geo & np.isin(L_, VERT_IDS)
        if GROUND_SCALE and f != 0:                                        # metric scale from the ground plane, not from MapAnything:
            gref = gsem & np.isfinite(tg) & (tg < 40) & zv                                     # views that share nothing with frame 0 come back 10x too small
            if gref.sum() > 300: s_g = float(np.median(tg[gref] / np.maximum(z[gref], 1e-3))); z = z * float(np.clip(s_g, 0.2, 20)); self.log.append((f, f"ground scale x{s_g:.2f}"))
        if os.environ.get("GUIDE", "1") == "1" and f != 0 and getattr(self, "wmeshes", None):   # where the world already has surfaces,
            gz = self._frozen([f])[0][vi, ui]; mg = (gz > 0) & zv & ~sky & ~self.ch[f][vi, ui]      # fit this view to it: disparity scale + shift, RANSAC
            if mg.sum() > 800:
                X = 1.0 / np.maximum(z[mg], 1e-3); Y = 1.0 / gz[mg]; rng = np.random.default_rng(f); best = (0, 1.0, 0.0)
                for _ in range(200):
                    i2 = rng.choice(len(X), 2, replace=False); dx = X[i2[0]] - X[i2[1]]
                    if abs(dx) < 1e-6: continue
                    a_ = (Y[i2[0]] - Y[i2[1]]) / dx; b_ = Y[i2[0]] - a_ * X[i2[0]]
                    if a_ <= 0: continue
                    n_in = int((np.abs(a_ * X + b_ - Y) < 0.15 * Y + 1e-3).sum())
                    if n_in > best[0]: best = (n_in, a_, b_)
                if best[0] > 0.3 * len(X):
                    inl = np.abs(best[1] * X + best[2] - Y) < 0.15 * Y + 1e-3; A_ = np.stack([X[inl], np.ones(inl.sum())], 1); a_, b_ = np.linalg.lstsq(A_, Y[inl], rcond=None)[0]
                    if a_ > 0:
                        z = np.where(zv, 1.0 / np.maximum(a_ / np.maximum(z, 1e-3) + b_, 1.0 / (2 * self.BD)), z); self.log.append((f, f"world guide a{a_:.2f} b{b_:.4f} inl{inl.mean():.2f} n{len(X)}"))
        hgt = c[1] + z * dw[..., 1] - self.g0
        dxz = dw[..., [0, 2]]; o = c[[0, 2]] - self.c0[[0, 2]]; A = (dxz ** 2).sum(-1); B = 2 * (dxz * o).sum(-1); Cq = (o ** 2).sum() - self.BD ** 2
        tb = (-B + np.sqrt(np.maximum(B * B - 4 * A * Cq, 0))) / (2 * A)                      # ... and the backdrop ring
        ground = geo & ((np.abs(hgt) < np.maximum(0.6, 0.07 * z)) | water[vi, ui]) & np.isfinite(tg)   # water is the plane too               # flat ground / water, near or far: the plane itself (smooth normals)
        if GROUND_SCALE and SEG_ON: ground = geo & np.isfinite(tg) & (gsem | ((np.abs(hgt) < np.maximum(0.6, 0.07 * z)) & ~vsem))   # mountains / cliffs / buildings never lie flat
        if GROUND_SCALE and f != 0 and os.environ.get("FOOT", "0") == "1":          # what stands on the ground sits where it meets the ground:
            up_ = geo & ~ground; n_c, lab_c = cv2.connectedComponents(up_.astype(np.uint8))      # each standing part scaled so its foot = the plane's depth there
            below = np.zeros_like(up_); below[:-1] = ground[1:]; foot = up_ & below; tgb = np.full_like(tg, np.inf); tgb[:-1] = tg[1:]
            for k_ in range(1, n_c):
                mk_ = lab_c == k_; ft = mk_ & foot & np.isfinite(tgb)
                if ft.sum() < 5:                                                              # no foot on the ground: a mass on / above
                    if os.environ.get("NOFOOT", "far") == "far" and not mk_[-1].any():          # the horizon (cloud bank or far rock) -> the backdrop distance, never near
                        z = np.where(mk_, np.maximum(z, tb), z)
                    continue
                s_k = float(np.median(np.minimum(tgb[ft], 2 * self.BD) / np.maximum(z[ft], 1e-3))); z = np.where(mk_, z * np.clip(s_k, 0.2, 50), z)
            hgt = c[1] + z * dw[..., 1] - self.g0
        far = geo & ~ground & (z > self.NEAR)
        if os.environ.get("BACKDROP", "1") == "0": far = far & False                          # keep the reconstructed shape / layering of hills and islands
        zf = np.where(ground, np.minimum(tg, tb), np.where(far, np.minimum(tb, tg), z)); zf = np.where(geo, zf, 0.0)   # the ground runs out to the backdrop ring, where the backdrop stands
        cls = np.where(ground, 1, np.where(far, 2, np.where(geo, 3, 0))).astype(np.uint8)     # 1 ground 2 backdrop 3 near
        return dict(u=u, v=v, ui=ui, vi=vi, inb=inb, zf=zf, geo=geo, sky=sky, cls=cls, skypx=skypx, p=p, yw=yw, dw=dw, c=c, tg=np.minimum(tg, tb), ground=ground)

    def update_world(self, made, s0, s1, t_from):
        fs = sorted(set([0, s0] + [f for f in range(s0 + self.every, s1 + 1, self.every) if f < len(made)]))
        PC, MK, CF = self._recon(made, fs)
        if not hasattr(self, "wmeshes"):
            self.wmeshes = []; self.wm_kf = []; self.c0 = self.pos[0].numpy().astype(np.float64)
            x, y, z = [PC[0][..., k] for k in range(3)]; R = self.R[0].numpy().astype(float); P = self.pos[0].numpy() + x[..., None] * R[0] - y[..., None] * R[1] + z[..., None] * R[2]
            near = MK[0] & np.isfinite(z) & (np.linalg.norm(P[..., [0, 2]] - self.c0[[0, 2]], axis=-1) < 8); self.g0 = float(np.median(P[..., 1][near]))
            if self.coarse:                                                                  # the case's own ground under the player (one ground for both)
                Pc = self._world(0, self.Zc[0]); kc = (self.Tc[0] >= 2) & ~self.ch[0] & (self.Zc[0] > 0) & (np.arange(self.H)[:, None] > 0.8 * self.H)
                if kc.sum() > 200: self.g0 = float(np.median(Pc[..., 1][kc]))
        L = [self._layered(PC, MK, i, f, made, self._Km[i]) for i, f in enumerate(fs)]
        if self.case_seen:                                                                          # meet the case's world without a step
            for i, (f, l) in enumerate(zip(fs, L)):
                kd = np.where(self.Tc[f] >= 2, self.Zc[f], 0.0)[l["vi"], l["ui"]]; zf = l["zf"]
                both = l["inb"] & (kd > 0) & (zf > 0) & (l["cls"] == 3) & ~self.ch[f][l["vi"], l["ui"]]
                if both.sum() < 200: continue
                lr = np.where(both, np.clip(np.log(kd / np.maximum(zf, 1e-3)), -1.5, 1.5), 0.0); glob = float(np.median(lr[both]))
                sig = 0.08 * max(zf.shape); num = cv2.GaussianBlur((lr - glob) * both, (0, 0), sig); den = cv2.GaussianBlur(both.astype(np.float64), (0, 0), sig)
                corr = np.exp(glob + num / np.maximum(den, 1e-6) * np.clip(den / 0.05, 0, 1))
                l["zf"] = np.where(l["cls"] == 3, zf * corr, zf); self.log.append((f, f"stitch to the case world x{np.exp(glob):.2f}"))
        fm = []                                                                                  # this window's keyframe surfaces, each separate (to fuse)
        for i, (f, l) in enumerate(zip(fs, L)):
            act = cv2.dilate(self.chr[f].astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
            ag = act[l["vi"], l["ui"]]                                                             # behind the characters: the ground plane continues where ground surrounds them
            gfill = ag & np.isfinite(l["tg"]) & (cv2.dilate(l["ground"].astype(np.uint8), np.ones((61, 61), np.uint8)) > 0) & l["inb"]
            zf = np.where(gfill, l["tg"], l["zf"])
            bfill = np.zeros_like(gfill)
            if os.environ.get("BEHIND_FILL", "1") == "1":                                      # behind the characters, away from the ground: the surroundings continue
                from scipy import ndimage as ndi
                src = l["geo"] & ~ag & (zf > 0); dd, (jy, jx) = ndi.distance_transform_edt(~src, return_indices=True)
                bfill = ag & ~gfill & l["inb"] & (dd < 0.15 * max(zf.shape)) & ~l["sky"]
                zf = np.where(bfill, zf[jy, jx], zf)
            Pw = l["c"] + zf[..., None] * l["dw"]; m = (l["geo"] & ~ag) | gfill | bfill; l["zf"] = zf
            fm.append(self._trim(MD.grid_mesh(Pw.astype(np.float32), m, zf, ratio=1.15, step=1, loose=(l["cls"] == 1) | (l["cls"] == 2) | gfill), f))   # far rows near the horizon grow fast in depth
        stats = []
        for i, (f, l) in enumerate(zip(fs, L)):                                                 # ordered growth with the FUSED depth, only where the world is empty
            if f == 0 and s0 > 0 and os.environ.get("REGROW0", "0") != "1": continue          # frame 0 is in every joint recon only as the anchor; growing it again
                                                                                                 # added the other keyframes' fused surfaces as slivers 0.3-1 m in front of its camera
            Z = np.stack([MD.render([m_], self.pos[[f]].numpy(), self.R[[f]].numpy(), self.f, self.cx, self.cy, self.H, self.W)[0] for m_ in fm])
            Zm = np.where(Z > 0, Z, np.inf); z0 = Zm.min(0); ag = (Z > 0) & (Z < 1.3 * z0[None])
            with np.errstate(all="ignore"): zfu = np.nan_to_num(np.nanmedian(np.where(ag, Z, np.nan), 0))
            cur = self._frozen([f])[0]
            if self.case_seen: kz = np.where(self.Tc[f] >= 2, self.Zc[f], 0.0); cur = np.where(cur > 0, cur, kz)          # the case's world counts as seen
            act = cv2.dilate(self.chr[f].astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
            new = (zfu > 0) & (cur == 0) & (True if os.environ.get("BEHIND_FILL", "1") == "1" else ~self.ch[f]) & ((self.Tc[f] == 1) if self.case_seen else True)   # the fused surfaces already leave the characters out, except the filled background behind them
            n_, lab, st, _ = cv2.connectedComponentsWithStats(new.astype(np.uint8)); keep = np.zeros(n_, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= 200; new = keep[lab]
            if new.sum() >= 200:
                ring = cv2.dilate(new.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~new & (cur > 0)
                if self.case_seen and os.environ.get("EDGE_BLEND", "1") == "1":                       # next to the case's world, continue ITS surface
                    zfu, new = self._edge_blend(f, zfu, new, kz, cur)
                    ring = cv2.dilate(new.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~new & (cur > 0)
                zr = np.where(new, zfu, np.where(ring, cur, 0.0)); Pz = self._world(f, zr)
                if self.case_seen and os.environ.get("SKY_MATCH", "0") == "1": Pz = self._skyline_match(f, Pz, new, kz)
                lo = (np.abs(Pz[..., 1] - self.g0) < np.maximum(0.6, 0.07 * zr)) | (zr > 0.9 * self.BD)      # ground plane or backdrop
                self.wmeshes.append(self._trim(MD.grid_mesh(Pz.astype(np.float32), new | ring, zr, ratio=1.15, step=2, loose=lo), f)); self.wm_kf.append(f)
            sk = np.zeros((self.H, self.W), np.uint8); sk[l["vi"][l["sky"]], l["ui"][l["sky"]]] = 1; sk = (cv2.dilate(sk, np.ones((5, 5), np.uint8)) > 0) & l["skypx"]
            g = np.zeros((360, 720), np.uint8); g[l["p"][sk], l["yw"][sk]] = 1; self.sky |= cv2.dilate(g, np.ones((3, 3), np.uint8)).astype(bool)
            stats.append(f"{f}:new{int(new.sum())}/g{int((l['cls'] == 1).sum())}/b{int((l['cls'] == 2).sum())}/n{int((l['cls'] == 3).sum())}")
        self.log.append(("world", " ".join(stats))); self.meshes = self.wmeshes                # (_paint reads self.meshes)
        return self._paint_world(t_from)

    def _edge_blend(self, f, zfu, new, kz, cur, D=None):
        """in a band of D px beside the case's world (in this keyframe) the new part takes the case's depth (same row, nearest case pixel) and
        the case's silhouette (column top), fading (cosine) to the reconstruction's own depth / top; so the two are one surface in 3D, joined from any view"""
        from scipy import ndimage as ndi
        D = D or int(os.environ.get("EDGE_D", "160")); H, W = new.shape
        cm = (kz > 0) & ~self.ch[f]
        if cm.sum() < 200 or new.sum() < 200: return zfu, new
        _, (iy, ix) = ndi.distance_transform_edt(~cm, sampling=[1000.0, 1.0], return_indices=True)   # nearest case pixel IN THE SAME ROW (rows weigh 1000x)
        same = iy == np.arange(H)[:, None]; dx = np.abs(ix - np.arange(W)[None, :]).astype(np.float64)
        wpx = np.where(same & (dx < D), 0.5 + 0.5 * np.cos(np.pi * np.clip(dx / D, 0, 1)), 0.0)
        zc = kz[iy, ix]
        zfu = zfu.copy()
        m = new & (wpx > 0) & (zfu > 0) & (zc > 0)
        zfu[m] = np.exp(wpx[m] * np.log(zc[m]) + (1 - wpx[m]) * np.log(zfu[m]))
        hor = int(self.cy); up = np.arange(H)[:, None] < hor
        n_, lab = cv2.connectedComponents(cm.astype(np.uint8)); gl = np.unique(lab[hor:][cm[hor:]]); cg = np.isin(lab, gl[gl > 0])   # grounded case geometry
        cg = cm if os.environ.get("EB_GROUNDED", "0") != "1" else cg                    # all the case's geometry beside (its far layer may be cut off from the ground in 2D)
        ctop = np.where((cg & up).any(0), (cg & up).argmax(0), H); cc = np.where(ctop < H)[0]
        if len(cc) == 0: return zfu, new
        ctr = ctop.astype(np.float64).copy(); ctr[cc] = ndi.median_filter(ctop[cc].astype(np.float64), size=15, mode="nearest")   # the edge column itself is ragged
        ntop = np.where(new.any(0), new.argmax(0), H); new = new.copy(); grown = cut = 0
        for x in np.where((ctop >= H) & (ntop < H))[0]:                                      # columns the new part fills beside the case's hills
            j = cc[np.argmin(np.abs(cc - x))]; d = abs(int(j) - int(x))
            if d >= D: continue
            tc = float(np.percentile(ctop[cc[np.abs(cc - j) <= 20]], float(os.environ.get("EB_PCT", "50")))); w = 0.5 + 0.5 * np.cos(np.pi * d / D); tgt = int(round(w * tc + (1 - w) * ntop[x]))   # the case's hill top over its last 20 columns (the very edge is ragged)
            if os.environ.get("EB_DBG") and f == 10 and x % 6 == 0: print("EBDBG", x, j, d, ctop[j], float(np.percentile(ctop[cc[np.abs(cc - j) <= 20]], float(os.environ.get("EB_PCT", "50")))), ntop[x], tgt, int(((cur[tgt:ntop[x], x] == 0) & ~self.ch[f][tgt:ntop[x], x]).sum()) if tgt < ntop[x] else 0, flush=True)
            if tgt < ntop[x]:                                                                  # the case's hills stand higher: grow the new hill up, at the case's depth there
                ys = np.arange(tgt, ntop[x]); free = (cur[ys, x] == 0) & ~self.ch[f][ys, x]; ys = ys[free]
                if len(ys) == 0: continue
                zt = np.where(kz[ys, j] > 0, kz[ys, j], zfu[ntop[x], x]); zfu[ys, x] = zt; new[ys, x] = True; grown += len(ys)
            elif tgt > ntop[x]:                                                                # the case's hills stand lower: trim the new hill's top
                ys = np.arange(ntop[x], tgt); new[ys, x] = False; cut += len(ys)
        self.log.append((f, f"edge blend: depth {int(m.sum())} px, silhouette +{grown} / -{cut} px"))
        return zfu, new

    def _skyline_match(self, f, Pz, new, kz, D=160):
        """where new geometry meets the case's world side by side, raise / lower the new hills so the skyline
        is continuous in this view, fading back to the reconstruction's own height over D px"""
        cm = (kz > 0) & ~self.ch[f]; H, W = cm.shape; cam = self.pos[f].numpy().astype(np.float64)
        Pc = self._world(f, kz); rows = np.arange(H)
        hor = int(self.cy); up = (rows < hor)[:, None]                                        # only hills (above the horizon row) take part; ground may span every column
        n_, lab = cv2.connectedComponents(cm.astype(np.uint8)); gl = np.unique(lab[hor:][cm[hor:]]); cm = np.isin(lab, gl[gl > 0])   # the case's grounded hills only (not floating pieces)
        ch_ = cm & up; nh_ = new & up
        ctop = np.where(ch_.any(0), ch_.argmax(0), H); ntop = np.where(nh_.any(0), nh_.argmax(0), H)
        cc = np.where(ctop < H)[0]
        if len(cc) == 0: return Pz
        nn = np.where((ntop < H) & (ctop >= H))[0]                                             # columns whose hills only the new geometry fills
        if len(nn) == 0: return Pz
        Pz = Pz.copy(); done = 0; skip = 0
        for x in nn:
            j = cc[np.argmin(np.abs(cc - x))]; d = abs(int(j) - int(x))
            if d > D: continue
            yc, yn = int(ctop[j]), int(ntop[x])
            col = new[:, x] & up[:, 0]                                                          # this column's new hill (above the horizon row)
            pc = Pc[yc, j]; pn = Pz[yn, x]; base = float(Pz[col, x, 1].min())                   # scale about the hill's own foot (far hills may stand below the player's ground)
            dc = np.linalg.norm((pc - cam)[[0, 2]]); dn = np.linalg.norm((pn - cam)[[0, 2]])
            if dc < 1 or dn < 1 or pn[1] - base < 0.3: skip += 1; continue
            tgt = cam[1] + (pc[1] - cam[1]) / dc * dn                                           # same elevation angle as the case's skyline next to it
            sc = np.clip((tgt - base) / (pn[1] - base), 0.2, float(os.environ.get("SKY_MATCH_MAX", "8"))); wgt = 0.5 + 0.5 * np.cos(np.pi * d / D); sc = 1.0 + (sc - 1.0) * wgt
            Pz[col, x, 1] = base + (Pz[col, x, 1] - base) * sc; done += 1
            if os.environ.get("SKY_MATCH_DBG") and f == 10 and x % 24 == 0: print("SKYDBG", x, j, yc, yn, np.round(pc, 1), np.round(pn, 1), round(base, 1), round(float(tgt), 1), round(float(sc), 3), int(col.sum()), flush=True)
        if os.environ.get("SKY_MATCH_DBG") and f == 10:
            np.savez("/tmp/skydbg10.npz", ctop=ctop, ntop=ntop, cc=cc, nn=nn, new=new, cm=cm)
        self.log.append((f, f"skyline match {done} columns, skipped {skip}"))
        return Pz

    def _paint_world(self, t_from):
        """every environment pixel of the later frames from the frozen world (one encoding); sky directions black; the rest unknown"""
        self.sky = cv2.morphologyEx(self.sky.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)).astype(bool)
        T = list(range(t_from, len(self.v7c)))
        if os.environ.get("PAINT_AHEAD"): T = T[:int(os.environ["PAINT_AHEAD"])]            # only the frames about to be made (the rest is painted by later updates)
        import time as _t; _a = _t.time(); GZ = self._frozen(T); changed = 0; self.cells = {}; _tm = dict(frozen=_t.time() - _a, crack=0.0, shade=0.0, med=0.0, rest=0.0)
        _a = _t.time()
        for i in range(len(GZ)):                                                              # thin horizontal cracks (the horizon seam): fill from the geometry above / below
            m = GZ[i] > 0; cl = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 1), np.uint8)).astype(bool) & ~m
            if cl.any(): GZ[i] = np.where(cl, cv2.dilate(GZ[i], np.ones((9, 1), np.uint8)), GZ[i])
        _tm["crack"] = _t.time() - _a
        def _one(i, t):                                                                       # one frame; frames are independent (PAINT_THREADS > 1 paints them in parallel)
            px = self.v7c[t]; env = ~self.ch[t] & ((self.Tc[t] == 1) if self.case_seen else True); new = env & (GZ[i] > 0)
            p, yw = MV.dir_grid(self.R[t].numpy().astype(float), self.H, self.W, self.f, self.cx, self.cy)
            skn = env & ~new & self.sky[p, yw] & (p >= 180); unk = env & ~new & ~skn
            zf = GZ[i]; Pw = self._world(t, zf); Rr = np.clip(self.ab[0] * Pw[..., 1] + self.ab[1], 0.5, 1.0); Bb = 1 - np.clip((zf - 0.1) / 79.9, 0, 1)
            sh = E7.shade(np.where(zf > 0, zf, 0.0), self.f, self.R[t].numpy().astype(float), zf > 0, sig1280=float(os.environ.get("MAINT_SIG", "2.0")))
            px[..., 0][new] = np.round(Rr[new] * 255).astype(np.uint8); px[..., 2][new] = np.round(Bb[new] * 255).astype(np.uint8)
            if self.coarse: px[..., 1][new] = E7.G_COARSE                                          # estimated geometry = coarse, as training's estimated worlds
            else: px[..., 1][new] = np.round(E7.G_EX0 + (E7.G_EX1 - E7.G_EX0) * sh[new]).astype(np.uint8)
            px[skn] = 0; px[unk] = (0, E7.G_UNK, 0)
            fn = new.reshape(60, 8, 104, 8).mean((1, 3)); fs_ = skn.reshape(60, 8, 104, 8).mean((1, 3)); fu = unk.reshape(60, 8, 104, 8).mean((1, 3))
            zt = np.where(new, zf, np.nan).reshape(60, 8, 104, 8).transpose(0, 2, 1, 3).reshape(60, 104, 64)
            with np.errstate(all="ignore"): zmed = np.nanmedian(zt, -1)
            cn = torch.from_numpy(fn > 0.5); cs = torch.from_numpy(fs_ > 0.5); cu = torch.from_numpy((fu > 0.5) & ~(fn > 0.5) & ~(fs_ > 0.5))
            if self.case_seen: k1 = torch.from_numpy(self.t7[t].numpy() == 1); cn = cn & k1; cs = cs & k1; cu = cu & k1
            self.t7[t][cn] = 2.0 if self.coarse else 3.0; self.dep[t][cn] = torch.from_numpy(np.nan_to_num(zmed)).float()[cn]; self.t7[t][cs] = 0.0; self.dep[t][cs] = 0.0; self.t7[t][cu] = 1.0
            return t, (cn, cs), int(new.sum() + skn.sum())
        _a = _t.time(); nth = int(os.environ.get("PAINT_THREADS", "1"))
        if nth > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(nth) as ex: res = list(ex.map(_one, range(len(T)), T))
        else: res = [_one(i, t) for i, t in enumerate(T)]
        for t, c_, n_ in res: self.cells[t] = c_; changed += n_
        _tm["rest"] = _t.time() - _a; del _tm["shade"], _tm["med"]; self.log.append(("paint_time", {k: round(v, 2) for k, v in _tm.items()}))
        return changed

    def update(self, made, s0, s1, t_from):
        fs = [f for f in range(s0 + self.every, s1 + 1, self.every) if f < len(made)]
        Z, OK = MV.moge_depth(np.stack([made[f] for f in fs]))
        for i, f in enumerate(fs):
            cur = self._cur(f, self._grown([f])[0]); z = Z[i].astype(np.float64); ok = OK[i] & np.isfinite(z) & (z > 0)
            act = cv2.dilate((self.cond[f][..., 1] >= 20).astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
            zk = np.where(self.Tc[f] >= 2, self.Zc[f], 0.0)                              # scale from the case's NEAR known geometry only (the ground round the player,
            sel = ok & ~act & (zk > 0) & (zk < 25.0)                                      # which the picture follows); far case geometry the picture may not have drawn
            s = float(np.median(zk[sel] / z[sel])) if sel.sum() >= 300 else None
            if self.scales and (s is None or not (0.67 < s / np.median(self.scales) < 1.5)): s = float(np.median(self.scales))
            if s is None: self.log.append((f, "no near ground to scale by")); continue
            self.scales.append(s); zs = s * z
            p, yw = MV.dir_grid(self.R[f].numpy().astype(float), self.H, self.W, self.f, self.cx, self.cy); up = p >= 180
            skyc = up & (sky_colour(made[f]) & ((zs > 150) | ~ok)); sky = (~ok & up) | top_connected(skyc | (~ok & up))
            unk = (self.Tc[f] == 1) & (cur == 0)
            new = unk & ok & ~sky & ~act & (zs < 1000)
            # behind the characters: from the surroundings, where they are not sky
            fillm = unk & act & ~sky & (cv2.blur(sky.astype(np.float32), (61, 61)) < 0.3)
            zsn = np.where(new, zs, 0.0)
            if fillm.any(): zsn = E7.fill_under(zsn, fillm, new | fillm); new = new | fillm
            n_, lab, st, _ = cv2.connectedComponentsWithStats(new.astype(np.uint8)); keep = np.zeros(n_, bool); keep[1:] = st[1:, cv2.CC_STAT_AREA] >= self.min_frag; new = keep[lab]
            if new.sum() < self.min_frag: self.log.append((f, f"scale {s:.3f}: nothing new")); continue
            ring = cv2.dilate(new.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~new & (cur > 0)           # stitch to the world it touches
            zm = np.where(new, zsn, np.where(ring, cur, 0.0)); self.meshes.append(MD.grid_mesh(self._world(f, zm), new | ring, zm, ratio=1.12, step=2))
            g = np.zeros((360, 720), np.uint8); sk = sky & (self.Tc[f] == 1); g[p[sk], yw[sk]] = 1; self.sky |= cv2.dilate(g, np.ones((3, 3), np.uint8)).astype(bool)
            self.log.append((f, f"scale {s:.3f} new {int(new.sum())} sky {int(sky.sum())}"))
        return self._paint(t_from)

    def _paint(self, t_from):
        if not self.meshes: return 0
        T = list(range(t_from, len(self.v7c))); GZ = self._grown(T, fused=True); changed = 0; self.cells = {}
        for i, t in enumerate(T):
            px = self.v7c[t]; free = ((self.Tc[t] != 0) if self.one else (self.Tc[t] == 1)) & ~self.ch[t]
            new = free & (GZ[i] > 0); p, yw = MV.dir_grid(self.R[t].numpy().astype(float), self.H, self.W, self.f, self.cx, self.cy)
            skn = free & ~new & (self.Tc[t] == 1) & self.sky[p, yw] & (p >= 180)
            if new.any():
                zf = self._cur(t, GZ[i]); Pw = self._world(t, zf)
                Rr = np.clip(self.ab[0] * Pw[..., 1] + self.ab[1], 0.5, 1.0); Bb = 1 - np.clip((zf - 0.1) / 79.9, 0, 1)
                sh = E7.shade(np.where(zf > 0, zf, 0.0), self.f, self.R[t].numpy().astype(float), zf > 0, sig1280=1.0)
                px[..., 0][new] = np.round(Rr[new] * 255).astype(np.uint8); px[..., 2][new] = np.round(Bb[new] * 255).astype(np.uint8)
                px[..., 1][new] = np.round(E7.G_EX0 + (E7.G_EX1 - E7.G_EX0) * sh[new]).astype(np.uint8)
            px[skn] = 0
            fn = new.reshape(60, 8, 104, 8).mean((1, 3)); fs_ = skn.reshape(60, 8, 104, 8).mean((1, 3))
            zt = np.where(new, GZ[i], np.nan).reshape(60, 8, 104, 8).transpose(0, 2, 1, 3).reshape(60, 104, 64)
            with np.errstate(all="ignore"): zmed = np.nanmedian(zt, -1)
            cn = torch.from_numpy((fn > 0.5) & ((self.t7[t].numpy() != 0) if self.one else (self.t7[t].numpy() == 1))); cs = torch.from_numpy((fs_ > 0.5) & (self.t7[t].numpy() == 1))
            self.t7[t][cn] = 3.0; self.dep[t][cn] = torch.from_numpy(np.nan_to_num(zmed)).float()[cn]; self.t7[t][cs] = 0.0; self.dep[t][cs] = 0.0
            self.cells[t] = (cn, cs); changed += int(new.sum() + skn.sum())
        return changed
