"""inference-time world maintenance: after a window is generated, MoGe-2 on its frames -> scale-aligned to the known world ->
characters removed -> the same surface splat as training's MoGe world (wizard.data.cond_encode_gpu.splat_grid) rendered along the later frames' cameras.
Only cells that are still UNKNOWN (dark green) change: to coarse (G 77, R = height, B = depth, as build_v7) where the new surface lands,
to sky where the generated frames' MoGe saw sky in that direction (above the horizon). Tokens (tier, depth) follow."""
import os, sys, subprocess, tempfile
import numpy as np, torch, torch.nn.functional as F, cv2

import wizard.data.cond_encode_gpu as VG, wizard.data.cond_encode as E7
EXACT = not os.environ.get("MAINT_COARSE")       # maintained geometry = true geometry: exact tier with normal shading
from wizard.paths import PY_MOGE; HERE = os.path.dirname(os.path.abspath(__file__))
G_UNK, G_COARSE = 38, 77


def moge_depth(frames):
    with tempfile.TemporaryDirectory() as d:
        np.save(f"{d}/in.npy", np.ascontiguousarray(frames))
        r = subprocess.run([PY_MOGE, f"{HERE}/workers/moge_frames.py", f"{d}/in.npy", f"{d}/out.npz"], capture_output=True, text=True, env={**os.environ, "HF_HUB_OFFLINE": "1", **({"CUDA_VISIBLE_DEVICES": os.environ["MOGE_CVD"]} if os.environ.get("MOGE_CVD") else {})})   # MOGE_CVD: MoGe on the maintenance card
        assert "MOGE_DONE" in r.stdout, r.stderr[-2000:]
        o = np.load(f"{d}/out.npz"); return o["z"], o["ok"]


def dir_grid(R, H, W, f, cx, cy):
    """per pixel (pitch, yaw) bins of 0.5 deg (as build_v7.sky_dirs): pitch index 180 = horizon, larger = up"""
    v, u = np.mgrid[0:H, 0:W] + 0.5; x = (u - cx) / f; y = -(v - cy) / f
    d = x[..., None] * R[0] - y[..., None] * (-R[1]) + R[2]                                # world direction (rows right, up, forward)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    p = np.clip(np.round((np.degrees(np.arcsin(np.clip(d[..., 1], -1, 1))) + 90) * 2).astype(int), 0, 359)
    yw = ((np.degrees(np.arctan2(d[..., 0], d[..., 2])) + 180) * 2).astype(int) % 720
    return p, yw


class Maintainer:
    def __init__(self, v7c, cond, t7, dep_all, pos_all, R_all, K, actc, dev="cuda", every=10):
        self.v7c, self.cond, self.t7, self.dep, self.pos, self.R = v7c, cond, t7, dep_all, pos_all, R_all
        self.actc = actc; self.dev = dev; self.every = every; self.H, self.W = v7c.shape[1:3]
        k = [float(x) for x in K]; self.f, self.cx, self.cy = k[0] * 8, k[2] * 8, k[3] * 8          # the token grid is the 832x480 crop / 8
        self.src = []; self.msrc = []; self.sky = np.zeros((360, 720), bool)
        # the case's height code R = a * world height + b, fitted on frame 0's known cells (as build_v7.height_fit)
        P = self._world(0, np.kron(self.dep[0].numpy(), np.ones((8, 8))))
        Rr = v7c[0][..., 0].astype(float) / 255; tier_px = np.kron(self.t7[0].numpy(), np.ones((8, 8)))
        ok = (tier_px >= 2) & (Rr > 0.52) & (Rr < 0.98) & (cond[0][..., 1] < 20) & (np.kron(self.dep[0].numpy(), np.ones((8, 8))) > 0)
        A = np.stack([P[..., 1][ok], np.ones(ok.sum())], 1); self.ab = np.linalg.lstsq(A, Rr[ok], rcond=None)[0] if ok.sum() > 500 else (0.5 / 56, 0.5)
        self.log = []

    def _world(self, t, z):
        v, u = np.mgrid[0:z.shape[0], 0:z.shape[1]] + 0.5; R = self.R[t].numpy().astype(float); pos = self.pos[t].numpy().astype(float)
        return pos + ((u - self.cx) / self.f * z)[..., None] * R[0] - ((v - self.cy) / self.f * z)[..., None] * R[1] + z[..., None] * R[2]

    def update(self, made, s0, s1, t_from):
        """made: pixel frames so far; the window just made covers s0..s1; re-render the cells of frames >= t_from that are still unknown"""
        fs = [f for f in range(s0 + self.every, s1 + 1, self.every) if f < len(made)]
        Z, OK = moge_depth(np.stack([made[f] for f in fs]))
        for i, f in enumerate(fs):
            z = Z[i].astype(np.float64); ok = OK[i] & np.isfinite(z) & (z > 0)
            known = np.kron(self.dep[f].numpy(), np.ones((8, 8))); tier_px = np.kron(self.t7[f].numpy(), np.ones((8, 8)))
            act = cv2.dilate((self.cond[f][..., 1] >= 20).astype(np.uint8), np.ones((41, 41), np.uint8)).astype(bool)
            sel = ok & ~act & (known > 0) & (tier_px >= 2)
            if sel.sum() < 300: self.log.append((f, "no overlap")); continue
            s = float(np.median(known[sel] / z[sel])); zs = s * z
            near = (known > 0) & (tier_px >= 2) & (zs < 0.9 * known - 0.3)                 # nearer than the KNOWN environment = a character (the case depth of unknown cells is invented backdrop)
            okm = ok & ~act & ~cv2.dilate(near.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool) & (zs < 1000.0)   # clouds MoGe calls geometry: beyond 1 km = not a surface
            hole = (act | cv2.dilate(near.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)) & ~okm
            fillm = hole & (cv2.blur((~ok).astype(np.float32), (61, 61)) < 0.5)                # behind the characters: ground / walls around them, not sky
            if fillm.any(): zs = E7.fill_under(np.where(okm, zs, 0.0), fillm, okm | fillm); okm = okm | fillm
            Pw = self._world(f, np.where(okm, zs, 0.0))
            self.src.append((torch.from_numpy(Pw[::2, ::2]).float().to(self.dev), torch.from_numpy(okm[::2, ::2]).to(self.dev))); self.msrc.append((Pw, okm, np.where(okm, zs, 0.0)))
            sk = (~OK[i]) & ~act; sk = cv2.morphologyEx(sk.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)).astype(bool)
            p, yw = dir_grid(self.R[f].numpy().astype(float), self.H, self.W, self.f, self.cx, self.cy); sk &= p >= 180
            g = np.zeros((360, 720), np.uint8); g[p[sk], yw[sk]] = 1; self.sky |= cv2.dilate(g, np.ones((3, 3), np.uint8)).astype(bool)
            self.log.append((f, f"scale {s:.3f} pts {int(okm.sum())} sky {int(sk.sum())}"))
        if not self.src: return 0
        T = list(range(t_from, len(self.v7c))); pos = self.pos[T].to(self.dev).float(); R = self.R[T].to(self.dev).float()
        if os.environ.get("MAINT_SPLAT"):
            zc = None
            for Pg, okg in self.src:
                z_ = VG.splat_grid(Pg, okg, pos, R, self.f, self.cx, self.cy, self.H, self.W)
                zc = z_ if zc is None else torch.where((z_ > 0) & ((zc == 0) | (z_ < zc)), z_, zc)
            zc = zc.cpu().numpy()
        else:                                                                            # a triangle mesh per source frame, rasterized: smooth surfaces for the normals
            import wizard.world.mesh_depth as MD
            zc = MD.render([MD.grid_mesh(P_, o_, z_) for P_, o_, z_ in self.msrc], self.pos[T].numpy(), self.R[T].numpy(), self.f, self.cx, self.cy, self.H, self.W)
        changed = 0; self.cells = {}
        for i, t in enumerate(T):
            px = self.v7c[t]; unk = (px[..., 1] >= 30) & (px[..., 1] <= 46) & (px[..., 0] <= 4) & (px[..., 2] <= 4)
            new = unk & (zc[i] > 0)
            if new.any():
                P = self._world(t, zc[i]); Rr = np.clip(self.ab[0] * P[..., 1] + self.ab[1], 0.5, 1.0); Bb = 1 - np.clip((zc[i] - 0.1) / 79.9, 0, 1)
                px[..., 0][new] = np.round(Rr[new] * 255).astype(np.uint8); px[..., 2][new] = np.round(Bb[new] * 255).astype(np.uint8)
                if EXACT: sh = E7.shade(np.where(new, zc[i], 0.0), self.f, self.R[t].numpy().astype(float), new, sig1280=float(os.environ.get('MAINT_SIG', '2.0'))); px[..., 1][new] = np.round(E7.G_EX0 + (E7.G_EX1 - E7.G_EX0) * sh[new]).astype(np.uint8)
                else: px[..., 1][new] = G_COARSE
            p, yw = dir_grid(self.R[t].numpy().astype(float), self.H, self.W, self.f, self.cx, self.cy)
            skn = unk & ~new & self.sky[p, yw] & (p >= 180); px[skn] = 0
            # tokens: an 8x8 cell follows its majority
            fn = new.reshape(60, 8, 104, 8).mean((1, 3)); fs_ = skn.reshape(60, 8, 104, 8).mean((1, 3))
            zt = np.where(new, zc[i], np.nan).reshape(60, 8, 104, 8).transpose(0, 2, 1, 3).reshape(60, 104, 64)
            with np.errstate(all="ignore"): zmed = np.nanmedian(zt, -1)
            cn = torch.from_numpy((fn > 0.5) & (self.t7[t].numpy() == 1)); cs = torch.from_numpy((fs_ > 0.5) & (self.t7[t].numpy() == 1))
            self.t7[t][cn] = 3.0 if EXACT else 2.0; self.dep[t][cn] = torch.from_numpy(np.nan_to_num(zmed)).float()[cn]
            self.t7[t][cs] = 0.0; self.dep[t][cs] = 0.0; self.cells[t] = (cn, cs)
            changed += int(new.sum() + skn.sum())
        return changed
