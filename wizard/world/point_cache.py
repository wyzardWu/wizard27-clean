"""world maintenance as a POINT CACHE, the GEN3C recipe (nv-tlabs/GEN3C,
cosmos_predict1/diffusion/inference/cache_3d.py + camera_utils.align_depth, Apache-2.0) instead of the layered meshes of wizard.world.world_maintainer:
  * the cache starts from the case's first-frame world (GEN3C: MoGe depth of the input image)
  * each new keyframe: depth from the joint reconstruction (MapAnything / DA3, our cameras) ->
      aligned to the depth the cache renders at that camera: inverse-depth scale + shift on trimmed overlap (GEN3C "rigid"),
      then a smooth per-pixel scale map (GEN3C "non_rigid": Adam on a scale map, ARAP smoothness)
    -> depth-discontinuity pixels dropped (GEN3C reliable_depth_mask_range, window 5, ratio 0.05), sky (SegFormer) and characters dropped
    -> only pixels the cache does not cover yet are added (Voyager: world cache with point culling)
  * rendering = z-buffered point splatting in torch (no EGL), footprint from each point's source spacing, small holes closed
A keyframe with too little overlap to align falls back to the ground plane scale; with neither it is skipped (never added unscaled).
Same interface as WorldMaintainer (update_world / _frozen / _paint_world / cells), selected with MAINT=pc."""
import os, numpy as np, torch, cv2, torch.nn.functional as F
import wizard.world.world_maintainer as MW
from wizard.world.world_maintainer import WorldMaintainer, seg_labels, top_connected, GROUND_IDS, WATER_IDS
import wizard.world.maintainer as MV

DEV = "cuda"


def reliable_mask(z, window=5, ratio=0.05):
    """GEN3C's range test (below) drops grazing ground, whose depth changes fast but smoothly; a plane has inverse depth LINEAR in the
    pixel coordinates, so the test here is the second difference of inverse depth relative to it (PC_EDGE=range restores GEN3C's)"""
    if os.environ.get("PC_EDGE", "lap") == "lap":
        inv = torch.where(z > 0, 1.0 / z.clamp(min=1e-3), torch.zeros_like(z))[None, None]
        kx = torch.tensor([[[[0, 0, 0], [1, -2, 1], [0, 0, 0]]]], device=z.device, dtype=z.dtype); ky = kx.transpose(2, 3)
        lap = torch.maximum(F.conv2d(inv, kx, padding=1).abs(), F.conv2d(inv, ky, padding=1).abs())
        lap = F.max_pool2d(lap, 3, 1, 1)[0, 0]; zero = F.max_pool2d((z == 0).float()[None, None], 3, 1, 1)[0, 0] > 0
        return (lap < float(os.environ.get("PC_LAP", "0.08")) * inv[0, 0]) & (z > 0) & ~zero
    d = z[None, None]; mx = F.max_pool2d(d, window, 1, window // 2); mn = -F.max_pool2d(-d, window, 1, window // 2)
    me = F.avg_pool2d(d, window, 1, window // 2)
    return (((mx - mn) / (me + 1e-6)) < ratio)[0, 0] & (z > 0)


def align_rigid(src, tgt, m):
    """GEN3C camera_utils._align_inv_depth_to_depth: inverse depth scale + shift, 10-90% quantiles trimmed, least squares"""
    si, ti = 1.0 / src, 1.0 / tgt; q = torch.tensor([0.1, 0.9], device=src.device)
    sl, sh = torch.quantile(si[m], q); tl, th = torch.quantile(ti[m], q)
    mm = m & (si > sl) & (si < sh) & (ti > tl) & (ti < th)
    if os.environ.get("PC_ALIGN", "affine") == "scale":                                         # our cameras are given to the reconstruction, so its depth is
        a, b = float(torch.median(ti[mm] / si[mm])), 0.0                                       # metric up to one scale: no shift (a shift pulled far content near)
    else:
        A = torch.stack([si[mm], torch.ones_like(si[mm])], 1); sol = torch.linalg.lstsq(A, ti[mm][:, None]).solution
        a, b = float(sol[0, 0]), float(sol[1, 0])
    return 1.0 / torch.clamp(si * a + b, min=1e-4), a, b, int(mm.sum())


def align_nonrigid(src, tgt, m, rx, ry, iters=100, lam=0.1, k=3):
    """GEN3C camera_utils.align_depth(alignment_method="non_rigid"): per-pixel scale map, L1 on the unprojected points of the overlap + ARAP
    smoothness (camera frame: the rays rx, ry are the same for source and target)"""
    sc = torch.ones_like(src, requires_grad=True); opt = torch.optim.Adam([sc], lr=1e-3); ker = torch.ones(1, 1, k, k, device=src.device) / k ** 2
    Pt = torch.stack([rx * tgt, ry * tgt, tgt], -1)[m]
    with torch.enable_grad():
        for _ in range(iters):
            z = src * sc; Ps = torch.stack([rx * z, ry * z, z], -1)[m]
            loss = (Ps - Pt).abs().mean() + lam * (F.conv2d(sc[None, None], ker, padding=k // 2)[0, 0] - sc).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return (src * sc).detach()


class PCMaintainer(WorldMaintainer):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        v, u = np.mgrid[0:self.H, 0:self.W] + 0.5
        self.rx = torch.tensor((u - self.cx) / self.f, device=DEV, dtype=torch.float32); self.ry = torch.tensor((v - self.cy) / self.f, device=DEV, dtype=torch.float32)
        self.P = torch.zeros(0, 3, device=DEV); self.S = torch.zeros(0, device=DEV); self.wm_kf = []; self.wmeshes = []
        self.maxz = float(os.environ.get("PC_MAXZ", "3000")); self.nonrigid = os.environ.get("PC_NONRIGID", "1") == "1"

    # ---------- cache ----------
    def _cam(self, t):
        return torch.as_tensor(self.pos[t].numpy(), device=DEV, dtype=torch.float32), torch.as_tensor(self.R[t].numpy(), device=DEV, dtype=torch.float32)

    def _add(self, t, z, m):
        """world points of frame t's pixels m at view depth z; S = the spacing between neighbouring source pixels (m)"""
        c, R = self._cam(t)
        G = c + (self.rx * z)[..., None] * R[0] - (self.ry * z)[..., None] * R[1] + z[..., None] * R[2]    # [H,W,3]
        # footprint = the real 3D distance to the next pixel right / below (grazing ground spreads in depth), capped
        dx = torch.zeros_like(z); dy = torch.zeros_like(z); v = z > 0
        dx[:, :-1] = torch.where(v[:, :-1] & v[:, 1:], (G[:, 1:] - G[:, :-1]).norm(dim=-1), torch.zeros_like(z[:, :-1]))
        dy[:-1] = torch.where(v[:-1] & v[1:], (G[1:] - G[:-1]).norm(dim=-1), torch.zeros_like(z[:-1]))
        sp = torch.minimum(torch.maximum(torch.maximum(dx, dy), z / self.f), 8 * z / self.f)
        self.P = torch.cat([self.P, G[m]]); self.S = torch.cat([self.S, sp[m]]); self.wm_kf.append(int(t))

    @torch.no_grad()
    def _render1(self, t):
        H, W = self.H, self.W; out = torch.full((H * W,), float("inf"), device=DEV)
        if len(self.P) == 0: return torch.zeros(H, W, device=DEV)
        c, R = self._cam(t); d = self.P - c; zc = d @ R[2]; ok = zc > 0.05
        d, zc, s = d[ok], zc[ok], self.S[ok]
        u = self.cx + self.f * (d @ R[0]) / zc; v = self.cy - self.f * (d @ R[1]) / zc
        rad = torch.clamp(torch.ceil(0.75 * s * self.f / zc), 0, 3)                  # footprint: half the source spacing seen from here
        ui, vi = torch.floor(u).long(), torch.floor(v).long()
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                r = max(abs(dx), abs(dy)); sel = rad >= r
                if r and not sel.any(): continue
                x, y = ui[sel] + dx, vi[sel] + dy; inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
                out.scatter_reduce_(0, (y[inb] * W + x[inb]), zc[sel][inb], reduce="amin")
        z = out.view(H, W); z = torch.where(torch.isfinite(z), z, torch.zeros_like(z))
        for _ in range(int(os.environ.get("PC_FILL", "4"))):                                  # close pinholes: empty pixels with >= 5 of 8 filled neighbours
            m = (z > 0).float()[None, None]; n = F.conv2d(m, torch.ones(1, 1, 3, 3, device=DEV), padding=1)[0, 0] - m[0, 0]
            inv = torch.where(z > 0, 1.0 / z.clamp(min=1e-3), torch.zeros_like(z)); mx = F.max_pool2d(inv[None, None], 3, 1, 1)[0, 0]
            hole = (z == 0) & (n >= 5) & (mx > 0); z = torch.where(hole, 1.0 / mx.clamp(min=1e-6), z)
        return z

    def _frozen(self, ts):
        if os.environ.get("PC_BATCH") == "1" and len(ts) > 1: return self._frozen_batch(ts)
        out = []
        for t in ts:
            z = self._render1(t)
            if os.environ.get("PC_RFILL", "1") == "1":                                         # enclosed gaps (ground strips not seen yet, splat seams):
                v = z > 0; no_sky = torch.zeros_like(v)                                         # inverse depth diffused in; regions open to the image top stay empty
                z, _ = self._fill_hidden(z, v, ~v, no_sky, iters=150)
            out.append(z.cpu().numpy())
        return np.stack(out).astype(np.float32) if len(ts) else np.zeros((0, self.H, self.W), np.float32)

    def _frozen_batch(self, ts, chunk=17):
        """_frozen with the enclosed-gap diffusion run on all frames at once (the 150 iterations once per chunk, not per frame)"""
        import time as _t; out = []; tr = tf = 0.0
        for c0 in range(0, len(ts), chunk):
            _a = _t.time(); Z = self._render_b(ts[c0:c0 + chunk]); torch.cuda.synchronize(); tr += _t.time() - _a; _a = _t.time()
            if os.environ.get("PC_RFILL", "1") == "1":
                V = Z > 0; Z, _ = self._fill_hidden_b(Z, V, ~V, torch.zeros_like(V), iters=150)
            out.append(Z.cpu().numpy()); tf += _t.time() - _a
        self.log.append(("frozen_time", f"render {tr:.2f} fill {tf:.2f}"))
        return np.concatenate(out).astype(np.float32)

    @torch.no_grad()
    def _render_b(self, ts):
        """_render1 for several frames at once: one scatter per footprint offset over all of them (frame offset in the flat index)"""
        H, W = self.H, self.W; n = len(ts)
        if len(self.P) == 0: return torch.zeros(n, H, W, device=DEV)
        cs = torch.stack([self._cam(t)[0] for t in ts]); Rs = torch.stack([self._cam(t)[1] for t in ts])     # [n,3], [n,3,3]
        out = torch.full((n * H * W,), float("inf"), device=DEV)
        for i in range(n):                                                                    # per frame projection (the big [n,P] tensors would not fit), scatter batched below
            d = self.P - cs[i]; zc = d @ Rs[i, 2]; ok = zc > 0.05
            d, zc_, s = d[ok], zc[ok], self.S[ok]
            u = self.cx + self.f * (d @ Rs[i, 0]) / zc_; v = self.cy - self.f * (d @ Rs[i, 1]) / zc_
            rad = torch.clamp(torch.ceil(0.75 * s * self.f / zc_), 0, 3)
            if i == 0: U, V, Zc, Rad, Fi = [], [], [], [], []
            U.append(torch.floor(u).long()); V.append(torch.floor(v).long()); Zc.append(zc_); Rad.append(rad); Fi.append(torch.full_like(U[-1], i))
        ui, vi, zc, rad, fi = torch.cat(U), torch.cat(V), torch.cat(Zc), torch.cat(Rad), torch.cat(Fi)
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                r = max(abs(dx), abs(dy)); sel = rad >= r
                if r and not sel.any(): continue
                x, y = ui[sel] + dx, vi[sel] + dy; inb = (x >= 0) & (x < W) & (y >= 0) & (y < H)
                out.scatter_reduce_(0, fi[sel][inb] * (H * W) + y[inb] * W + x[inb], zc[sel][inb], reduce="amin")
        z = out.view(n, H, W); z = torch.where(torch.isfinite(z), z, torch.zeros_like(z))
        for _ in range(int(os.environ.get("PC_FILL", "4"))):
            m = (z > 0).float()[:, None]; nb = F.conv2d(m, torch.ones(1, 1, 3, 3, device=DEV), padding=1)[:, 0] - m[:, 0]
            inv = torch.where(z > 0, 1.0 / z.clamp(min=1e-3), torch.zeros_like(z)); mx = F.max_pool2d(inv[:, None], 3, 1, 1)[:, 0]
            hole = (z == 0) & (nb >= 5) & (mx > 0); z = torch.where(hole, 1.0 / mx.clamp(min=1e-6), z)
        return z

    def _fill_hidden_b(self, Z, valid, ch, sky, iters=300):
        """_fill_hidden over a batch [N,H,W]: per frame hole labelling as there, the diffusion batched"""
        hole = (ch | ((~valid) & ~sky)) & ~valid & ~sky; OK = torch.zeros_like(hole)
        skyn = F.max_pool2d(sky.float()[:, None], 9, 1, 4)[:, 0] > 0
        hc = hole.cpu().numpy().astype(np.uint8)
        for i in range(len(Z)):
            if not hc[i].any(): continue
            n_, lab_ = cv2.connectedComponents(hc[i]); lab_ = torch.tensor(lab_, device=DEV)
            top = torch.unique(lab_[0]); sz = torch.bincount(lab_.flatten())
            ok = (lab_ > 0) & ~torch.isin(lab_, top) & (sz[lab_] < 0.05 * Z[i].numel())
            bad = torch.unique(lab_[skyn[i] & hole[i]]); OK[i] = ok & ~torch.isin(lab_, bad[bad > 0])
        inv = torch.where(valid, 1.0 / Z.clamp(min=1e-3), torch.zeros_like(Z)); w = valid.float()
        def loop(inv, w, OK):
            k = torch.ones(1, 1, 3, 3, device=inv.device)
            for _ in range(iters):
                s_ = F.conv2d((inv * w)[:, None], k, padding=1)[:, 0]; n = F.conv2d(w[:, None], k, padding=1)[:, 0]
                upd = OK & (n > 0); inv = torch.where(upd, s_ / n.clamp(min=1e-6), inv); w = torch.where(upd, torch.ones_like(w), w)
            return inv
        if os.environ.get("PC_GRAPH") == "1":                                                  # the 150 small-kernel iterations as one graph
            B = int(os.environ.get("PC_GRAPH_B", "17")); N0 = len(Z)
            if not hasattr(self, "_graphs"): self._graphs = {}
            key = (B, inv.shape[1], inv.shape[2], iters, torch.cuda.current_device())
            if key not in self._graphs:
                si, sw, so = torch.zeros(B, *inv.shape[1:], device=inv.device), torch.zeros(B, *inv.shape[1:], device=inv.device), torch.zeros(B, *inv.shape[1:], dtype=torch.bool, device=inv.device)
                st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(st): loop(si, sw, so)
                torch.cuda.current_stream().wait_stream(st); G = torch.cuda.CUDAGraph()
                with torch.cuda.graph(G): res = loop(si, sw, so)
                self._graphs[key] = (G, si, sw, so, res)
            G, si, sw, so, res = self._graphs[key]; outs = []
            for b0 in range(0, N0, B):
                m_ = min(B, N0 - b0); si.zero_(); sw.zero_(); so.zero_()
                si[:m_].copy_(inv[b0:b0 + m_]); sw[:m_].copy_(w[b0:b0 + m_]); so[:m_].copy_(OK[b0:b0 + m_]); G.replay(); outs.append(res[:m_].clone())
            inv = torch.cat(outs)
        else: inv = loop(inv, w, OK)
        filled = OK & (inv > 0)
        return torch.where(filled, 1.0 / inv.clamp(min=1e-5), Z), filled

    def _grown(self, ts, fused=False):
        return self._frozen(ts)

    # ---------- per keyframe ----------
    def _to_grid(self, PC, km):
        """reconstruction (camera-frame points on the model grid, intrinsics km) -> view depth on our pixel grid"""
        z = PC[..., 2].astype(np.float32); z = np.where(np.isfinite(z) & (z > 0.05), z, 0.0)
        mx = (self.rx.cpu().numpy() * km[0, 0] + km[0, 2] - 0.5).astype(np.float32); my = (self.ry.cpu().numpy() * km[1, 1] + km[1, 2] - 0.5).astype(np.float32)
        return cv2.remap(z, mx, my, cv2.INTER_NEAREST, borderValue=0)

    def _sky_char(self, t, made):
        lab = seg_labels(made[t]); sky = top_connected(lab == 2)
        sky = cv2.dilate(sky.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)      # Voyager erodes the kept region by 3x3
        return sky, self.ch[t], lab

    def _ground_scale(self, t, z, lab):
        c, R = self.pos[t].numpy().astype(float), self.R[t].numpy().astype(float); rx, ry = self.rx.cpu().numpy(), self.ry.cpu().numpy()
        dwy = rx * R[0, 1] - ry * R[1, 1] + R[2, 1]; tg = np.where(dwy < -1e-6, (self.g0 - c[1]) / np.minimum(dwy, -1e-6), np.inf)
        g = (np.isin(lab, GROUND_IDS) | np.isin(lab, WATER_IDS)) & np.isfinite(tg) & (tg < 40) & (z > 0) & ~self.ch[t]
        return (float(np.median(tg[g] / z[g])), int(g.sum())) if g.sum() > 300 else (None, int(g.sum()))

    def update_world(self, made, s0, s1, t_from):
        fs = sorted(set([0, s0] + [f for f in range(s0 + self.every, s1 + 1, self.every) if f < len(made)]))
        if len(self.P) == 0:
            sky0, ch0, lab0 = self._sky_char(0, made)
            z0 = np.where(self.Tc[0] >= 2, self.Zc[0], 0.0).astype(np.float32)
            Pw = self._world(0, np.where(z0 > 0, z0, np.nan)); near = (z0 > 0) & (np.linalg.norm(Pw[..., [0, 2]] - self.pos[0].numpy()[[0, 2]], axis=-1) < 8)
            self.g0 = float(np.nanmedian(Pw[..., 1][near])) if near.sum() > 50 else float(self.pos[0].numpy()[1] - 1.6)
            zt = torch.tensor(z0, device=DEV); m = (zt > 0) & torch.tensor(~sky0 & ~ch0, device=DEV)   # the case world is clean: no edge test
            self._add(0, zt, m); self._sky_dirs(0, sky0); self.obs = [(0, torch.where(m, zt, torch.zeros_like(zt)), torch.tensor(sky0, device=DEV))]; self.log.append(("pc_init", f"frame 0 case world {int(m.sum())} pts g0 {self.g0:.2f}"))
        todo = [f for f in fs if f != 0 and f not in self.wm_kf]; import time as _t; T0 = _t.time(); T1 = T0
        if todo:
            PC, MK, CF = self._recon(made, fs); stats = []; T1 = _t.time()
            for i, f in enumerate(fs):
                if f not in todo: continue
                sky, ch, lab = self._sky_char(f, made)
                z = self._to_grid(PC[i], self._Km[i]); keep = (z > 0) & ~sky & ~ch
                zt = torch.tensor(z, device=DEV); tgt = self._render1(f); kt = torch.tensor(keep, device=DEV)
                ov = kt & (tgt > 0) & (zt > 0); how = None
                if int(ov.sum()) > 2000:
                    za, a, b, n = align_rigid(torch.where(zt > 0, zt, torch.ones_like(zt)), torch.where(tgt > 0, tgt, torch.ones_like(tgt)), ov)
                    za = torch.where(zt > 0, za, torch.zeros_like(za))
                    if a <= 0.2 or a > 5: ov = ov & False                                           # a flipped / absurd fit: not an alignment
                if int(ov.sum()) > 2000:
                    if self.nonrigid: za = align_nonrigid(torch.where(za > 0, za, torch.ones_like(za)), torch.where(tgt > 0, tgt, torch.ones_like(tgt)), ov, self.rx, self.ry) * (za > 0)
                    how = f"align a{a:.3f} b{b:.4f} n{n}"
                else:
                    s, n = self._ground_scale(f, z, lab)
                    if s is not None: za = zt * float(np.clip(s, 0.05, 50)); how = f"ground x{s:.2f} n{n}"
                if how is None: stats.append(f"{f}:skip(ov{int(ov.sum())})"); continue
                rel = reliable_mask(za); valid = kt & rel & (za > 0) & (za < self.maxz)
                if os.environ.get("PC_BEHIND", "1") == "1":                                      # what the characters hide (the camera follows the player, so
                    za, filled = self._fill_hidden(za, valid, torch.tensor(ch, device=DEV), torch.tensor(sky, device=DEV))   # the ground under it is never seen)
                    valid = valid | filled
                new = valid & (tgt == 0)                                                           # only what the cache lacks
                lab_ = torch.tensor(cv2.connectedComponentsWithStats(new.cpu().numpy().astype(np.uint8))[1], device=DEV)
                cnt = torch.bincount(lab_.flatten()); new = new & (cnt[lab_] >= 200) & (lab_ > 0)  # no specks
                nfs = 0
                if int(new.sum()) > 0 and os.environ.get("PC_CARVE", "1") == "1":
                    ok = self._free_space(f, za, new); nfs = int((new & ~ok).sum()); new = new & ok
                if int(new.sum()) > 0: self._add(f, za, new)
                self.obs.append((f, torch.where(kt & rel, za, torch.zeros_like(za)), torch.tensor(sky, device=DEV)))
                self._sky_dirs(f, sky); self.log.append((f, how)); stats.append(f"{f}:{how} +{int(new.sum())} drop_edge{int((kt & ~rel).sum())} carved{nfs}")
            self.log.append(("world", " ".join(stats) + f" | cache {len(self.P)} pts"))
        self.meshes = [1]                                                                       # (_paint_world / sync read "has a world")
        T2 = _t.time(); n_ = self._paint_world(t_from); T3 = _t.time()
        self.log.append(("time", f"recon {T1 - T0:.1f}s keyframes {len(todo)} {T2 - T1:.1f}s paint {len(self.cells)}f {T3 - T2:.1f}s {self.log[-1][1]} {[l[1] for l in self.log if l[0] == 'frozen_time'][-1:]}")); return n_

    def _fill_hidden(self, z, valid, ch, sky, iters=300):
        """inverse depth diffused into the character pixels (and small holes) from the valid surroundings: a plane is linear in inverse depth,
        so the ground runs on under the player; a hole touching the sky or the image top is left empty"""
        hole = (ch | ((~valid) & ~sky)) & ~valid & ~sky
        if not hole.any(): return z, hole
        n_, lab_ = cv2.connectedComponents(hole.cpu().numpy().astype(np.uint8)); lab_ = torch.tensor(lab_, device=DEV)
        top = torch.unique(lab_[0]); sz = torch.bincount(lab_.flatten())
        ok = (lab_ > 0) & ~torch.isin(lab_, top) & (sz[lab_] < 0.05 * z.numel())
        skyn = F.max_pool2d(sky.float()[None, None], 9, 1, 4)[0, 0] > 0                        # holes reaching the sky are not ground
        bad = torch.unique(lab_[skyn & hole]); ok &= ~torch.isin(lab_, bad[bad > 0])
        inv = torch.where(valid, 1.0 / z.clamp(min=1e-3), torch.zeros_like(z)); w = valid.float()
        k = torch.ones(1, 1, 3, 3, device=DEV)
        for _ in range(iters):
            s_ = F.conv2d((inv * w)[None, None], k, padding=1)[0, 0]; n = F.conv2d(w[None, None], k, padding=1)[0, 0]
            upd = ok & (n > 0); inv = torch.where(upd, s_ / n.clamp(min=1e-6), inv); w = torch.where(upd, torch.ones_like(w), w)
        filled = ok & (inv > 0)
        return torch.where(filled, 1.0 / inv.clamp(min=1e-5), z), filled

    def _free_space(self, f, z, m):
        """free-space carving (as in TSDF fusion): a new point must not sit in space an earlier view saw THROUGH -- nearer than 0.85x the
        surface that view observed along its ray, or in that view's sky. Returns a mask over m."""
        c, R = self._cam(f); zz = z[m]
        P = c + (self.rx[m] * zz)[:, None] * R[0] - (self.ry[m] * zz)[:, None] * R[1] + zz[:, None] * R[2]
        bad = torch.zeros(len(P), dtype=torch.bool, device=DEV)
        for t, D, SK in self.obs:
            ct, Rt = self._cam(t); d = P - ct; zc = d @ Rt[2]; ok = zc > 0.05
            u = (self.cx + self.f * (d @ Rt[0]) / zc.clamp(min=1e-3)).long(); v = (self.cy - self.f * (d @ Rt[1]) / zc.clamp(min=1e-3)).long()
            ok &= (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H); ui, vi = u.clamp(0, self.W - 1), v.clamp(0, self.H - 1)
            Do = D[vi, ui]; bad |= ok & (((Do > 0) & (zc < 0.85 * Do)) | (SK[vi, ui] & (os.environ.get("PC_CARVE_SKY", "1") == "1")))
        out = m.clone(); out[m] = ~bad; return out

    def _sky_dirs(self, t, sky):
        p, yw = MV.dir_grid(self.R[t].numpy().astype(float), self.H, self.W, self.f, self.cx, self.cy); sk = sky & (p >= 180)
        g = np.zeros((360, 720), np.uint8); g[p[sk], yw[sk]] = 1; self.sky |= cv2.dilate(g, np.ones((3, 3), np.uint8)).astype(bool)
