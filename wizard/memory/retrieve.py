"""memory retrieval of the memory model v2 -- ONE function for the training index (index_bank.py over the recorded
candidate bank) and for the live engine (over the frames generated so far), so the model is trained on exactly the memory it will be given.
  A CHUNK = 4 latent frames (16 pixel frames), the live engine's unit. For each chunk:
    points   the world points the chunk will show (its G-buffer: recorded depth in training, the growing world in the engine), sub-sampled
    seen     a candidate frame sees a point if it projects inside the candidate's grid and the candidate's own depth there agrees (planar, 5% + 0.3 m)
    greedy   set cover: take the candidate that sees the most still-uncovered points, until k frames or the gain is < min_gain of the points.
  Candidates: one frame per second, at least GAP seconds before the chunk (what is newer is in the model's own context / KV cache).
  retrieve_chunks(points_per_chunk, bank) -> per chunk [(candidate index, gain)], plus the seen-fraction of every candidate (for distractors)."""
import torch

GAP, K_MAX, MIN_GAIN, TOL_REL, TOL_ABS = 3.0, 3, 0.05, 0.05, 0.3


def seen(P, depth, pos, R, K):
    """P [N,3] world points; candidates depth [C,H,W] (0 = none), pos [C,3], R [C,3,3] (rows right, up, forward), K (fx, fy, cx, cy) of the grid, [4] or [C,4]
    -> [C,N] bool"""
    C, H, W = depth.shape; K = torch.as_tensor(K, dtype=torch.float32, device=P.device).reshape(-1, 4).expand(C, 4)
    fx, fy, cx, cy = [K[:, i:i + 1] for i in range(4)]                            # per candidate [C,1]
    d = torch.einsum("cnk,cjk->cnj", P[None] - pos[:, None], R)                 # camera coordinates [C,N,3]
    z = d[..., 2]; zc_ = z.clamp(min=1e-3); u = cx + fx * d[..., 0] / zc_; v = cy - fy * d[..., 1] / zc_
    ui, vi = u.floor().long(), v.floor().long(); inb = (z > 0.1) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    zc = depth.float().reshape(C, -1).gather(1, vi.clamp(0, H - 1) * W + ui.clamp(0, W - 1))
    return inb & (zc > 0) & ((zc - z).abs() < TOL_REL * z + TOL_ABS)


def greedy(S, k=K_MAX, min_gain=MIN_GAIN, allowed=None):
    """S [C,N] bool (candidate sees point) -> [(index, gain as a fraction of the N points)] in order of choice"""
    N = S.shape[1]
    if N == 0 or S.shape[0] == 0: return []
    cov = torch.zeros(N, dtype=torch.bool, device=S.device); out = []
    ok = torch.ones(S.shape[0], dtype=torch.bool, device=S.device) if allowed is None else allowed.clone()
    for _ in range(k):
        g = (S & ~cov).sum(1).float() / N; g[~ok] = -1; j = int(g.argmax())
        if float(g[j]) < min_gain: break
        out.append((j, float(g[j]))); cov |= S[j]; ok[j] = False
    return out


def retrieve_chunks(points, bank, t_chunks, k=K_MAX, min_gain=MIN_GAIN, gap=GAP, context=None):
    """points: list over chunks of [N_c,3] world points; bank: dict depth [C,H,W], pos [C,3], R [C,3,3], K, t [C] (seconds)
    t_chunks: the time (s) each chunk starts; context: frames the model already has in front of it (same keys as bank; training = the window's clean
    first frame, live = the frames in the KV cache) -- points they see need no memory and do not count.
    -> (picks: list over chunks of [(candidate, gain)], frac: list over chunks of [C] seen fractions)"""
    picks, fracs = [], []
    for P, tc in zip(points, t_chunks):
        if context is not None and len(context["depth"]) and P.shape[0]: P = P[~seen(P, context["depth"], context["pos"], context["R"], context["K"]).any(0)]
        allowed = bank["t"] <= tc - gap
        if P.shape[0] == 0 or not bool(allowed.any()): picks.append([]); fracs.append(torch.zeros(len(bank["t"]))); continue
        S = seen(P, bank["depth"], bank["pos"], bank["R"], bank["K"]) & allowed[:, None]
        picks.append(greedy(S, k, min_gain, allowed)); fracs.append(S.float().mean(1).cpu())
    return picks, fracs


# the chunks of a 121-frame window: latent 0 = pixel frame 0, latent j >= 1 = pixel frames 4j-3 .. 4j; chunk c = latents 1+4c .. 4+4c (latent 0 joins chunk 0)
def chunk_latents(n_lat=31, per=4):
    out = []; j = 1
    while j < n_lat: out.append(list(range(j, min(j + per, n_lat)))); j += per
    out[0] = [0] + out[0]; return out


def chunk_pixels(lats):
    return sorted(set(f for j in lats for f in ([0] if j == 0 else range(4 * j - 3, 4 * j + 1))))


def proj_cover(P, pos, R, K, H, W):
    """which grid cells of cameras (pos [T,3], R [T,3,3], K fx,fy,cx,cy) the world points P [N,3] land on: each point a 2x2-cell
    footprint, holes closed 3x3 -> bool [T,H,W]. A projection test, not the per-cell depth-consistency of seen(), which is speckled on
    estimated depth (SceneFly): a place the first frame does see came out 'unseen' in small patches."""
    import torch.nn.functional as F_
    fx, fy, cx, cy = [float(v) for v in torch.as_tensor(K).reshape(-1)[:4]]
    T = pos.shape[0]; cov = torch.zeros(T, H, W, dtype=torch.bool, device=P.device)
    for t in range(T):
        c = (P - pos[t]) @ R[t].T; z = c[:, 2]; ok = z > 0.1
        u = (cx + fx * c[:, 0] / z.clamp(min=1e-3))[ok]; v = (cy - fy * c[:, 1] / z.clamp(min=1e-3))[ok]
        for du in (0, -1):
            for dv in (0, -1):
                ui = (u + du).floor().long(); vi = (v + dv).floor().long(); m = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
                cov[t].view(-1)[vi[m] * W + ui[m]] = True
    c = cov[:, None].float(); c = -F_.max_pool2d(-F_.max_pool2d(c, 3, 1, 1), 3, 1, 1)
    return c[:, 0] > 0.5
