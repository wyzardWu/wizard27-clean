"""Causal (autoregressive) sampling on a prepared case, window by window: latent frame by latent frame through the
KV-cached stream (history window + sink, per-chunk memory, gated references). Few-step students use --cm_steps (e.g. 1000,500) and
--cfg 1; the causal teacher uses --steps / --cfg.

    python -m wizard.inference.causal --dit <dit> --case <case dir> --frames 241 --cm_steps 1000,500 --cfg 1 --out out.mp4"""
import os, time, sys, argparse, subprocess
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
import safetensors.torch as st
from wizard.models.dit import WizardDiT, ti2v_5b_config
from wizard.models.flow_match import FlowMatchScheduler
from wizard.models.cond import attach_cond, cat_cond
from wizard.vendor.wan.vae2_2 import Wan2_2_VAE
from wizard.data.video import W, H, GREY, read
import wizard.data.gbuffer as G


def read7(path, frames):
    """the proxy exactly as the data pipeline fed the VAE: AREA to 853x480, centre crop x 10..842"""
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", "scale=853:480:flags=area,crop=832:480:10:0", "-frames:v", str(frames), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    v = np.frombuffer(raw, np.uint8).reshape(-1, 480, 832, 3); return v if len(v) >= frames else None
from wizard.models.world_cond import attach_world, scene_gbuf, M_MAX, CH
from wizard.memory.retrieve import retrieve_chunks, chunk_pixels, MIN_GAIN, GAP
import wizard.train.bidirectional as TV
from wizard.paths import WAN_CKPT
CK = WAN_CKPT; N = 121; L = 31


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dit", required=True); ap.add_argument("--case", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=241); ap.add_argument("--ref1", default=None); ap.add_argument("--ref2", default=None); ap.add_argument("--ref-slots", default="0,0")
    ap.add_argument("--prompt-file", default=None); ap.add_argument("--steps", type=int, default=30); ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--cm_steps", default=None, help="CD student: consistency sampling at these levels, e.g. 1000,500 (KV path only; use --cfg 1)")
    ap.add_argument("--shift", type=float, default=5.0); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--nomem", action="store_true")
    ap.add_argument("--nofill", action="store_true", help="(old) the window's depth as given; default: the cells behind the characters filled + smoothed as training")
    ap.add_argument("--rest", action="store_true", help="identity coordinates: the coordinate channels and the retrieval use where each point was before any edit (gbuf rest)")
    ap.add_argument("--skeleton_depth_cond", action="store_true", help="the condition video is the whole cond.mp4 (skeleton + depth)")
    ap.add_argument("--hide_all", action="store_true", help="memory frames: hide all characters' tokens (player too), not only referenced slots")
    ap.add_argument("--empty_actor", action="store_true", help="full G-buffer, no geometry on character cells")
    ap.add_argument("--maintain", action="store_true", help="after each window: MoGe of the frames it made -> the still-unknown cells of later frames (wizard.world.maintainer)")
    ap.add_argument("--coords_only", action="store_true", help="G-buffer = world coordinate + has bit on background cells only (character cells: the fill rule's region)")
    ap.add_argument("--slow", action="store_true", help="the full teacher-forcing forward per step (reference for the KV path)"); ap.add_argument("--window", type=int, default=20); ap.add_argument("--sink", type=int, default=1)
    a = ap.parse_args(); dev = "cuda"; C = a.case; NF = a.frames; assert (NF - 1) % 120 == 0
    from wizard.vendor.wan.t5 import T5EncoderModel
    t5 = T5EncoderModel(text_len=512, dtype=torch.bfloat16, device=dev, checkpoint_path=f"{CK}/models_t5_umt5-xxl-enc-bf16.pth", tokenizer_path=f"{CK}/google/umt5-xxl")
    def emb(p):
        with torch.no_grad(): e = t5([p], dev)[0][:512]
        return torch.cat([e, e.new_zeros(512 - e.shape[0], e.shape[1])])[None].to(torch.bfloat16)
    ctx = emb(open(a.prompt_file or f"{C}/prompt.txt").read().strip()); neg = emb(""); del t5; torch.cuda.empty_cache()
    vae = Wan2_2_VAE(vae_pth=f"{CK}/Wan2.2_VAE.pth", device=dev); T = lambda v: torch.from_numpy(np.ascontiguousarray(v)).permute(3, 0, 1, 2).float().div(127.5).sub(1.0).to(dev)
    cond = read(f"{C}/cond.mp4", NF); assert cond is not None                                       # the character masks
    v7c = read7(f"{C}/cond_v7.mkv", NF); assert v7c is not None, "the case has no proxy video (cond_v7.mkv)"      # the condition the model sees
    t7 = torch.from_numpy(np.load(f"{C}/tier7.npy")[:NF].astype(np.float32))                          # token tier per cell
    act_v = np.where((cond[..., 1] >= 20)[..., None], cond, 0).astype(np.uint8)                       # the actor layer (encode_actor.py, G_MIN 20)
    fi = Image.open(f"{C}/first_frame.png").convert("RGB"); ff = np.asarray(fi.resize((round(H * 16 / 9), H), Image.LANCZOS)); o = (ff.shape[1] - W) // 2; ff = ff[:, o:o + W]
    img = lambda p: T(np.asarray((Image.open(p) if p else Image.new("RGB", (480, 480), GREY)).convert("RGB").resize((480, 480), Image.LANCZOS))[None])
    with torch.no_grad(): zr = torch.stack([vae.encode([img(p)])[0] for p in (a.ref1, a.ref2)])[None].to(torch.bfloat16)
    slots = torch.tensor([[int(s) for s in a.ref_slots.split(",")]], device=dev); live = [int(s) for s in a.ref_slots.split(",") if int(s) > 0]
    pb = np.load(f"{C}/slots.npz")["pixel_bits"][:NF].repeat(4, 1).repeat(4, 2)
    ys = (np.arange(H) * 576 / H).astype(int); xs = ((np.arange(W) + (853 - W) / 2) * 1024 / 853).astype(int); pb = pb[:, ys][:, :, xs]
    tok_all = np.stack([(pb >> k) & 1 > 0 for k in range(3)], -1).reshape(NF, 15, H // 15, 26, W // 26, 3).mean((2, 4)).astype(np.float32)   # [NF,15,26,3]
    g = np.load(f"{C}/gbuf.npz"); dep_all = torch.from_numpy(np.where(np.isfinite(g["depth"][:NF]) & (g["depth"][:NF] < 1000), g["depth"][:NF], 0).astype(np.float32)); pos_all = torch.from_numpy(g["pos"][:NF]); R_all = torch.from_numpy(g["R"][:NF])
    K = torch.from_numpy(g["K"]); origin = torch.from_numpy(g["origin"]); hp = float(g["hp"])
    actc = torch.from_numpy(cond[..., 1] >= 20).float()[:, None]; actc = F.max_pool2d(actc, 8)[:, 0] > 0                                  # [NF,60,104] character cells
    actc = F.max_pool2d(actc.float()[:, None], 5, 1, 2)[:, 0] > 0                                                                         # grown 2 cells (as training)
    if not a.nofill:                                    # training's window depth is env-filled + harmonic-smoothed behind every character (gbuf2 + smooth_env):
        from wizard.data.env_fill import fill_env, smooth_env      # do the same here, actor cells as gbuf_synth (skeleton cells > 5%, grown 3 cells), else crisp background behind the boss
        import cv2
        a7 = ((cond[..., 1] >= 20).reshape(NF, 60, 8, 104, 8).mean((2, 4)) > 0.05)
        a7 = np.stack([cv2.dilate(x.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0 for x in a7]); z = g["depth"][:NF].astype(np.float32)
        from scipy.ndimage import binary_fill_holes     # a big body's skeleton+3 leaves holes inside it (belly, between limbs) where the crisp background showed
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))     # through -> hollow bodies; WW (38%) trains on the solid silhouette: close + fill the holes
        a7 = np.stack([binary_fill_holes(cv2.morphologyEx(np.pad(x, 6).astype(np.uint8), cv2.MORPH_CLOSE, kc)[6:-6, 6:-6] > 0) for x in a7])
        if os.environ.get("ACTOR_HULL"):                # each character's region = the convex hull of its skeleton cells, grown ACTOR_HULL cells
            gm = int(os.environ["ACTOR_HULL"]); Gc = cond[..., 1].reshape(NF, 60, 8, 104, 8); hl = np.zeros_like(a7)
            for lo, hi in ((20, 148), (148, 256)):
                m_ = ((Gc >= lo) & (Gc < hi)).mean((2, 4)) > 0.05
                for t in range(NF):
                    ys, xs = np.nonzero(m_[t])
                    if len(ys) < 3: continue
                    h_ = np.zeros((60, 104), np.uint8); cv2.fillConvexPoly(h_, cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32)), 1)
                    hl[t] |= cv2.dilate(h_, np.ones((2 * gm + 1, 2 * gm + 1), np.uint8)) > 0
            a7 = a7 | hl
        dep_win = torch.from_numpy(smooth_env(np.stack([fill_env(z[t], a7[t]) for t in range(NF)]), a7))
    else: dep_win = dep_all
    dep_g = dep_all * torch.from_numpy(~a7) if True else dep_all * torch.from_numpy(~a7)                       # no geometry on character cells
    import wizard.train.causal as TAR, wizard.models.ref as REF_
    with torch.device("meta"): mdl = WizardDiT(**ti2v_5b_config(), causal=True, zero_init_head=False, train_local_attn_size=a.window, train_sink_size=a.sink)
    attach_cond(mdl); attach_world(mdl); REF_.forward_ref_tf = TAR.forward_tf; info = mdl.load_state_dict(st.load_file(a.dit), strict=False, assign=True)
    assert not [k for k in info.missing_keys if "rope" not in k], info.missing_keys[:5]
    mdl.rope = type(mdl.rope)(mdl.dim // mdl.num_heads, max_pos=1024); mdl = mdl.to(dev, torch.bfloat16).eval()
    made = [ff]; MT = None; v7c = np.array(v7c)                                          # (writable: --maintain edits later frames)                                                                                         # pixel frames made so far (frame 0 = the first frame)
    log = open(a.out[:-4] + ".log", "w")
    WORLD = a.maintain and os.environ.get("MAINT_WORLD") and os.path.exists(f"{C}/world_px.npz")
    if WORLD:                                                                            # the first frame's world is the maintainer's from the start
        from wizard.world.world_maintainer import WorldMaintainer
        os.environ["ONE_RECON"] = "1"; MT = WorldMaintainer(C, v7c, cond, t7, dep_all, pos_all, R_all, K)
        nch = MT.update_world(made, 0, 0, 0) if (os.environ.get("COARSE_WORLD") != "1" or os.environ.get("SELF_FIRST") == "1") else 0   # coarse world: the case's own first-frame world is kept as it is
        for t_, (cn, cs) in MT.cells.items():
            dep_win[t_][cn] = dep_all[t_][cn]; dep_win[t_][cs] = 0.0; dep_g[t_] = dep_all[t_] * torch.from_numpy(~a7[t_])
        print(f"world from the first frame: {nch} pixels | {MT.log[-1:]}", flush=True)
    for s in range(0, NF - 1, 120):
        sl = slice(s, s + N); first = made[s]
        with torch.no_grad(): z0 = vae.encode([T(first[None])])[0][None].float(); zc = vae.encode([T(v7c[sl])])[0][None].to(torch.bfloat16)
        cover = np.zeros((L, 15, 26, 3), np.float16); tok = tok_all[sl]; cover[0] = tok[0]
        for j in range(1, L): cover[j] = tok[4 * j - 3: 4 * j + 1].max(0)
        dep = dep_win[sl]; pos = pos_all[sl]; R = R_all[sl]
        # ---- memory: the frames made so far, one per second, >= GAP s before the chunk (the engine's bank), per chunk <= 3, pool <= 6
        cand = [f for f in range(0, s + 1, 30)] if not a.nomem else []
        per = [[] for _ in CH]
        if cand:
            P = G.world_points(dep, pos, R, K); ok = dep > 0; pts, tcs = [], []
            for lats in CH:
                px = chunk_pixels(lats)[::4]; pts.append(P[px][:, ::4, ::4][ok[px][:, ::4, ::4]]); tcs.append((s + chunk_pixels(lats)[0]) / 30.0)
            bank = dict(depth=torch.stack([dep_all[f] * (~actc[f]) for f in cand]), pos=pos_all[cand], R=R_all[cand], K=K.expand(len(cand), 4), t=torch.tensor([f / 30.0 for f in cand], dtype=torch.float64))
            if a.rest: per = rest_retrieve(g["rest"], np.where(actc.numpy(), 0, g["depth"][:NF]), s, cand)
            else:
                picks, _ = retrieve_chunks(pts, bank, tcs, k=3, min_gain=MIN_GAIN, gap=float(os.environ.get("MEM_GAP", GAP)), context=dict(depth=dep[:1], pos=pos[:1], R=R[:1], K=K))
                per = [[cand[j] for j, _ in pk] for pk in picks]
        pool, use = TV.ClipDataset.pick(None, per); n = len(pool)
        mem_lat = torch.zeros(1, 48, M_MAX, 30, 52, device=dev); gm = torch.zeros(M_MAX, 8, 60, 104, device=dev); hide = torch.zeros(1, M_MAX, 15 * 26, dtype=torch.bool)
        mdep_w = torch.zeros(M_MAX, 60, 104, device=dev)
        for j, f in enumerate(pool):
            with torch.no_grad(): mem_lat[0, :, j] = vae.encode([T(made[f][None])])[0][:, 0].float()
            dmem = dep_all[f]
            if WORLD and os.environ.get("MEM_FROM_WORLD", "1") == "1" and getattr(MT, "wmeshes", None):   # a memory frame's geometry = the world as maintained NOW, seen from that frame
                zf_ = MT._frozen([f])[0]; zc_ = np.where(zf_ > 0, zf_, np.nan).reshape(60, 8, 104, 8).transpose(0, 2, 1, 3).reshape(60, 104, 64)
                with np.errstate(all="ignore"): zm_ = np.nan_to_num(np.nanmedian(zc_, -1))
                dmem = torch.where(torch.from_numpy(zm_ > 0), torch.from_numpy(zm_).float(), dep_all[f])
            gm[j] = scene_gbuf((dmem * (~actc[f]))[None].to(dev), pos_all[f:f + 1].to(dev), R_all[f:f + 1].to(dev), K, origin.to(dev), hp)[:, 0]
            mdep_w[j] = (dmem * (~actc[f])).to(dev)
            if a.rest: gm[j] = rest_xyz(gm[j][None], g["rest"][f:f + 1], (dep_all[f] * (~actc[f]))[None].numpy(), origin, hp, dev)[0]
            hm = torch.zeros(15, 26, dtype=torch.bool)
            hm |= torch.from_numpy(tok_all[f].max(-1) > 0.02) | (F.max_pool2d(actc[f].float()[None, None], 4)[0, 0] > 0)   # every character, the player too
            gm[j, 7] = t7[f].to(dev) / 3.0
            if os.environ.get("MEM_TIER_TRAIN", "1") == "1": gm[j, 7] = ((dmem * (~actc[f])) > 0).float().to(dev)   # training's memory frames: tier = 3 wherever they have depth (roam bank), whatever the window's tier
            hide[0, j] = (F.max_pool2d(hm.float()[None, None], 3, 1, 1)[0, 0] > 0).reshape(-1)
        print(f"window {s}: memory pool {pool} | per chunk {per}", flush=True); log.write(f"window {s}: pool {pool} per chunk {per}\n")
        gc = scene_gbuf(dep_g[sl].to(dev), pos.to(dev), R.to(dev), K, origin.to(dev), hp)
        if a.rest: gc = rest_xyz(gc.permute(1, 0, 2, 3), g["rest"][sl], dep.numpy(), origin, hp, dev).permute(1, 0, 2, 3)
        ref = dict(lat=zr, slots=slots, cover=torch.from_numpy(cover)[None].to(dev), mem_use=use[None], mem_n=torch.tensor([n]), mem_hide=hide)
        if os.environ.get("ATTN_DIAG") and s == 0: import wizard.models.attn_diag as AD; AD.install()
        sched = FlowMatchScheduler(shift=a.shift); tsteps = sched.set_timesteps(a.steps, device=dev)
        gen = torch.Generator(device=dev).manual_seed(a.seed + s); x = torch.randn((1, 48, L, 30, 52), device=dev, generator=gen); x[:, :, 0] = z0[:, :, 0]
        gc = torch.cat([gc[:3], (gc[3:4] > 0).to(gc.dtype), torch.zeros_like(gc[4:7]), (t7[sl].to(dev) / 3.0)[None].to(gc.dtype)], 0)   # coordinate + has + tier
        gm = torch.cat([gm[:, :3], (gm[:, 3:4] > 0).to(gm.dtype), torch.zeros_like(gm[:, 4:7]), gm[:, 7:8]], 1)
        if os.environ.get("INF_UNK_NOCOORD") == "1": gc[:4] = torch.where((t7[sl].to(dev) == 1)[None], torch.zeros_like(gc[:4]), gc[:4])   # as training UNK_NOCOORD=1
        mdl._gb_in = (gc[None], gm[None])
        mdl._mem_warp = None
        mdl._mem_wrope = None
        if os.environ.get("MEM_WROPE") == "1" and n:                                             # world RoPE for reading memory
            from wizard.models.world_cond import world_rope_layout
            mdl._mem_wrope = world_rope_layout(mdep_w, pos_all[pool].to(dev), R_all[pool].to(dev), K.to(dev), n, hide[0], use, dep_g[sl].to(dev), pos.to(dev), R.to(dev), K)
            if mdl._mem_wrope is not None:
                print(f"window {s}: world-RoPE memory keys {mdl._mem_wrope['ksrc'].numel()}", flush=True)
                if os.environ.get("MEM_WARP_DUMP"): torch.save({k_: v_.cpu() for k_, v_ in mdl._mem_wrope.items()} | dict(pool=pool, use=use.cpu()), os.environ["MEM_WARP_DUMP"] + f"_wrope_{s}.pt")
        if os.environ.get("MEM_WARP") == "1" and n:                                              # memory RoPE warping (wizard.models.world_cond.mem_warp_pos)
            from wizard.models.world_cond import mem_warp_pos
            mdl._mem_warp = mem_warp_pos(mdep_w, pos_all[pool].to(dev), R_all[pool].to(dev), K.to(dev), pos.to(dev), R.to(dev), K, use, n)
            if os.environ.get("MEM_WARP_DUMP"): torch.save(dict(pos=mdl._mem_warp.cpu(), pool=pool, use=use.cpu(), s=s), os.environ["MEM_WARP_DUMP"] + f"_{s}.pt")
            print(f"window {s}: warped memory tokens {int(torch.isfinite(mdl._mem_warp).all(-1).sum())}/{n * 390}", flush=True)
        cnd = torch.cat([torch.zeros_like(mem_lat), zc.float()], 2).to(torch.bfloat16)
        clean = torch.zeros_like(x); clean[:, :, 0] = z0[:, :, 0]; t_w = time.time()            # (AR) the frames made so far in this window
        if not a.slow:                                                                      # KV-cache streaming (wizard.inference.stream), == the TF forward
            import wizard.inference.stream as AS
            gc_, gm_ = mdl._gb_in; zcf = zc.float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                sts = [AS.Stream(mdl, c_, gc_, gm_, mem_lat, ref, W=a.window, sink=a.sink) for c_ in ((ctx, neg) if a.cfg != 1.0 else (ctx,))]
                HE = os.environ.get("HIDE_EDIT", "0")
                if HE in ("1", "2"):                                                      # hide history / memory tokens whose place the geometry has changed
                    a7t = torch.from_numpy(np.asarray(a7)); fx_, fy_, cx_, cy_ = [float(v_) for v_ in K]; pxof = lambda f_: min(s + 4 * f_, NF - 1)
                    def changed(pq, ph):
                        zh = dep_all[ph]; okh = (zh > 0) & ~a7t[ph]
                        P_ = G.world_points(zh[None], pos_all[ph:ph + 1], R_all[ph:ph + 1], K)[0]; rel = P_ - pos_all[pq]; Rq = R_all[pq]
                        xr, yr, zr = rel @ Rq[0], rel @ Rq[1], rel @ Rq[2]; zs = torch.where(zr > 0.1, zr, torch.ones_like(zr))
                        u_ = cx_ + fx_ * xr / zs; v_ = cy_ - fy_ * yr / zs
                        inv = okh & (zr > 0.1) & (u_ >= 0) & (u_ < 104) & (v_ >= 0) & (v_ < 60)
                        ui = u_.clamp(0, 103).long(); vi = v_.clamp(0, 59).long(); zq = dep_all[pq][vi, ui]; chq = a7t[pq][vi, ui]
                        ch = inv & ~chq & (((zq > 0) & ((zq < 0.8 * zr) | (zq > 1.25 * zr))) | (zq <= 0))
                        return (ch.float().reshape(15, 4, 26, 4).mean((1, 3)) >= 0.25).reshape(-1)
                    HHC = {}
                    def hh_(fq, fh):
                        k_ = (pxof(fq), pxof(fh))
                        if k_ not in HHC: HHC[k_] = changed(*k_)
                        return HHC[k_]
                    for st_ in sts: st_.kv["hist_hide"] = hh_
                    nh = sum(int(hh_(f_, 0).sum()) for f_ in range(1, L)); print(f"window {s}: HIDE_EDIT={HE} history tokens hidden vs the sink, summed over frames: {nh}", flush=True)
                    if HE == "2":
                        nm = 0
                        for f_ in range(L):
                            for j_, pf_ in enumerate(pool):
                                cm_ = changed(pxof(f_), pf_).to(dev); nm += int(cm_.sum())
                                for st_ in sts: st_.kv["mem_allow"][f_][0, j_ * 390:(j_ + 1) * 390] &= ~cm_
                        print(f"window {s}: HIDE_EDIT memory tokens hidden, summed over frames: {nm}", flush=True)
                for st_ in sts: st_.frame(clean[:, :, 0:1], zcf[:, :, 0:1], 0, 0.0, commit=True)
                for j in range(1, L):
                    xj = x[:, :, j:j + 1].clone()
                    lv = [float(v_) for v_ in a.cm_steps.split(",")] if a.cm_steps else []
                    for i, t in enumerate(lv):                                                # consistency sampling of the CD student, as infer_ar_ref.py
                        vc = sts[0].frame(xj, zcf[:, :, j:j + 1], j, t).float()
                        vu = sts[1].frame(xj, zcf[:, :, j:j + 1], j, t).float() if a.cfg != 1.0 else vc
                        xj = xj - t / 1000.0 * (vu + a.cfg * (vc - vu))
                        if i + 1 < len(lv): ns = lv[i + 1] / 1000.0; xj = (1 - ns) * xj + ns * torch.randn(xj.shape, device=dev, generator=gen)
                    for i, t in enumerate(tsteps if not lv else ()):
                        vc = sts[0].frame(xj, zcf[:, :, j:j + 1], j, float(t)).float()
                        vu = sts[1].frame(xj, zcf[:, :, j:j + 1], j, float(t)).float() if a.cfg != 1.0 else vc
                        xj = sched.step(vu + a.cfg * (vc - vu), xj, i)
                    clean[:, :, j:j + 1] = xj
                    for st_ in sts: st_.frame(xj, zcf[:, :, j:j + 1], j, 0.0, commit=True)
                    if j % 10 == 0: print(f"window {s}: latent frame {j}/{L - 1} ({time.time() - t_w:.0f} s, KV)", flush=True)
            del sts; torch.cuda.empty_cache()
        for j in (range(1, L) if a.slow else []):
            xj = x[:, :, j:j + 1].clone()
            for i, t in enumerate(tsteps):
                noisy = clean.clone(); noisy[:, :, j:j + 1] = xj
                ts = torch.zeros(1, L, device=dev); ts[:, j] = float(t)
                xn = cat_cond(torch.cat([mem_lat, noisy], 2).to(torch.bfloat16), cnd); xcl = cat_cond(torch.cat([mem_lat, clean], 2).to(torch.bfloat16), cnd)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    vc = mdl(xn, ts, ctx, clean_x=xcl, ref=ref).float()[:, :, j:j + 1]
                    vu = mdl(xn, ts, neg, clean_x=xcl, ref=ref).float()[:, :, j:j + 1] if a.cfg != 1.0 else vc
                xj = sched.step(vu + a.cfg * (vc - vu), xj, i)
            clean[:, :, j:j + 1] = xj
            if j % 5 == 0: print(f"window {s}: latent frame {j}/{L - 1} ({time.time() - t_w:.0f} s)", flush=True)
        x = clean
        mdl._gb_in = None
        with torch.no_grad(): out = vae.decode([x[0]])[0]
        fr = out.permute(1, 2, 3, 0).add(1).mul(127.5).clamp(0, 255).byte().cpu().numpy()
        made += list(fr[1:])
        if a.maintain and s + 120 < NF - 1:                                              # world maintenance from what this window made
            if MT is None:
                if os.path.exists(f"{C}/world_px.npz"):                                   # one world: case geometry + meshes grown from the generated frames
                    from wizard.world.world_maintainer import WorldMaintainer
                    MT = WorldMaintainer(C, v7c, cond, t7, dep_all, pos_all, R_all, K)
                else:
                    from wizard.world.maintainer import Maintainer
                    MT = Maintainer(v7c, cond, t7, dep_all, pos_all, R_all, K, actc)
            nch = MT.update_world(made, s, s + 120, s + 121) if WORLD else (MT.update_recon if os.environ.get("MAINT_RECON") and hasattr(MT, "update_recon") else MT.update)(made, s, s + 120, s + 121)
            for t_, (cn, cs) in MT.cells.items():
                dep_win[t_][cn] = dep_all[t_][cn]; dep_win[t_][cs] = 0.0; dep_g[t_] = dep_all[t_] * torch.from_numpy(~a7[t_])
            print(f"maintain after window {s}: {nch} pixels changed | {MT.log[-3:]}", flush=True); log.write(f"maintain {s}: {nch} | {MT.log}\n")
    if os.environ.get("ATTN_DIAG"): AD.dump(os.environ["ATTN_DIAG"])
    vid = np.stack(made[:NF]); cv = cond[:NF]
    side = np.concatenate([vid, cv], 2)
    for p, v in ((a.out, side), (a.out[:-4] + "_gen.mp4", vid)):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{v.shape[2]}x{v.shape[1]}", "-r", "30", "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", p], input=v.tobytes(), check=True)
    if a.maintain:                                                                       # the condition as maintained (what the later windows saw)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "1664x480", "-r", "30", "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", a.out[:-4] + "_cond7.mp4"], input=np.ascontiguousarray(np.concatenate([vid, v7c[:NF]], 2)).tobytes())
    print("INFER wrote", a.out, side.shape, flush=True)


def rest_xyz(g, rest, depth, org, hp, dev):
    """g [T,C,H,W]: the coordinate channels -> the REST (identity) coordinate where there is geometry (as turn_mem2.rest_xyz)"""
    kn = torch.from_numpy(np.asarray(depth) > 0).to(dev)
    w = (torch.tensor(np.asarray(rest), device=dev, dtype=torch.float32).permute(0, 3, 1, 2) - org.to(dev).view(1, 3, 1, 1)) / hp
    g = g.clone(); g[:, :3] = torch.where(kn[:, None], w, torch.zeros_like(w)); return g


def rest_retrieve(rest, depth, s0, cand, k=3, min_gain=MIN_GAIN, gap=int(GAP * 30), vox=0.5):
    """the greedy cover on IDENTITY (as turn_mem2.rest_retrieve, the engine's forward table): a chunk's points = the 0.5 m voxels of its rest coordinates
    minus those the window's first frame shows; a candidate covers the voxels its own rest coordinates fall in (characters' cells excluded)."""
    V = lambda f, st=1: set(map(tuple, np.floor(rest[f][::st, ::st][depth[f][::st, ::st] > 0] / vox).astype(np.int64)))
    seen0 = V(s0); cv = {f: V(f) for f in cand}; per = []
    for lats in CH:
        px = chunk_pixels(lats); t0 = s0 + px[0]; pts = set().union(*[V(s0 + q, 2) for q in px[::4]]) - seen0; ok = [f for f in cand if f <= t0 - gap]; out = []; cov = set()
        for _ in range(k):
            if not pts or not ok: break
            best = max(ok, key=lambda f: len((cv[f] & pts) - cov)); gn = len((cv[best] & pts) - cov) / len(pts)
            if gn < min_gain: break
            out.append(best); cov |= cv[best] & pts; ok = [f for f in ok if f != best]
        per.append(out)
    return per


if __name__ == "__main__":
    main()
