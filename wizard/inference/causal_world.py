"""Causal sampling as ONE continuous KV stream (positions roll past the trained range) while the world is maintained
from what has been generated: every few latent frames the new frames are reconstructed and the unknown parts of the later condition
frames are repainted from the maintained world (MAINT=pc: point cache, wizard.world.point_cache).

Speed options: ASYNC_MAINT=1 MAINT_GPU=<card> runs maintenance on a second card one segment behind generation; RECON_RESIDENT=1 keeps
the reconstruction model loaded; PAINT_AHEAD=<frames>, PAINT_THREADS=<n>, PC_BATCH=1 batch the repainting.

    MAINT=pc SELF_FIRST=1 MAINT_WORLD=1 COARSE_WORLD=1 python -m wizard.inference.causal_world --dit <dit> --case <case dir> \
        --frames 241 --cm_steps 1000,500 --cfg 1 --maintain --out out.mp4"""
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
    a = ap.parse_args(); dev = "cuda"; C = a.case; NF = a.frames; assert (NF - 1) % 4 == 0
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
    # ================= continuous stream: one KV stream over all latent frames, the world maintained every UPD latent frames =================
    import wizard.inference.stream_cont as AC
    LT = (NF - 1) // 4 + 1; UPD = int(os.environ.get("UPD", "8")); CTXN = int(os.environ.get("CTX_FRAMES", "3"))
    made = [ff]; v7c = np.array(v7c); MT = None; log = open(a.out[:-4] + ".log", "w")
    WORLD = a.maintain and os.environ.get("MAINT_WORLD") and os.path.exists(f"{C}/world_px.npz")
    def sync_cells():
        for t_, (cn, cs) in MT.cells.items():
            dep_win[t_][cn] = dep_all[t_][cn]; dep_win[t_][cs] = 0.0; dep_g[t_] = dep_all[t_] * torch.from_numpy(~a7[t_])
    unk = lambda lo, hi: float((t7[lo:hi] == 1).float().mean()) if hi > lo else 0.0
    # ASYNC_MAINT=1: the maintainer lives on a worker thread bound to MAINT_GPU (the 2nd visible card); generation never waits
    # for the update launched after segment k until segment k+1 is made, so segment k+2 sees the world from frames up to k (one segment late, like a live loop).
    ASY = os.environ.get("ASYNC_MAINT") == "1"; FUT = None
    if ASY:
        from concurrent.futures import ThreadPoolExecutor
        POOL = ThreadPoolExecutor(1, initializer=lambda: torch.cuda.set_device(int(os.environ.get("MAINT_GPU", "1"))))
    on_m = (lambda f, *x: POOL.submit(f, *x).result()) if ASY else (lambda f, *x: f(*x))
    if ASY:   # a second VAE on the maintenance card: the new condition video is encoded there too (3.4 s the generator no longer stops for)
        vae_m = on_m(lambda: Wan2_2_VAE(vae_pth=f"{CK}/Wan2.2_VAE.pth", device="cuda"))
        if os.environ.get("RECON_RESIDENT") == "1": import wizard.world.world_maintainer as MW_; on_m(MW_._mapany_server)   # load MapAnything before the stream starts
        def maint_job(made_, s0_, s1_, tf_):
            n_ = MT.update_world(made_, s0_, s1_, tf_)
            with torch.no_grad(): z_ = vae_m.encode([torch.from_numpy(np.ascontiguousarray(v7c[:NF])).permute(3, 0, 1, 2).float().div(127.5).sub(1.0).cuda()])[0][None].float()
            return n_, z_
    if WORLD:
        from wizard.world.world_maintainer import WorldMaintainer
        os.environ["ONE_RECON"] = "1"; MT = on_m(lambda: (__import__('wizard.world.point_cache', fromlist=['PCMaintainer']).PCMaintainer if os.environ.get('MAINT') == 'pc' else WorldMaintainer)(C, v7c, cond, t7, dep_all, pos_all, R_all, K))   # MAINT=pc: GEN3C point cache
        nch = on_m(MT.update_world, made, 0, 0, 0) if (os.environ.get("COARSE_WORLD") != "1" or os.environ.get("SELF_FIRST") == "1") else 0
        sync_cells(); print(f"world from the first frame: {nch} pixels | unknown later {unk(1, NF):.3f}", flush=True)
    def coords():
        gc = scene_gbuf(dep_g[:NF].to(dev), pos_all[:NF].to(dev), R_all[:NF].to(dev), K, origin.to(dev), hp)
        gc = torch.cat([gc[:3], (gc[3:4] > 0).to(gc.dtype), torch.zeros_like(gc[4:7]), (t7[:NF].to(dev) / 3.0)[None].to(gc.dtype)], 0)
        if os.environ.get("INF_UNK_NOCOORD") == "1": gc[:4] = torch.where((t7[:NF].to(dev) == 1)[None], torch.zeros_like(gc[:4]), gc[:4])
        return gc[None]
    def cond_lat():
        with torch.no_grad(): return vae.encode([T(v7c[:NF])])[0][None].float()
    cover = np.zeros((LT, 15, 26, 3), np.float16); cover[0] = tok_all[0]
    for j in range(1, LT): cover[j] = tok_all[4 * j - 3: 4 * j + 1].max(0)
    ref = dict(lat=zr, slots=slots, cover=torch.from_numpy(cover)[None].to(dev))
    with torch.no_grad(): z0 = vae.encode([T(ff[None])])[0][None].float()
    zcf = cond_lat(); gc = coords()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sts = [AC.ContStream(mdl, c_, gc, ref, W=a.window, sink=a.sink) for c_ in ((ctx, neg) if a.cfg != 1.0 else (ctx,))]
        for st_ in sts: st_.frame(z0[:, :, 0:1], zcf[:, :, 0:1], 0, 0.0, commit=True)
    sched = FlowMatchScheduler(shift=a.shift); tsteps = sched.set_timesteps(a.steps, device=dev)
    gen = torch.Generator(device=dev).manual_seed(a.seed); x = torch.randn((1, 48, LT, 30, 52), device=dev, generator=gen); x[:, :, 0] = z0[:, :, 0]
    clean = torch.zeros_like(x); clean[:, :, 0] = z0[:, :, 0]; t_w = time.time(); last_px = 0
    for p in range(1, LT, UPD):
        seg = list(range(p, min(p + UPD, LT)))
        # ---- memory for this segment: per 4-latent chunk, candidates = frames made so far (one per second), context = what the KV cache holds
        chunks = [seg[i:i + 4] for i in range(0, len(seg), 4)]; per = [[] for _ in chunks]
        cand = [f for f in range(0, len(made), 30)] if not a.nomem else []
        if cand:
            pts, tcs = [], []
            for lats in chunks:
                px = chunk_pixels(lats)[::4]; P = G.world_points(dep_win[px], pos_all[px], R_all[px], K); ok = dep_win[px] > 0
                pts.append(P[:, ::4, ::4][ok[:, ::4, ::4]]); tcs.append(chunk_pixels(lats)[0] / 30.0)
            bank = dict(depth=torch.stack([dep_all[f] * (~actc[f]) for f in cand]), pos=pos_all[cand], R=R_all[cand], K=K.expand(len(cand), 4), t=torch.tensor([f / 30.0 for f in cand], dtype=torch.float64))
            lastf = len(made) - 1; hist_px = sorted(set(([0] if os.environ.get("CTX_SINK") == "1" else []) + [max(0, lastf - 4 * i * (a.window // CTXN)) for i in range(CTXN)]))   # the sink is NOT context: frame 0 may come back as a memory frame
            picks, _ = retrieve_chunks(pts, bank, tcs, k=3, min_gain=MIN_GAIN, gap=float(os.environ.get("MEM_GAP", GAP)), context=dict(depth=dep_win[hist_px], pos=pos_all[hist_px], R=R_all[hist_px], K=K))
            per = [[cand[j] for j, _ in pk] for pk in picks]
        pool, use = TV.ClipDataset.pick(None, per + [[] for _ in range(8 - len(per))]); n = len(pool)
        mem_lat = torch.zeros(1, 48, M_MAX, 30, 52, device=dev); gm = torch.zeros(M_MAX, 8, 60, 104, device=dev); hide = torch.zeros(M_MAX, 15 * 26, dtype=torch.bool)
        for j, f in enumerate(pool):
            with torch.no_grad(): mem_lat[0, :, j] = vae.encode([T(made[f][None])])[0][:, 0].float()
            dmem = dep_all[f]
            if WORLD and os.environ.get("MEM_FROM_WORLD", "1") == "1" and getattr(MT, "wmeshes", None):
                zf_ = MT._frozen([f])[0]; zc_ = np.where(zf_ > 0, zf_, np.nan).reshape(60, 8, 104, 8).transpose(0, 2, 1, 3).reshape(60, 104, 64)
                with np.errstate(all="ignore"): zm_ = np.nan_to_num(np.nanmedian(zc_, -1))
                dmem = torch.where(torch.from_numpy(zm_ > 0), torch.from_numpy(zm_).float(), dep_all[f])
            gm[j] = scene_gbuf((dmem * (~actc[f]))[None].to(dev), pos_all[f:f + 1].to(dev), R_all[f:f + 1].to(dev), K, origin.to(dev), hp)[:, 0]
            gm[j, 7] = ((dmem * (~actc[f])) > 0).float().to(dev)
            hm = torch.from_numpy(tok_all[f].max(-1) > 0.02) | (F.max_pool2d(actc[f].float()[None, None], 4)[0, 0] > 0)
            hide[j] = (F.max_pool2d(hm.float()[None, None], 3, 1, 1)[0, 0] > 0).reshape(-1)
        gm = torch.cat([gm[:, :3], (gm[:, 3:4] > 0).to(gm.dtype), torch.zeros_like(gm[:, 4:7]), gm[:, 7:8]], 1)
        vis = (~hide).reshape(-1).to(dev); slot = torch.arange(M_MAX * 15 * 26, device=dev) // (15 * 26)
        allow = {f: (use[c].to(dev)[slot] & vis)[None] for c, lats in enumerate(chunks) for f in lats} if n else {}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for st_ in sts: st_.set_memory(mem_lat, gm[None], allow)
        print(f"latent {p}-{seg[-1]} (px {4 * p - 3}-{4 * seg[-1]}): memory pool {pool} | per chunk {per} | unknown in these frames {unk(4 * p - 3, 4 * seg[-1] + 1):.3f}", flush=True)
        log.write(f"latent {p}: pool {pool} per {per} unknown {unk(4 * p - 3, 4 * seg[-1] + 1):.3f}\n")
        # ---- the segment, latent frame by latent frame
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for j in seg:
                xj = x[:, :, j:j + 1].clone()
                lv = [float(v_) for v_ in a.cm_steps.split(",")] if a.cm_steps else []
                for i, t in enumerate(lv):                                                # CD student, as wizard.inference.causal
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
        with torch.no_grad(): out = vae.decode([clean[0, :, :seg[-1] + 1]])[0]
        made = list(out.permute(1, 2, 3, 0).add(1).mul(127.5).clamp(0, 255).byte().cpu().numpy()); made[0] = ff
        print(f"made {len(made)} frames ({time.time() - t_w:.0f} s)", flush=True)
        # ---- world maintenance from what was just made -> the later frames' condition, coordinates (and next memory)
        if a.maintain and seg[-1] + 1 < LT:
            s0, s1 = last_px, len(made) - 1
            if MT is None:
                from wizard.world.world_maintainer import WorldMaintainer
                MT = (__import__('wizard.world.point_cache', fromlist=['PCMaintainer']).PCMaintainer if os.environ.get('MAINT') == 'pc' else WorldMaintainer)(C, v7c, cond, t7, dep_all, pos_all, R_all, K)   # MAINT=pc: GEN3C point cache
            u0 = unk(s1 + 1, NF); t_m = time.time()
            if ASY and WORLD:   # apply the update launched after the previous segment (it ran while this one was made), then launch this segment's
                if FUT is not None:
                    nch, zc2 = FUT.result(); t_w2 = time.time(); sync_cells(); t_a = time.time(); zcf = zc2.to(dev); torch.cuda.synchronize(); t_b = time.time(); gc = coords()
                    for st_ in sts: st_.set_coords(gc)
                    torch.cuda.synchronize(); print(f"(async) applied maintain {FUT_SPAN}: {nch} pixels, waited {t_w2 - t_m:.1f} s, apply {time.time() - t_w2:.1f} s (sync {t_a - t_w2:.1f} vae {t_b - t_a:.1f} coords {time.time() - t_b:.1f}) | unknown later {unk(s1 + 1, NF):.3f} | {MT.log[-1]}", flush=True)
                    log.write(f"(async) maintain {FUT_SPAN}: {nch} | waited {time.time() - t_m:.0f} s\n"); log.flush()
                FUT = POOL.submit(maint_job, list(made), s0, s1, s1 + 1); FUT_SPAN = f"{s0}-{s1}"; last_px = s1
                continue
            nch = MT.update_world(made, s0, s1, s1 + 1) if WORLD else MT.update(made, s0, s1, s1 + 1)
            sync_cells(); last_px = s1
            zcf = cond_lat(); gc = coords()
            for st_ in sts: st_.set_coords(gc)
            print(f"maintain {s0}-{s1}: {nch} pixels | unknown in later frames {u0:.3f} -> {unk(s1 + 1, NF):.3f} | {time.time() - t_m:.0f} s | {MT.log[-1:]}", flush=True)
            log.write(f"maintain {s0}-{s1}: {nch} | unknown {u0:.3f}->{unk(s1 + 1, NF):.3f} | {MT.log[-1:]}\n"); log.flush()
    vid = np.stack(made[:NF]); cv = cond[:NF]
    for p_, v in ((a.out, np.concatenate([vid, cv], 2)), (a.out[:-4] + "_gen.mp4", vid), (a.out[:-4] + "_cond7.mp4", np.concatenate([vid, v7c[:NF]], 2))):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{v.shape[2]}x{v.shape[1]}", "-r", "30", "-i", "-", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", p_], input=np.ascontiguousarray(v).tobytes(), check=True)
    np.save(a.out[:-4] + "_tier.npy", t7[:NF].numpy().astype(np.uint8))
    print("INFER wrote", a.out, vid.shape, flush=True)


if __name__ == "__main__":
    main()
