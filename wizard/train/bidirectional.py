"""Stage 1: the bidirectional teacher.

Trains Wan2.2-TI2V-5B with the visual proxy (height / depth / tier + normals, skeletons, shields) concatenated on the
channels, world-coordinate tokens on background cells, retrieved memory frames and gated reference images. Also holds the
dataset (`ClipDataset`), `collate` and the data-root constants used by the later stages.

    accelerate launch --config_file configs/accelerate/fsdp.yaml --num_processes 8 -m wizard.train.bidirectional \
        --data $WIZARD_DATA/wan_v7 --out runs/bidir [data / mixing arguments]"""
import os, sys, time, argparse, json, random, collections, math
import numpy as np
import torch, torch.nn.functional as F
HERE = os.path.dirname(os.path.abspath(__file__))
from wizard.models.dit import WizardDiT, ti2v_5b_config, tiny_config
from wizard.models.flow_match import FlowMatchScheduler
from wizard.models.cond import attach_cond, load_official, cat_cond
from wizard.models.world_cond import attach_world, scene_gbuf, ident_aug, M_MAX, CH, HW
from wizard.memory.retrieve import retrieve_chunks, chunk_pixels, MIN_GAIN, seen
import wizard.data.gbuffer as G
from wizard.data.env_fill import fill_env, smooth_env
from wizard.data.roam_actor import roam_actor
from wizard.paths import WAN_CKPT, DATA_ROOT
CK = WAN_CKPT; DS = DATA_ROOT; MEMROAM = f"{DS}/wan_memroam"; SFLY = f"{DS}/wan_sfly"; V7 = f"{DS}/wan_v7/cond7"


def roam_depth(d):
    """the roaming window's depth with the player's hole filled by the environment behind (wizard.data.roam_actor + fill_env)"""
    z = d.float().numpy(); a = roam_actor(z); return torch.from_numpy(smooth_env(np.stack([fill_env(z[t], a[t]) for t in range(len(z))]), a))


def lt(p): return torch.load(p, map_location="cpu", weights_only=False)


def unk_nocoord(dep, unk, g, mem_dep, mem_pos, mem_R, mem_K, n, use, dep_w=None, ret_cov=False):
    """unknown (tier 1, dark green) cells that neither the window's first frame nor a memory frame its chunk
    reads covers -> depth 0, so no coordinate token there (at inference only observed places have coordinates). Returns (dep, drop mask)."""
    from wizard.memory.retrieve import proj_cover
    T = dep.shape[0]; Kw = g["K"].float().reshape(-1, 4)[0]; pf = [0] + [4 * j for j in range(1, 31)]
    Dt = (dep if dep_w is None else dep_w).float()
    d0 = Dt[0].clone(); ch0 = (d0 <= 0) & (unk.reshape(dep.shape)[0].float() > 0)              # frame 0's characters: the ground behind them counts as seen (inference fills the cut-out before meshing)
    if ch0.any() and (~ch0).any():
        from scipy import ndimage as _ndi
        _, (iy, ix) = _ndi.distance_transform_edt(ch0.numpy(), return_indices=True); d0 = torch.where(ch0, d0[torch.from_numpy(iy), torch.from_numpy(ix)], d0)
    P0 = G.world_points(d0[None], g["pos"][:1].float(), g["R"][:1].float(), Kw)[0][d0 > 0]
    cov = proj_cover(P0, g["pos"][pf].float(), g["R"][pf].float(), Kw, 60, 104)
    for c, lats in enumerate(CH):
        for j in range(n):
            if bool(use[c, j]):
                Pm = G.world_points(mem_dep[j:j + 1].float(), mem_pos[j:j + 1], mem_R[j:j + 1], mem_K[j])[0][mem_dep[j] > 0]
                cov[lats] |= proj_cover(Pm, g["pos"][[pf[l] for l in lats]].float(), g["R"][[pf[l] for l in lats]].float(), Kw, 60, 104)
    cov = (-F.max_pool2d(-F.max_pool2d(cov.float()[:, None], 7, 1, 3), 7, 1, 3))[:, 0] > 0.5   # closing: near ground seen larger than at frame 0 leaves holes between projected points
    lat_of = torch.tensor([0] + [(t + 3) // 4 for t in range(1, T)])
    drop = ((unk.reshape(dep.shape).float() * 3).round() == 1) & ~cov[lat_of]
    if ret_cov: return torch.where(drop, torch.zeros_like(dep), dep), drop, cov[lat_of]
    return torch.where(drop, torch.zeros_like(dep), dep), drop



def cover_dtest(P, pos, R, K, Dt, H, W, tol=0.15):
    """proj_cover with a depth test: a world point counts for a cell of camera t only if its z agrees with that frame's
    own depth Dt[t] there (relative tol), so points hidden behind nearer surfaces or from another place do not mark it seen."""
    fx, fy, cx, cy = [float(v) for v in torch.as_tensor(K).reshape(-1)[:4]]
    T = pos.shape[0]; cov = torch.zeros(T, H, W, dtype=torch.bool)
    for t in range(T):
        c = (P - pos[t]) @ R[t].T; z = c[:, 2]; ok = z > 0.1
        u = (cx + fx * c[:, 0] / z.clamp(min=1e-3))[ok]; v = (cy - fy * c[:, 1] / z.clamp(min=1e-3))[ok]; zz = z[ok]
        for du in (0, -1):
            for dv in (0, -1):
                ui = (u + du).floor().long(); vi = (v + dv).floor().long(); m = (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
                d = Dt[t][vi[m], ui[m]]; m2 = (d > 0) & ((zz[m] - d).abs() < tol * d)
                cov[t].view(-1)[(vi[m] * W + ui[m])[m2]] = True
    c = cov[:, None].float(); c = -F.max_pool2d(-F.max_pool2d(c, 3, 1, 1), 3, 1, 1)
    return c[:, 0] > 0.5

class ClipDataset(torch.utils.data.Dataset):
    def __init__(self, root, rows, a):
        self.O, self.rows, self.a = root, rows, a
        T = lt(f"{DS}/wan_mix5/text_table.pt"); self.emb, self.L = T["emb"], T["text_len"]
        self.roam_text = lt(f"{MEMROAM}/text.pt")["emb"]; self.mem_index = lt(f"{MEMROAM}/{a.mem_index}"); self.meta = collections.OrderedDict()
        self.mroot = {}                                                                   # SceneFly in the memroam layout under its own root
        for root in (SFLY, SFLY + "_replay", SFLY + "_replay_test"):                                             # replay windows read the same videos' banks (in wan_sfly)
            if os.path.exists(f"{root}/mem_index.pt"):
                si = lt(f"{root}/mem_index.pt"); self.mem_index.update(si); self.mroot.update({v["seg"]: SFLY for v in si.values()})
        self.with_prev = [i for i, r in enumerate(rows) if r["prev"]]

    def __len__(self): return len(self.rows)

    def pad(self, e): return torch.cat([e, e.new_zeros(self.L - e.shape[0], e.shape[1])]) if e.shape[0] < self.L else e[:self.L]

    def bankmeta(self, s):
        if s not in self.meta:
            self.meta[s] = lt(f"{self.mroot.get(s, MEMROAM)}/bank/{s}/meta.pt")
            if len(self.meta) > 16: self.meta.popitem(last=False)
        return self.meta[s]

    def first_frame(self, key):
        """a window's first frame as a memory candidate: latent [48,1,30,52], depth/pos/R/K of its frame 0, its character cover [15,26,3]"""
        c = lt(f"{DS}/wan_mix5/clips/clip_{key}.pt"); g = lt(f"{self.O}/gbuf2/clip_{key}.pt"); p7 = f"{V7}/clip_{key}.pt"
        if os.path.exists(p7): v = lt(p7); dep, tier = torch.nan_to_num(v["dep"][0].float(), posinf=0.0).clamp(0, 1000), v["tier"][0].float()
        else: dep = g["depth"][0].float() * (~g["actor"][0]); tier = (dep > 0).float() * 3
        return dict(lat=c["latent"][:, :1], depth=dep, tier=tier, pos=g["pos"][0], R=g["R"][0], K=g["K"], cover=c["cover"][0].float(), slots=c["ref_slots"], hideall=True)

    def other_frame(self):
        if not self.with_prev:                                                            # memory-only training: an irrelevant frame = a bank frame of another recording
            ix = random.choice(list(self.mem_index.values())); s = ix["seg"]; m = self.bankmeta(s); j = random.randrange(len(m["f"])); f = int(m["f"][j])
            x = lt(f"{self.mroot.get(s, MEMROAM)}/bank/{s}/{f}.pt"); dj = m["depth"][j].float()
            return dict(lat=x["lat"], depth=dj, tier=(dj > 0).float() * 3, pos=m["pos"][j], R=m["R"][j], K=m["K"][j], cover=torch.zeros(15, 26, 3), slots=torch.zeros(2, dtype=torch.long), hidedep=True)
        r = self.rows[random.choice(self.with_prev)]; return self.first_frame(random.choice(r["prev"]))

    def pick(self, per):
        pool = []
        for rk in range(3):
            for p in per:
                if rk < len(p) and p[rk] not in pool and len(pool) < M_MAX: pool.append(p[rk])
        use = torch.zeros(len(CH), M_MAX, dtype=torch.bool)
        for c, p in enumerate(per):
            for f in p:
                if f in pool: use[c, pool.index(f)] = True
        return pool, use

    def __getitem__(self, i):
        r = self.rows[i]; a = self.a
        if r["group"] in ("roam", "sfly", "sfly_replay"): return self.roam(r)                             # SceneFly = roam layout, no characters
        c = lt(f"{DS}/wan_mix5/clips/clip_{r['key']}.pt"); g = lt(f"{self.O}/gbuf2/clip_{r['key']}.pt"); v7 = lt(f"{V7}/clip_{r['key']}.pt"); act = v7["cond"] if not os.environ.get("SKELETON_DEPTH_COND") else c["cond"]
        u2 = f"{V7}_unk2/clip_{r['key']}.pt"
        if getattr(a, "p_unk2", 0.0) > 0 and os.path.exists(u2) and random.random() < a.p_unk2: v7 = lt(u2); act = v7["cond"]   # unk2_build.py
        dep = v7["dep"].float(); dep = torch.where(torch.isfinite(dep) & (dep < 1000), dep, torch.zeros_like(dep)); unk = v7["tier"].float() / 3.0                         # characters' cells 0; tier rides in the "unknown" slot
        cands = {k: self.first_frame(k) for k in r["prev"]}; per = [[] for _ in CH]
        if cands:
            P = G.world_points(dep, g["pos"], g["R"], g["K"]); ok = dep > 0; pts, tcs = [], []
            for lats in CH:
                px = chunk_pixels(lats)[::4]; pts.append(P[px][:, ::4, ::4][ok[px][:, ::4, ::4]]); tcs.append(chunk_pixels(lats)[0] / 30.0)
            ks = list(cands); bank = dict(depth=torch.stack([cands[k]["depth"] for k in ks]), pos=torch.stack([cands[k]["pos"] for k in ks]),
                                          R=torch.stack([cands[k]["R"] for k in ks]), K=torch.stack([cands[k]["K"] for k in ks]),
                                          t=torch.tensor([-4.0 * (j + 1) for j in range(len(ks))], dtype=torch.float64))
            picks, _ = retrieve_chunks(pts, bank, tcs, k=3, min_gain=MIN_GAIN, context=dict(depth=dep[:1], pos=g["pos"][:1], R=g["R"][:1], K=g["K"]))
            per = [[ks[j] for j, _ in pk] for pk in picks]
        for c_ in range(len(per)):
            if random.random() < a.p_nomem: per[c_] = []
            elif per[c_] and random.random() < a.p_irrel:
                wk = f"other{c_}"; cands[wk] = self.other_frame(); per[c_][random.randrange(len(per[c_]))] = wk
        pool, use = self.pick(per); fr = [cands[k] for k in pool]
        return self.pack(c["latent"], act, self.pad(self.emb[c["prompt"]]), c["ref_latents"], c["ref_slots"], c["cover"], g, dep, unk, fr, use, c["ref_slots"])

    def roam(self, r):
        a = self.a; name = os.path.basename(r["video"]); d = lt(r["video"]); ix = self.mem_index[name]; s = ix["seg"]; m = self.bankmeta(s)
        per = []
        for ch in ix["chunks"]:
            p = [f for f, gn in ch if gn >= MIN_GAIN][:3]
            if random.random() < a.p_nomem: p = []
            elif p and random.random() < a.p_irrel: p[random.randrange(len(p))] = "other"
            per.append(p)
        pool, use = self.pick(per); fr = []
        for f in pool:
            if f == "other": fr.append(self.other_frame()); continue
            j = int((m["f"] == f).nonzero()[0]); x = lt(f"{self.mroot.get(s, MEMROAM)}/bank/{s}/{f}.pt")
            dj = m["depth"][j].float(); fr.append(dict(lat=x["lat"], depth=dj, tier=(dj > 0).float() * 3, pos=m["pos"][j], R=m["R"][j], K=m["K"][j], cover=torch.zeros(15, 26, 3), slots=torch.zeros(2, dtype=torch.long), hidedep=True))
        g = dict(pos=d["pos"], R=d["R"], K=d["K"], origin=d["origin"], hp=d["hp"])
        v7 = lt(f"{V7}/clip_{name[5:-3]}.pt")
        mu = f"{V7}_memunk/clip_{name[5:-3]}.pt"                                          # the environment unknown in the video, coordinates kept
        if a.p_memunk > 0 and pool and os.path.exists(mu) and random.random() < a.p_memunk: v7 = lt(mu)
        tu = f"{V7}_{os.environ.get('TU_SUB', 'trueunk')}/clip_{name[5:-3]}.pt"   # TU_SUB=unkcoord: the eval variant with coordinates kept
        dep_w = torch.nan_to_num(v7["dep"].float(), posinf=0.0).clamp(0, 1000)               # the memory-only loss weight needs the true geometry: trueunk zeroes it
        if getattr(a, "p_trueunk", 0.0) > 0 and pool and os.path.exists(tu) and random.random() < a.p_trueunk: v7 = lt(tu)
        elif getattr(a, "p_unk2", 0.0) > 0 and os.path.exists(f"{V7}_unk2/clip_{name[5:-3]}.pt") and random.random() < a.p_unk2: v7 = lt(f"{V7}_unk2/clip_{name[5:-3]}.pt")
        return self.pack(d["latent"], v7["cond"], self.pad(self.roam_text), torch.zeros(2, 48, 1, 30, 30, dtype=torch.bfloat16), torch.zeros(2, dtype=torch.long),
                         torch.zeros(31, 15, 26, 3, dtype=torch.float16), g, torch.nan_to_num(v7["dep"].float(), posinf=0.0).clamp(0, 1000), v7["tier"].float() / 3.0, fr, use, torch.zeros(2, dtype=torch.long), dep_w=dep_w)

    def pack(self, latent, actor, ctx, ref_lat, ref_slots, cover, g, dep, unk, fr, use, slots_now, dep_w=None):
        n = len(fr); z = lambda shape, dt: torch.zeros((M_MAX,) + shape, dtype=dt)
        mem_lat, mem_dep, mem_pos, mem_R, mem_K, hide = z((48, 1, 30, 52), torch.bfloat16), z((60, 104), torch.float32), z((3,), torch.float32), z((3, 3), torch.float32), z((4,), torch.float32), z((HW,), torch.bool)
        mem_tier = z((60, 104), torch.float32)
        live = [int(s) for s in slots_now.tolist() if int(s) > 0]                     # the slots that have a reference in THIS sample
        for j, f in enumerate(fr):
            mem_lat[j] = f["lat"].to(torch.bfloat16); mem_dep[j] = f["depth"]; mem_pos[j] = f["pos"]; mem_R[j] = f["R"]; mem_K[j] = f["K"].float(); mem_tier[j] = f.get("tier", (f["depth"] > 0).float() * 3)
            cov = f["cover"]; hm = cov.max(-1).values > 0.02                                # every character, not only the referenced ones
            if f.get("hidedep"): hm |= F.max_pool2d((f["depth"] == 0).float()[None, None], 4)[0, 0] > 0   # roam bank: the player's cells carry no depth
            hide[j] = (F.max_pool2d(hm.float()[None, None], 3, 1, 1)[0, 0] > 0).reshape(-1)            # grown one token
        memw = torch.ones(31, 30, 52)
        if n and self.a.mem_loss_w > 1:                                                   # where only memory can tell: not seen by the window's
            try:                                                                          # first frame, seen by a memory frame its chunk reads -> loss weight mem_loss_w
                Kw = g["K"].float().reshape(-1, 4)[0]; pf = [0] + [4 * j for j in range(1, 31)]
                D = (dep if dep_w is None else dep_w)[pf].float(); P = G.world_points(D, g["pos"][pf].float(), g["R"][pf].float(), Kw); ok = D > 0
                COVER = os.environ.get("MEMW_COVER", "1") == "1"
                if COVER:
                    from wizard.memory.retrieve import proj_cover
                    s0 = proj_cover(P[0][ok[0]], g["pos"][pf].float(), g["R"][pf].float(), Kw, 60, 104)
                    HWIN = int(os.environ.get("MEMW_HISTW", "0"))
                    if HWIN > 0:
                        # seen = by any clean history frame the noisy frame reads (window W + sink), not only frame 0:
                        # the weight then sits on first reveals and on places that slid out of the window, where only memory can tell
                        SK = int(os.environ.get("MEMW_SINK", "1")); s0 = torch.zeros(D.shape, dtype=torch.bool)
                        for gi in range(30):
                            tg = [l for l in range(gi + 1, 31) if l < gi + HWIN or gi < SK]
                            if not tg or not bool(ok[gi].any()): continue
                            s0[tg] |= cover_dtest(P[gi][ok[gi]], g["pos"][[pf[l] for l in tg]].float(), g["R"][[pf[l] for l in tg]].float(), Kw, D[tg], 60, 104)
                else:
                    s0 = seen(P.reshape(-1, 3), D[:1], g["pos"][:1].float(), g["R"][:1].float(), Kw)[0].reshape(D.shape)
                mk = torch.zeros(D.shape, dtype=torch.bool)
                for c, lats in enumerate(CH):
                    sl = [j for j in range(n) if bool(use[c, j])]
                    if not sl: continue
                    if COVER:
                        sm = torch.zeros(len(lats), 60, 104, dtype=torch.bool)
                        for j in sl:
                            Pm = G.world_points(mem_dep[j:j + 1].float(), mem_pos[j:j + 1], mem_R[j:j + 1], mem_K[j])[0][mem_dep[j] > 0]
                            sm |= proj_cover(Pm, g["pos"][[pf[l] for l in lats]].float(), g["R"][[pf[l] for l in lats]].float(), Kw, 60, 104)
                    else:
                        Pc = P[lats].reshape(-1, 3); sm = seen(Pc, mem_dep[sl], mem_pos[sl], mem_R[sl], mem_K[sl]).any(0).reshape(len(lats), 60, 104)
                    mk[lats] = sm
                mk = mk & ok & ~s0; mk[0] = False
                memw = 1.0 + (self.a.mem_loss_w - 1.0) * F.max_pool2d(mk.float()[:, None], 2)[:, 0]
            except Exception as ex: print("memw failed", repr(ex)[:120], flush=True)
        if os.environ.get("UNK_NOCOORD") == "1" and unk is not None:                     # every unknown (tier 1) cell: no coordinate token
            dep = torch.where((unk.reshape(dep.shape).float() * 3).round() == 1, torch.zeros_like(dep), dep)
        return dict(memw=memw, latent=latent, actor=actor, context=ctx, ref_lat=ref_lat, ref_slots=ref_slots.long(), cover=cover.float(), depth=dep, pos=g["pos"], R=g["R"], K=g["K"].float(),
                    origin=g["origin"], hp=torch.tensor(float(g["hp"])), unknown=(unk if unk is not None else torch.zeros(dep.shape, dtype=torch.bool)),
                    mem_lat=mem_lat, mem_depth=mem_dep, mem_tier=mem_tier, mem_pos=mem_pos, mem_R=mem_R, mem_K=mem_K, mem_n=torch.tensor(n), mem_use=use, mem_hide=hide)


def collate(b): return {k: torch.stack([x[k] for x in b]) for k in b[0]}


def loss_fn(model, core, sched, b, neg, a, dev):
    lat = b["latent"].to(dev).float(); act = b["actor"].to(dev).float(); mem = b["mem_lat"].to(dev).float()[:, :, :, 0].permute(0, 2, 1, 3, 4)
    if random.random() < a.p_memnoise: am = random.uniform(0, a.memnoise); mem = (1 - am) * mem + am * torch.randn_like(mem)   # memories at inference are generated frames
    B, C, Fl = lat.shape[:3]; assert B == 1
    idx = sched.sample_timestep_ids(B, dev); noise = torch.randn_like(lat); noisy = sched.add_noise(lat, noise, idx)
    ts = sched.timesteps.to(dev, torch.float32)[idx][:, None].expand(B, Fl).clone(); noisy[:, :, 0] = lat[:, :, 0]; ts[:, 0] = 0.0   # first frame always given
    gc = scene_gbuf(b["depth"][0].to(dev), b["pos"][0].to(dev), b["R"][0].to(dev), b["K"][0], b["origin"][0].to(dev), float(b["hp"][0]), None)
    tier_c = b["unknown"][0].to(dev).float()
    n = int(b["mem_n"][0]); gm = torch.zeros(M_MAX, 8, 60, 104, device=dev)
    for j in range(n):
        gm[j] = scene_gbuf(b["mem_depth"][0, j:j + 1].to(dev), b["mem_pos"][0, j:j + 1].to(dev), b["mem_R"][0, j:j + 1].to(dev), b["mem_K"][0, j], b["origin"][0].to(dev), float(b["hp"][0]))[:, 0]
    if random.random() < a.p_ident:                                                     # coordinates as identities: same piecewise rigid transform
        aug = ident_aug([gc] + [gm[j:j + 1].permute(1, 0, 2, 3) for j in range(n)]); gc = aug[0]
        for j in range(n): gm[j] = aug[1 + j][:, 0]
    tier_m = b["mem_tier"][0].to(dev).float() / 3.0                                     # tokens = world coordinate + has bit + tier; no depth / normals
    gc = torch.cat([gc[:3], (gc[3:4] > 0).to(gc.dtype), torch.zeros_like(gc[4:7]), tier_c[None].to(gc.dtype)], 0)
    gm = torch.cat([gm[:, :3], (gm[:, 3:4] > 0).to(gm.dtype), torch.zeros_like(gm[:, 4:7]), tier_m[:, None].to(gm.dtype)], 1)
    if getattr(a, "no_coords", False): gc[:4] = 0; gm[:, :4] = 0                       # no coordinate / has tokens, tier kept
    core._gb_in = (gc[None], gm[None])
    core._mem_warp = None
    core._mem_wrope = None
    core._mem_wrope_dbg = None
    if a.mem_wrope and n:                                                                  # world RoPE for reading memory (true depth / cameras, un-augmented)
        from wizard.models.world_cond import world_rope_layout
        core._mem_wrope = world_rope_layout(b["mem_depth"][0].to(dev), b["mem_pos"][0].to(dev), b["mem_R"][0].to(dev), b["mem_K"][0].to(dev), n, b["mem_hide"][0], b["mem_use"][0], b["depth"][0].to(dev), b["pos"][0].to(dev), b["R"][0].to(dev), b["K"][0])
        if core._mem_wrope is not None:
            L_ = core._mem_wrope; core._mem_wrope_dbg = dict(nw=int(L_["ksrc"].numel()), qfin=bool(torch.isfinite(L_["qpos"]).all()), kfin=bool(torch.isfinite(L_["kpos"]).all()), kmax=float(L_["kpos"].abs().max()), qmax=float(L_["qpos"].abs().max()), chunks=torch.bincount(L_["kchunk"], minlength=8).tolist())
    if a.mem_warp and n:                                                                   # memory RoPE warping, from the true (un-augmented) depth and cameras
        from wizard.models.world_cond import mem_warp_pos
        core._mem_warp = mem_warp_pos(b["mem_depth"][0].to(dev), b["mem_pos"][0].to(dev), b["mem_R"][0].to(dev), b["mem_K"][0].to(dev), b["pos"][0].to(dev), b["R"][0].to(dev), b["K"][0], b["mem_use"][0], n)
    ctx = b["context"].to(dev).to(torch.bfloat16)
    if random.random() < a.p_text_drop: ctx = neg.to(dev).to(torch.bfloat16)
    x = torch.cat([mem, noisy], 2); cond = torch.cat([torch.zeros_like(mem), act], 2); tsa = torch.cat([torch.zeros(B, M_MAX, device=dev), ts], 1)
    ref = dict(lat=b["ref_lat"].to(dev), slots=b["ref_slots"].to(dev), cover=b["cover"].to(dev), mem_use=b["mem_use"], mem_n=b["mem_n"], mem_hide=b["mem_hide"])
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        v = model(cat_cond(x, cond), tsa, ctx, ref=ref)
    core._gb_in = None
    se = ((v.float() - (noise - lat)).pow(2) * b["memw"].to(dev).float()[:, None]).mean(dim=(1, 3, 4)); keep = torch.ones_like(se); keep[:, 0] = 0.0   # memory-only places weighted
    w = sched.training_weight(idx).float()[:, None]; return (w * se * keep).sum() / keep.sum().clamp(min=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); ap.add_argument("--init", default=""); ap.add_argument("--step0", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-5); ap.add_argument("--lr_cond", type=float, default=1e-4)
    ap.add_argument("--max_steps", type=int, default=30000); ap.add_argument("--warmup", type=int, default=200); ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--log_every", type=int, default=20); ap.add_argument("--num_workers", type=int, default=6); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p_irrel", type=float, default=0.2); ap.add_argument("--p_nomem", type=float, default=0.1); ap.add_argument("--p_text_drop", type=float, default=0.1)
    ap.add_argument("--p_ident", type=float, default=0.3); ap.add_argument("--skeleton_depth_cond", action="store_true"); ap.add_argument("--coords_only", action="store_true"); ap.add_argument("--no_coords", action="store_true")
    ap.add_argument("--p_memnoise", type=float, default=0.5); ap.add_argument("--memnoise", type=float, default=0.15)
    ap.add_argument("--empty_actor", action="store_true", help="full G-buffer (coords + depth + normals) but character cells get NO geometry (not filled); roam: raw depth (player = 0)")
    ap.add_argument("--group_rep", default="e1o:2,e2o:2,e1n:2,e2n:2,e1x:2,turn:2,roam:2"); ap.add_argument("--dry_run", action="store_true"); ap.add_argument("--rep_mem", type=int, default=0); ap.add_argument("--p_memunk", type=float, default=0.0); ap.add_argument("--mem_loss_w", type=float, default=1.0); ap.add_argument("--extra_rows", default=""); ap.add_argument("--mem_index", default="mem_index.pt"); ap.add_argument("--rep_keys", default=""); ap.add_argument("--mem_warp", action="store_true"); ap.add_argument("--mem_wrope", action="store_true"); ap.add_argument("--grad_accum", type=int, default=1); ap.add_argument("--p_trueunk", type=float, default=0.0); ap.add_argument("--p_unk2", type=float, default=0.0); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    accel = Accelerator(); set_seed(a.seed, device_specific=True); random.seed(a.seed + accel.process_index)
    main_ = accel.is_main_process; os.makedirs(a.out, exist_ok=True)
    log = (lambda *x: (print(*x, flush=True), open(os.path.join(a.out, "train.log"), "a").write(" ".join(map(str, x)) + "\n"))) if main_ else (lambda *x: None)
    model = WizardDiT(**(tiny_config() if a.dry_run else ti2v_5b_config()), causal=False, zero_init_head=False)
    if not a.dry_run and not a.init: info = load_official(model, CK); log(f"official Wan2.2-TI2V-5B: missing {len(info.missing_keys)} unexpected {len(info.unexpected_keys)}")
    attach_cond(model); attach_world(model); model.use_gradient_checkpointing = True; core = model
    if a.init:
        import safetensors.torch as st; log(f"init from {a.init}: {model.load_state_dict(st.load_file(a.init), strict=False)}")
    rows = [json.loads(l) for l in open(os.path.join(a.data, "v7_rows.jsonl"))]
    for er in [x for x in a.extra_rows.split(",") if x]: rows += [json.loads(l) for l in open(er)]   # e.g. SceneFly rows (group sfly)
    if a.limit: rows = rows[:a.limit] + [r for r in rows if r["group"] == "roam"][:a.limit]
    rep = {k: int(v) for k, v in (x.split(":") for x in a.group_rep.split(","))}
    mi = json.load(open(f"{MEMROAM}/index.json")) if a.rep_mem else {}                 # memory-rich roaming windows (best memory overlap >= 0.4, index_mem.py) x rep_mem
    rk = set(json.load(open(a.rep_keys))) if a.rep_keys else None                        # the weighted roam windows = memory-only share >= 0.3 (both directions)
    nrep = lambda r: a.rep_mem if (a.rep_mem and r["group"] == "roam" and ((r["key"] in rk) if rk is not None else mi.get(f"clip_{r['key']}.pt", {}).get("best", 0) >= 0.4)) else rep.get(r["group"], 1)
    train = [r for r in rows for _ in range(nrep(r))]
    c = collections.Counter(r["group"] for r in train); log(f"rows {len(rows)} -> {len(train)} with group_rep {rep}:", {k: f"{v} ({100 * v / len(train):.1f}%)" for k, v in sorted(c.items())})
    if main_: json.dump({"args": vars(a), "rows": len(rows)}, open(os.path.join(a.out, "args.json"), "w"), indent=1)
    neg = lt(f"{DS}/wan_mix5/neg_emb.pt")["neg_emb"]; neg = torch.cat([neg, neg.new_zeros(512 - neg.shape[0], neg.shape[1])])[None]   # the empty prompt, zero-padded to 512
    dl = torch.utils.data.DataLoader(ClipDataset(a.data, train, a), batch_size=1, shuffle=True, num_workers=a.num_workers, collate_fn=collate, drop_last=True, pin_memory=True, persistent_workers=True)
    hi = ("cond_embedding", "gbuf_embedding")
    opt = torch.optim.AdamW([dict(params=[p for n_, p in model.named_parameters() if not n_.startswith(hi)], lr=a.lr),
                             dict(params=[p for n_, p in model.named_parameters() if n_.startswith(hi)], lr=a.lr_cond)], weight_decay=0.01, betas=(0.9, 0.999))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))
    model, opt, dl, sch = accel.prepare(model, opt, dl, sch)
    sched = FlowMatchScheduler(shift=5.0).to(accel.device)
    step, t0, it = a.step0, time.time(), iter(dl); model.train()
    while step < a.max_steps:
        try: b = next(it)
        except StopIteration: it = iter(dl); b = next(it)
        loss = loss_fn(model, core, sched, b, neg, a, accel.device)
        if a.grad_accum > 1:                                                               # a batch of grad_accum samples per GPU per step
            accel.backward(loss / a.grad_accum)
            for _ in range(a.grad_accum - 2):
                try: b = next(it)
                except StopIteration: it = iter(dl); b = next(it)
                accel.backward(loss_fn(model, core, sched, b, neg, a, accel.device) / a.grad_accum)
            try: b = next(it)
            except StopIteration: it = iter(dl); b = next(it)
            loss = loss_fn(model, core, sched, b, neg, a, accel.device); loss_bw = loss / a.grad_accum
        else: loss_bw = loss
        if os.environ.get("DEBUG_NAN") and not torch.isfinite(loss):
            w = getattr(core, "_mem_wrope_dbg", None)
            print(f"NAN rank {accel.process_index} step {step} mem_n {int(b['mem_n'][0])} wrope {w}", flush=True)
        accel.backward(loss_bw); gn = accel.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(gn):                                                         # a non-finite gradient never reaches the weights
            print(f"NONFINITE_GRAD rank {accel.process_index} step {step} loss {float(loss):.4f} mem_n {int(b['mem_n'][0])} wrope {getattr(core, '_mem_wrope_dbg', None)}", flush=True)
            opt.zero_grad(); step += 1; sch.step(); continue
        opt.step(); opt.zero_grad(); sch.step(); step += 1
        if step % a.log_every == 0:
            l = accel.gather(loss.detach()).mean().item(); sps = a.log_every / (time.time() - t0); t0 = time.time()
            log(f"step {step:6d}/{a.max_steps} loss {l:.4f} gnorm {float(gn):.3f} {1 / max(sps, 1e-9):.2f} s/step")
        if step % a.save_every == 0 or step == a.max_steps:
            sd = accel.get_state_dict(model)
            if main_:
                import safetensors.torch as st
                os.makedirs(os.path.join(a.out, "dits"), exist_ok=True)
                st.save_file({k: v.contiguous().to(torch.bfloat16) for k, v in sd.items()}, os.path.join(a.out, "dits", f"dit_step{step}.safetensors")); log(f"saved step {step}")
                old = step - 2 * a.save_every                                         # (disk) keep every 5000th + the latest two
                if old > 0 and old % 5000: [os.remove(f) for f in [os.path.join(a.out, "dits", f"dit_step{old}.safetensors")] if os.path.exists(f)]
            accel.wait_for_everyone()
    log("TRAIN_DONE")


if __name__ == "__main__":
    main()
