# Inference

## Modes

```bash
export WAN_CKPT=/path/to/Wan2.2-TI2V-5B
bash scripts/infer/run_case.sh causal        <dit> <case> out.mp4 [args]   # 2-step student (CD / DMD)
bash scripts/infer/run_case.sh world         <dit> <case> out.mp4 [args]   # 2-step student + online world maintenance
bash scripts/infer/run_case.sh bidirectional <dit> <case> out.mp4 [args]   # bidirectional teacher, 30 steps, CFG 3
```

| Module | Use |
|---|---|
| `wizard.inference.causal` | window by window (241 = 2 × 120 + 1 frames), KV-cached stream |
| `wizard.inference.causal_world` | one continuous stream of any length, world maintained every 8 latent frames |
| `wizard.inference.bidirectional` | the teacher, for reference |

Common arguments: `--ref1 / --ref2` reference images, `--ref-slots 2,0` the character slot each reference binds to,
`--frames`, `--seed`, `--nomem` (no memory frames).

## Online world maintenance

While generating, the frames made so far are reconstructed with MapAnything, fused into a point cache (alignment, outlier
culling and free-space carving as in GEN3C), and the still-unknown cells of the upcoming proxy frames are repainted from
it. Settings used in our evaluations: `MAINT=pc SELF_FIRST=1 MAINT_WORLD=1 COARSE_WORLD=1`.

With two visible GPUs, `FAST=1` (in `run_case.sh`) runs the maintenance asynchronously on the second card, one segment
behind generation, so the generator never waits (≈ 3.5 s per update):

| Variable | Meaning |
|---|---|
| `ASYNC_MAINT=1`, `MAINT_GPU=1` | maintenance on a worker thread bound to the second visible GPU |
| `MAINT_CVD` | physical GPU id for the reconstruction subprocess |
| `RECON_RESIDENT=1` | keep MapAnything loaded in a resident process |
| `PAINT_AHEAD=68`, `PAINT_THREADS=16`, `PC_BATCH=1` | repaint only the next frames, in parallel and batched |

## Helper environments

The reconstruction workers in `wizard/world/workers/` run as subprocesses and may use their own Python environments:

| Variable | Python with |
|---|---|
| `WIZARD_PY_MAPANY` | MapAnything |
| `WIZARD_PY_MOGE` | MoGe-2 |
| `WIZARD_PY_DA3` | Depth Anything 3 |

Each defaults to the current interpreter.
