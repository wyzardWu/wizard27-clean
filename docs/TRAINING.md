# Training

All stages run on 8 GPUs with FSDP (`NGPU` changes the count). Set the data and checkpoint locations first:

```bash
export WAN_CKPT=/path/to/Wan2.2-TI2V-5B WIZARD_DATA=/path/to/data
```

## Pipeline

```bash
bash scripts/train/1_bidirectional.sh runs/bidir                                   # 0 -> 30k
PHASE=2 INIT=runs/bidir/dits/dit_step30000.safetensors STEP0=30000 \
  bash scripts/train/1_bidirectional.sh runs/bidir                                 # 30k -> 35k
bash scripts/train/2_causal.sh runs/bidir/dits/dit_step35000.safetensors runs/causal
SA_P=0.7 SA_MAX=0.5 SA_GROUPS=all \
  bash scripts/train/3_causal_selfaug.sh runs/causal/dits/dit_step3000.safetensors 3000 runs/causal_sa 3500
bash scripts/train/3_causal_selfaug.sh runs/causal_sa/dits/dit_step3500.safetensors 3500 runs/causal_sa2 11000
bash scripts/train/4_consistency.sh runs/causal_sa2/dits/dit_step11000.safetensors runs/cd
bash scripts/train/5_dmd.sh runs/cd/ema/ema_step10000.safetensors runs/bidir/dits/dit_step35000.safetensors runs/dmd
```

## Stages

| Stage | Steps | Batch | LR | Notes |
|---|---|---|---|---|
| 1. Bidirectional | 30k + 5k | 8 | 1e-5 (condition layers 1e-4) | phase 2: unknown cells carry no coordinates, `p_unk2 0.5` |
| 2. Causal teacher forcing | 3k | 8 | 1e-5 | history window 20 + sink 1; memory from the past only |
| 2b. History self-augmentation | 3k → 3.5k → 11k | 8 | 1e-5 | `SA_P 0.7 / SA_MAX 0.5` on all clips, then `0.5 / 0.25` on revisit clips |
| 3. Consistency distillation | 10k | 8 | 2e-6, EMA 0.99 | N = 48 levels, 2 sampling steps (1000, 500), teacher CFG 3 |
| 4. DMD | 400+ | 32 (8 × accum 4) | 2e-6 / critic 4e-7 | critic : student = 5 : 1, real score CFG 3 |

Hyper-parameters not set in the scripts keep the defaults of each module's argument parser.

**History self-augmentation.** Under teacher forcing the clean history already contains almost everything, so the model
never learns that memory should override a drifting history. With probability `SA_P` the history is replaced by a
one-step estimate from a noised copy (`σ ~ U(0, SA_MAX)`); frame 0, the memory frames and the target stay clean. In
stage 3 the teacher keeps the clean history and only the student / EMA read the augmented one.

**Resuming DMD.** Every save also writes `student_last.safetensors` and `fake_last.safetensors`:
`EXTRA="--fake runs/dmd/fake_last.safetensors --step0 <N>" bash scripts/train/5_dmd.sh runs/dmd/ema/ema_step<N>.safetensors ...`

## Training-time switches (environment)

| Variable | Used by | Meaning |
|---|---|---|
| `UNK_NOCOORD=1` | all | unknown cells carry no world coordinates |
| `MEMW_COVER=1` | all | cells the window's first frame has not seen but a retrieved memory frame has get loss weight `--mem_loss_w`; "seen" by projected coverage |
| `MEMW_HISTW`, `MEMW_SINK` | 2b | "seen" also by every clean history frame the noisy frame reads (window + sink), so the weight sits where only memory can tell |
| `MEM_FUTURE=1` | 2b | SceneFly windows may also retrieve frames after the window |
| `SA_P`, `SA_MAX`, `SA_GROUPS` | 2b, 3 | history self-augmentation |
