#!/bin/bash
# Stage 3: consistency distillation into a 2-step causal student (teacher reads the clean history, student / EMA the self-augmented one).
#   bash scripts/train/4_consistency.sh <causal teacher dit> <out dir> [max steps]
set -e; source "$(dirname "$0")/common.sh"
export SA_P=${SA_P:-0.7} SA_MAX=${SA_MAX:-0.5} SA_GROUPS=${SA_GROUPS:-lookback,lookback0,sfly,sfly_replay}
launch fsdp_dmd.yaml wizard.distill.consistency_selfaug --data $D/wan_v7 --ckpt $1 --out $2 --max_steps ${3:-10000} --save_every 500 --cfg 3 --window 20 --sink 1 \
  $MIX,lookback:4,lookback0:1 --mem_index mem_index.pt --extra_rows $ROWS,$D/wan_lookback_geo/rows_full.jsonl --p_trueunk 0 --p_unk2 0.5
