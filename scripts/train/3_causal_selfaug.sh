#!/bin/bash
# Stage 2b: causal teacher forcing with history self-augmentation, adding the look-back (revisit) data.
#   bash scripts/train/3_causal_selfaug.sh <causal dit> <step0> <out dir> [max steps]
# Ours: from the stage-2 checkpoint at 3000, to 3500 with SA_P=0.7 SA_MAX=0.5 SA_GROUPS=all, then to 11000 with the defaults below.
set -e; source "$(dirname "$0")/common.sh"
export MEMW_HISTW=20 MEMW_SINK=1 MEM_FUTURE=1 SA_P=${SA_P:-0.5} SA_MAX=${SA_MAX:-0.25} SA_GROUPS=${SA_GROUPS:-lookback,lookback0,sfly,sfly_replay}
launch fsdp.yaml wizard.train.causal_selfaug --data $D/wan_v7 --out $3 --init $1 --step0 $2 --max_steps ${4:-11000} --seed 5 --save_every 500 --num_workers 6 \
  --window 20 --sink 1 $MIX,lookback:4,lookback0:1 --p_memunk 0 --mem_loss_w 3 --mem_index mem_index.pt \
  --extra_rows $ROWS,$D/wan_lookback_geo/rows_full.jsonl --p_trueunk 0 --p_unk2 0.5 --log_every 5 --grad_accum ${GRAD_ACCUM:-1}
