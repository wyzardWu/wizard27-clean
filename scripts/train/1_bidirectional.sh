#!/bin/bash
# Stage 1: bidirectional teacher, from the official Wan2.2-TI2V-5B weights.
#   bash scripts/train/1_bidirectional.sh <out dir>
# Our teacher was trained in two phases on 8 GPUs: phase 1 (0 -> 30k steps, p_trueunk 0.3, p_memunk 0.5) and
# phase 2 (30k -> 35k: every unknown cell without coordinates, p_unk2 0.5, p_memunk 0). Set PHASE=2 INIT=<phase-1 dit> STEP0=30000 for phase 2.
set -e; source "$(dirname "$0")/common.sh"; OUT=$1
if [ "${PHASE:-1}" = 1 ]; then
  launch fsdp.yaml wizard.train.bidirectional --data $D/wan_v7 --out $OUT --max_steps 30000 --save_every 1000 --num_workers 6 \
    $MIX --p_memunk 0.5 --mem_loss_w 3 --mem_index mem_index_both.pt --extra_rows $ROWS --p_trueunk 0.3
else
  launch fsdp.yaml wizard.train.bidirectional --data $D/wan_v7 --out $OUT --init $INIT --step0 ${STEP0:-30000} --max_steps 35000 --save_every 1000 --num_workers 6 \
    $MIX --p_memunk 0 --mem_loss_w 3 --mem_index mem_index_both.pt --extra_rows $ROWS --p_trueunk 0 --p_unk2 0.5
fi
