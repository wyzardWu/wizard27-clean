#!/bin/bash
# Stage 2: causal teacher forcing from the bidirectional teacher (history window 20 + sink 1).
#   bash scripts/train/2_causal.sh <bidirectional dit> <out dir> [max steps]
set -e; source "$(dirname "$0")/common.sh"
launch fsdp.yaml wizard.train.causal --data $D/wan_v7 --out $2 --init $1 --step0 0 --max_steps ${3:-3000} --save_every 500 --num_workers 6 --window 20 --sink 1 \
  $MIX --p_memunk 0 --mem_loss_w 3 --mem_index mem_index.pt --extra_rows $ROWS --p_trueunk 0 --p_unk2 0.5 --log_every 5
