#!/bin/bash
# Stage 4: DMD of the 2-step student; real / fake score = the bidirectional teacher (CFG 3). Batch = NGPU x accum 4.
#   bash scripts/train/5_dmd.sh <consistency EMA> <bidirectional dit> <out dir> [max steps]
# To continue a run: STUDENT=<out>/ema/ema_step<N>.safetensors and EXTRA="--fake <out>/fake_last.safetensors --step0 <N>".
set -e; source "$(dirname "$0")/common.sh"
launch fsdp_dmd.yaml wizard.distill.dmd --data $D/wan_v7 --student $1 --real $2 --out $3 --accum 4 --max_steps ${4:-400} --save_every 15 --cfg 3 \
  --levels 1000,500 --num_workers 4 \
  --group_rep e1o:8,e2o:8,e1n:8,e2n:8,e1x:8,turn:16,roam:4,sfly:8,sfly_replay:12,lookback:8,lookback0:2,ww:2,er:2,ue:2,bk:2 --rep_mem 16 \
  --rep_keys $D/wan_memroam/rep_keys_both30.json --mem_index mem_index.pt --extra_rows $ROWS,$D/wan_lookback_geo/rows_full.jsonl --p_trueunk 0 --p_unk2 0.5 $EXTRA
