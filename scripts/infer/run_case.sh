#!/bin/bash
# Inference on a prepared case directory.
#   bash scripts/infer/run_case.sh causal       <dit> <case> <out.mp4> [extra args]    2-step student (CD / DMD)
#   bash scripts/infer/run_case.sh world        <dit> <case> <out.mp4> [extra args]    2-step student + online world maintenance
#   bash scripts/infer/run_case.sh bidirectional <dit> <case> <out.mp4> [extra args]   the bidirectional teacher (30 steps, CFG 3)
# FAST=1 with two visible GPUs runs the world maintenance asynchronously on the second card.
set -e; MODE=$1; DIT=$2; CASE=$3; OUT=$4; shift 4
export WAN_CKPT=${WAN_CKPT:-checkpoints/Wan2.2-TI2V-5B} INF_UNK_NOCOORD=1
case $MODE in
  causal) python -m wizard.inference.causal --dit $DIT --case $CASE --cm_steps 1000,500 --cfg 1 --frames ${FRAMES:-241} --out $OUT "$@" ;;
  world)
    export MAINT=pc SELF_FIRST=1 MAINT_WORLD=1 BACKDROP=0 COARSE_WORLD=${COARSE_WORLD:-1}
    [ "$FAST" = 1 ] && export ASYNC_MAINT=1 MAINT_GPU=1 MAINT_CVD=${MAINT_CVD:-$(echo ${CUDA_VISIBLE_DEVICES:-0,1} | cut -d, -f2)} RECON_RESIDENT=1 PAINT_AHEAD=68 PAINT_THREADS=16 PC_BATCH=1
    python -m wizard.inference.causal_world --dit $DIT --case $CASE --cm_steps 1000,500 --cfg 1 --frames ${FRAMES:-241} --ref-slots 0,0 --maintain --out $OUT "$@" ;;
  bidirectional) python -m wizard.inference.bidirectional --dit $DIT --case $CASE --steps 30 --cfg 3 --frames ${FRAMES:-241} --out $OUT "$@" ;;
  *) echo "mode: causal | world | bidirectional"; exit 1 ;;
esac
