#!/usr/bin/env bash
# Start or resume the SigLIP2 ViT-L pipeline. Safe to re-run after a shutdown:
# finished folds are skipped and an unfinished one resumes from its last epoch.
cd "$(dirname "$0")"
nohup setsid ./run_pipeline.sh siglip2_l16_v1 --backbone siglip2_l16 --epochs 5 --batch-size 32 \
  --lr 2e-5 --head-lr-mult 50 --layer-decay 0.8 --lp-epochs 1 --drop-path 0.1 --weight-decay 0.05 \
  --clip-grad 1.0 --ema 0.999 --mask-unlabeled --strong-lowres --jpeg-p 0.3 \
  >> runs/siglip2_l16_v1.log 2>&1 < /dev/null &
echo "SigLIP2 pipeline running in the background (log: runs/siglip2_l16_v1.log)"
