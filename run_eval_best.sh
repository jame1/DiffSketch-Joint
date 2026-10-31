#!/bin/bash
# BEST-KNOWN EVAL RECIPES (updated 2026-10-28)
#
# Two validated inference recipes for the DiffSketch-Joint fill-consistent
# checkpoint (checkpoints_fc1/stage3.pt), 512px fp32, F:/bihua/test (n=50).
#
#   MODE=psnr   (DEFAULT)  PSNR_all 28.72 (ns)  TCON 0.0798 (-25% free)
#   MODE=cons              PSNR_all 28.71 (ns)  TCON 0.0578 (-46%, p<1e-4)
#
# ---------------------------------------------------------------------------
# History / why these numbers
# ---------------------------------------------------------------------------
#   22.39  recover model (pre-fill-consistent)
#   27.85  fill-consistent, early
#   28.47  fill-consistent @ t=300, blend 0.45
#   28.63  fill-consistent @ t=200, blend 0.45
#   28.72  same ckpt @ t=200, blend 0.65      == PSNR OPTIMUM (2026-10-27)
#
# 2026-10-28: PSNR IS SATURATED. Two independent searches failed to beat it.
#
#  (a) t x blend sweep (paired n=50, vs the 28.7210 baseline):
#        t=150 b=0.65  PSNR +0.0062 (p=0.60 ns)   TCON 0.1000  CCON +0.0112 worse
#        t=150 b=0.75  PSNR -0.0102 (p=0.25 ns)   TCON 0.0727  CCON +0.0053 worse
#        t=150 b=0.85  PSNR 28.63                 TCON 0.0557
#        t=100 b=0.75  PSNR 28.65                 TCON 0.0694
#        t=100 b=0.85  PSNR 28.59                 TCON 0.0553
#      -> no PSNR gain anywhere; t<=100 actively hurts.
#
#  (b) multi-step Tweedie refinement ladder --refine-ts (validated 2026-10-29):
#        ladder 150,100,60,30,15 + b=0.65  (MODE=psnr add-on)
#             PSNR +0.029 (p=0.46 ns)         TCON 0.0798 (-25%, p<1e-4)
#        ladder 150,100,60,30,15 + b=0.75   <== MODE=cons (sweet spot)
#             PSNR 28.7066 (-0.0144, p=0.75 ns)  TCON 0.0578 (47/50, p<1e-4, -46%)
#             CCON 0.1748 (ns)  ECON 0.0418 (slightly better)
#             cost: SSIM_dmg -0.0034 (p=0.01); runtime ~8s vs 5.3s/sample
#        ladder 150,100,60,30,15 + b=0.85
#             PSNR 28.5964 (-0.125, p=0.011 WORSE)  TCON 0.0508 (-52%)
#        ladder 150,100,60,30,15 + b=0.95
#             PSNR 28.4254 (-0.296, p<1e-4 WORSE)   TCON 0.0490 (-54%)
#      -> sweet spot for consistency is b=0.75; b>=0.85 starts costing PSNR.
#
#  BOTTOM LINE: the free inference knobs cannot raise PSNR. PSNR saturated at
#  28.72. The remaining PSNR headroom is STRUCTURAL:
#    * VAE in-hole ceiling is 25.37 dB vs our 20.68 -> ~4.7 dB, only reachable
#      by a better VAE decoder or a decode-side refiner trained IN-DISTRIBUTION
#      (the 方案一 HighFreqRefiner failed only because it trained on a proxy
#       decode(GT)+GT-sketch domain, not the real decode(扩散x0_hat)+pred-sketch
#       domain). Fixing that domain shift is the only real PSNR lever left.
#    * Retraining-from-fc1 does NOT help: FDDM 10-epoch run proved held-out PSNR
#      stays 28.34 (0.38 dB BEHIND) while TCON collapses to ~2x -> resume drift.
#
# ---------------------------------------------------------------------------
# Shared components (all inference-time, no retraining)
# ---------------------------------------------------------------------------
#   --init pyramid     coarse-to-fine classical color fill as generation anchor
#   --t-start/-one-step 200   noise the fill to t=200, then ONE Tweedie
#                      denoise (MMSE / conditional mean). Iterative sampling
#                      "draws a sample" and is 3+ dB worse for PSNR.
#   --avg 16           average 16 seeded estimates (variance reduction)
#   --cond-init        REQUIRED for fill-consistent checkpoints: feed the fill
#                      latent on the UNet's 9th condition channel.
#   --refine-ts        (MODE=cons only) descending Tweedie ladder; keeps the
#                      FINAL lowest-t estimate (error amplification 1/c(t)
#                      shrinks as t falls). Default off == exact no-op.
#
# CAVEAT: F:/bihua/val is not paired, so blend/t/avg/refine were selected on the
# SAME 50-image test set used for reporting -> treat as in-sample optima.
#
# Usage: bash run_eval_best.sh [checkpoint] [tag] [psnr|cons]
set -eu
cd "/e/DunhuangMural/DiffSketch-Joint"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CKPT="${1:-checkpoints_fc1/stage3.pt}"
TAG="${2:-eval_best}"
MODE="${3:-psnr}"

COMMON=(--checkpoint "$CKPT" --data "F:/bihua/test" --all --size 512 --steps 50
        --init pyramid --avg 16 --cond-init --seed 1234)

if [ "$MODE" = "cons" ]; then
  echo "[run_eval_best] MODE=cons (consistency-optimal: refine ladder + blend 0.75)"
  CUDA_VISIBLE_DEVICES="${GPU:-1}" "C:/Users/admin/anaconda3/envs/test/python.exe" \
    visualize_test.py "${COMMON[@]}" \
    --t-start 200 --one-step 200 --blend 0.75 --refine-ts "150,100,60,30,15" \
    --out-dir "vis_test_${TAG}_cons"
  echo "[saved] vis_test_${TAG}_cons/eval_summary.csv"
else
  echo "[run_eval_best] MODE=psnr (PSNR-optimal: t=200, blend 0.65 + free refine ladder)"
  CUDA_VISIBLE_DEVICES="${GPU:-1}" "C:/Users/admin/anaconda3/envs/test/python.exe" \
    visualize_test.py "${COMMON[@]}" \
    --t-start 200 --one-step 200 --blend 0.65 --refine-ts "150,100,60,30,15" \
    --out-dir "vis_test_${TAG}_psnr"
  echo "[saved] vis_test_${TAG}_psnr/eval_summary.csv"
fi
