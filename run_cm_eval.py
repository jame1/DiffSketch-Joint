"""Wrapper to run vis_comparemodel_test.py against an arbitrary checkpoint.

vis_comparemodel_test.py hardcodes CKPT (no --checkpoint flag), so we override
the module-level CKPT global before calling main(). Usage:

    python run_cm_eval.py <ckpt_path> <out_dir> [extra vis_comparemodel_test args]

Example:
    python run_cm_eval.py C:/wb_diffsketch/checkpoints_fb2/stage3.pt vis_cm_fb2 --limit 60
"""
import sys
from pathlib import Path

ckpt = sys.argv[1]
out_dir = sys.argv[2]
extra = sys.argv[3:]

# vis_comparemodel_test reads CKPT at import; we override it before main().
sys.argv = ["vis_comparemodel_test.py", "--out-dir", out_dir] + extra
import vis_comparemodel_test  # noqa: E402  (must come after sys.argv is set)

vis_comparemodel_test.CKPT = Path(ckpt)
print(f"[run_cm_eval] CKPT -> {ckpt}  out-dir -> {out_dir}  extra -> {extra}", flush=True)
vis_comparemodel_test.main()
