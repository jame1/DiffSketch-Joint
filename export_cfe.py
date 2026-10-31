"""
Export the trained CFE (Condition Feature Extractor) from a training checkpoint
into a standalone, reusable "pretrained CFE" file.

The CFE lives inside the sketch branch's state dict (keys prefixed `cfe.*`) in
every checkpoint. This script pulls those tensors out and saves them under a
top-level `{"cfe": ...}` key so `SketchDiffusionBranch.load_pretrained_cfe()`
(or the `cfe_pretrained_path` config) can load them to bootstrap a new
from-scratch Stage-1 run, or override the CFE at inference.

Usage:
    python export_cfe.py                       # default: checkpoints/stage1.pt -> weights/cfe_pretrained.pt
    python export_cfe.py --ckpt checkpoints/stage3.pt --out weights/cfe_pretrained.pt
"""
import argparse
import os

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Export trained CFE to a standalone file")
    parser.add_argument(
        "--ckpt", default="checkpoints/stage1.pt",
        help="Source checkpoint (stage1 is the canonical sketch-pretrained CFE; "
             "stage3 is the joint-finetuned one).",
    )
    parser.add_argument(
        "--out", default="weights/cfe_pretrained.pt",
        help="Output standalone CFE file.",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        raise SystemExit(f"Source checkpoint not found: {args.ckpt}")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sketch_sd = ckpt.get("sketch", ckpt)
    cfe_sd = {k[len("cfe."):]: v for k, v in sketch_sd.items() if k.startswith("cfe.")}

    if not cfe_sd:
        raise SystemExit("No 'cfe.*' keys found in the checkpoint's sketch state dict.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(
        {
            "cfe": cfe_sd,
            "source_ckpt": os.path.abspath(args.ckpt),
            "source_stage": ckpt.get("stage"),
            "epoch": ckpt.get("epoch"),
        },
        args.out,
    )

    # Report a sanity stat so the user can see the weights are real (trained), not random.
    sample = next(iter(cfe_sd.values()))
    print(
        f"Exported {len(cfe_sd)} CFE tensors -> {args.out}\n"
        f"  source: {args.ckpt} (stage={ckpt.get('stage')}, epoch={ckpt.get('epoch')})\n"
        f"  param count: {sum(v.numel() for v in cfe_sd.values()):,}\n"
        f"  sample tensor {list(cfe_sd)[0]}: "
        f"mean={float(sample.float().mean()):+.4f} std={float(sample.float().std()):.4f}"
    )


if __name__ == "__main__":
    main()
