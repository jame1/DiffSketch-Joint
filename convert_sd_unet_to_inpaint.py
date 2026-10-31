"""
Convert a Stable Diffusion checkpoint into the 9-channel inpainting UNet that
models/inpaint_branch.py expects, and save it under
weights/sd-unet-inpainting/unet/ so the pipeline runs fully OFFLINE
(no Hugging Face download and no YAML config fetch over the network).

Supported sources:
  1. A single CompVis/auto1111 checkpoint file  (*.ckpt or *.safetensors)
     e.g. E:/sd-v1-5-inpainting.ckpt  (already 9-channel inpainting UNet)
          or a base sd-v1-4.ckpt        (4-channel -> expanded to 9)
  2. A diffusers pipeline directory  (has unet/config.json)
  3. A HuggingFace repo id           (only if network is available)

The CompVis -> diffusers conversion uses diffusers' OWN mapping machinery
(convert_ldm_unet_checkpoint + create_unet_diffusers_config_from_ldm) with a
locally-built SD1.5 original config, so the produced UNet layout exactly matches
the one diffusers' from_single_file would build (9-channel inpainting UNet =>
lossless weight transfer, 0 missing / 0 unexpected keys).

Usage:
  python convert_sd_unet_to_inpaint.py --src E:/sd-v1-5-inpainting.ckpt
  python convert_sd_unet_to_inpaint.py --src /f/models/sd-v1-4.ckpt
  python convert_sd_unet_to_inpaint.py --src stable-diffusion-v1-5/stable-diffusion-v1-5
"""
import argparse
import sys
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel


def _import_converters():
    """diffusers keeps these helpers in an internal module (with a fallback)."""
    try:
        from diffusers.loaders.single_file_utils import (
            convert_ldm_unet_checkpoint,
            create_unet_diffusers_config_from_ldm,
        )
    except Exception:
        from diffusers.pipelines.stable_diffusion.convert_from_ckpt import (
            convert_ldm_unet_checkpoint,
            create_unet_diffusers_config_from_ldm,
        )
    return convert_ldm_unet_checkpoint, create_unet_diffusers_config_from_ldm


# CompVis SD1.5 original config (the v1-inference.yaml content as a dict).
# use_linear_in_transformer=False => diffusers uses Conv1d proj_in/proj_out,
# which is what the SD1.5 SpatialTransformer stores.  attention_resolutions and
# channel_mult fully define the 4-level UNet layout.
SD15_ORIGINAL_CONFIG = {
    "model": {
        "params": {
            "unet_config": {
                "params": {
                    "image_size": 64,
                    "in_channels": 4,
                    "out_channels": 4,
                    "model_channels": 320,
                    "attention_resolutions": [4, 2, 1],
                    "num_res_blocks": 2,
                    "channel_mult": [1, 2, 4, 4],
                    "transformer_depth": 1,
                    "num_heads": 8,
                    "context_dim": 768,
                    "use_linear_in_transformer": False,
                }
            },
            "first_stage_config": {
                "params": {"ddconfig": {"ch_mult": [1, 2, 4, 4]}}
            },
        }
    }
}


def _load_compvis_unet(src: str, convert_fn, create_cfg_fn) -> UNet2DConditionModel:
    print(f"[convert] loading CompVis checkpoint {src} (offline) ...")
    ckpt = torch.load(src, map_location="cpu", mmap=True, weights_only=False)
    # Some checkpoints wrap everything under a "state_dict" key.
    while isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    # Detect how many input channels the conv_in actually has (4 for base SD,
    # 9 for an inpainting checkpoint) so we build the matching UNet skeleton.
    conv_w = ckpt["model.diffusion_model.input_blocks.0.0.weight"]
    in_channels = int(conv_w.shape[1])
    print(f"[convert] detected CompVis UNet in_channels = {in_channels}")

    unet_config = create_cfg_fn(SD15_ORIGINAL_CONFIG, ckpt, num_in_channels=in_channels)
    unet = UNet2DConditionModel(**unet_config)
    converted = convert_fn(ckpt, unet_config, extract_ema=False)
    # strict=True is a hard correctness check: every LDM weight must map 1:1.
    unet.load_state_dict(converted, strict=True)
    print(f"[convert] lossless load OK (in_channels={unet.config.in_channels})")
    return unet


def load_source_unet(src: str, convert_fn, create_cfg_fn) -> UNet2DConditionModel:
    p = Path(src)
    # Case 1: single CompVis / auto1111 checkpoint file
    if p.is_file() and p.suffix.lower() in (".ckpt", ".safetensors"):
        return _load_compvis_unet(str(p), convert_fn, create_cfg_fn)
    # Case 2: diffusers directory
    if p.is_dir():
        print(f"[convert] loading diffusers UNet from {src} (subfolder=unet) ...")
        return UNet2DConditionModel.from_pretrained(str(p), subfolder="unet", local_files_only=True)
    # Case 3: HF repo id (needs network)
    print(f"[convert] loading UNet from HuggingFace repo {src} ...")
    return UNet2DConditionModel.from_pretrained(src, subfolder="unet")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="local .ckpt/.safetensors / dir / HF repo id")
    ap.add_argument("--dst", default="weights/sd-unet-inpainting/unet",
                    help="output dir for the 9-channel inpainting UNet")
    args = ap.parse_args()

    convert_fn, create_cfg_fn = _import_converters()
    unet = load_source_unet(args.src, convert_fn, create_cfg_fn)

    # in-repo expansion helper: no-op if already 9 channels.
    sys_path = str(Path(__file__).resolve().parents[0])
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from models.inpaint_branch import expand_conv_in_to_9channels
    unet = expand_conv_in_to_9channels(unet)
    print(f"[convert] final UNet in_channels = {unet.config.in_channels}")

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(str(dst))
    ok_bin = (dst / "diffusion_pytorch_model.bin").exists()
    ok_safetensors = (dst / "diffusion_pytorch_model.safetensors").exists()
    print(f"[convert] saved 9-channel inpainting UNet -> {dst}")
    print(f"[convert] files: config.json={ (dst / 'config.json').exists() }, "
          f"bin={ok_bin}, safetensors={ok_safetensors}")


if __name__ == "__main__":
    main()
