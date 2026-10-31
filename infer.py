"""Top-level inference function supporting arbitrary image resolution, torch.no_grad(), and FP16."""
from typing import Optional
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from config import DiffSketchConfig
from models import SharedEncoder, SketchDiffusionBranch, InpaintDiffusionBranch
from models.tt_prior import TTGuidance


class DiffSketchJoint(torch.nn.Module):
    """Container bundling SharedEncoder, SketchDiffusionBranch, and InpaintDiffusionBranch."""

    def __init__(self, config: Optional[DiffSketchConfig] = None):
        super().__init__()
        self.cfg = config or DiffSketchConfig()
        self.device = torch.device(
            self.cfg.device if torch.cuda.is_available() else "cpu"
        )
        self.encoder = SharedEncoder(
            in_channels=self.cfg.image_channels + self.cfg.mask_channels,
            base_channels=self.cfg.base_channels,
        ).to(self.device)
        self.sketch_branch = SketchDiffusionBranch(
            feat_channels=self.encoder.out_channels,
            cond_channels=self.cfg.cfe_channels,
            unet_channels=self.cfg.sketch_unet_channels,
            timesteps=self.cfg.diffusion_steps,
            beta_start=self.cfg.beta_start,
            beta_end=self.cfg.beta_end,
            sigmoid_temp=self.cfg.sketch_sigmoid_temp,
        ).to(self.device)
        self.inpaint_branch = InpaintDiffusionBranch(
            latent_channels=self.cfg.latent_channels,
            unet_channels=self.cfg.unet_channels,
            timesteps=self.cfg.diffusion_steps,
            beta_start=self.cfg.beta_start,
            beta_end=self.cfg.beta_end,
            vae_model=self.cfg.pretrained_vae,
            sd_model_id=self.cfg.sd_model_id,
            use_diffusers_unet=self.cfg.use_diffusers_unet,
            allow_fallback_vae=self.cfg.allow_fallback_vae,
            gradient_checkpointing=self.cfg.gradient_checkpointing,
            beta_init=self.cfg.modulator_beta_init,
            mem_efficient=self.cfg.mem_efficient,
            vae_cpu_offload=self.cfg.vae_cpu_offload,
            # Eval/inference always loads a checkpoint that contains the full
            # trained UNet, so build the architecture from config (random init)
            # and let load_checkpoint fill the weights. Avoids the ~3.4GB eager
            # base-UNet read that OOMs/segfaults under host-RAM pressure.
            unet_pretrained=False,
            hf_refiner=getattr(self.cfg, "hf_refiner", False),
            hf_refiner_base=getattr(self.cfg, "hf_refiner_base", 32),
            hf_refiner_blocks=getattr(self.cfg, "hf_refiner_blocks", 4),
        ).to(self.device)

    def load_checkpoint(self, path: str, cfe_path: Optional[str] = None) -> "DiffSketchJoint":
        # Memory: the checkpoint also holds the fp32 Adam state (~2x params).
        # mmap=True (CPU) keeps it lazily on disk, and because we only touch the
        # three model state dicts the optimizer state is never materialised.
        # load_state_dict then copies each tensor CPU -> GPU one at a time, so
        # neither CPU RAM (3-4 GB free here) nor VRAM is blown up.
        # NOTE: mmap + map_location=cuda segfaults on Windows/torch 2.6 -- keep
        # the mmap on CPU and let load_state_dict do the device transfer.
        try:
            ckpt = torch.load(path, map_location="cpu", mmap=True,
                              weights_only=False)
        except (TypeError, RuntimeError, NotImplementedError):
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        # strict=False so a model with newer submodules (e.g. CFE global attention)
        # can resume from an older checkpoint that lacks those keys -- the missing
        # tensors simply keep their (zero-init) initial values.
        for name, module, sd in (
            ("encoder", self.encoder, ckpt["encoder"]),
            ("sketch", self.sketch_branch, ckpt["sketch"]),
            ("inpaint", self.inpaint_branch, ckpt["inpaint"]),
        ):
            missing, unexpected = module.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(
                    f"[load_checkpoint:{name}] strict=False -> "
                    f"missing={len(missing)} unexpected={len(unexpected)}; "
                    f"missing_keys={missing[:8]}"
                )
        # Optional: override the CFE with a standalone pretrained file.
        if cfe_path:
            self.sketch_branch.load_pretrained_cfe(cfe_path)
        return self


@torch.no_grad()
def infer(
    image: Image.Image,
    mask: Image.Image,
    num_steps: int = 50,
    model: Optional[DiffSketchJoint] = None,
    checkpoint: Optional[str] = None,
    config: Optional[DiffSketchConfig] = None,
    tt: Optional["TTGuidance"] = None,
) -> Image.Image:
    """
    Restore a damaged mural image of arbitrary resolution.

    Inputs:
        image:     PIL.Image (RGB)
        mask:      PIL.Image (L, 1/white = damaged region)
        num_steps: reverse diffusion sampling steps (default: 50)
        tt:        optional TTGuidance (low-rank Tensor-Train structure
                   guidance). None = standard sampling (default).
    Output:
        restored:  PIL.Image (RGB) matching original input dimensions
    """
    cfg = config or DiffSketchConfig()
    net = model or DiffSketchJoint(cfg)
    if checkpoint:
        net.load_checkpoint(checkpoint)
    net.eval()

    device = net.device
    orig_w, orig_h = image.size

    orig_img_t = (TF.to_tensor(image.convert("RGB")) * 2.0 - 1.0).unsqueeze(0).to(device)
    orig_mask_t = (
        TF.resize(
            TF.to_tensor(mask.convert("L")).unsqueeze(0),
            [orig_h, orig_w],
            interpolation=TF.InterpolationMode.NEAREST,
        ).to(device)
        > 0.5
    ).float()

    # Align working resolution to a multiple of 64 for multi-scale U-Net compatibility
    work_h = max(64, ((orig_h + 63) // 64) * 64)
    work_w = max(64, ((orig_w + 63) // 64) * 64)

    img_t = TF.resize(orig_img_t, [work_h, work_w], antialias=True)
    mask_t = TF.resize(
        orig_mask_t, [work_h, work_w], interpolation=TF.InterpolationMode.NEAREST
    )
    damaged_t = img_t * (1.0 - mask_t)

    amp_enabled = bool(cfg.amp and device.type == "cuda")
    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
        shared_feats = net.encoder(torch.cat([damaged_t, mask_t], dim=1))
        sketch_prob = net.sketch_branch(shared_feats, mode="infer", infer_steps=num_steps)
        restored_work = net.inpaint_branch.generate(
            img_t, mask_t, sketch_prob, num_steps=num_steps, tt=tt
        )

    if (work_h, work_w) != (orig_h, orig_w):
        restored_work = TF.resize(restored_work, [orig_h, orig_w], antialias=True)
    final_t = restored_work * orig_mask_t + orig_img_t * (1.0 - orig_mask_t)

    restored_01 = ((final_t[0].float().clamp(-1.0, 1.0) + 1.0) * 0.5).cpu()
    return TF.to_pil_image(restored_01)