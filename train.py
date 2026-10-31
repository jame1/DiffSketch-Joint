"""Command-line entry point for 3-stage training and gradient verification."""
import argparse

import torch
from torch.utils.data import DataLoader
from config import DiffSketchConfig
from data import MuralDataset
from models import SharedEncoder, SketchDiffusionBranch, InpaintDiffusionBranch
from trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DiffSketch-Joint")
    parser.add_argument(
        "--data", default=r"E:\DunhuangMural\DiffSketch-Joint\data",
        help="Dataset root directory (images/, sketches/, masks/). Defaults to the "
             "project-local dataset; override with --data <path> for other datasets.",
    )
    parser.add_argument("--stage", choices=["1", "2", "2b", "2c", "3", "check"], default="1")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=None, help="Override cfg.learning_rate")
    parser.add_argument("--injector-lr", type=float, default=None,
                        help="Override cfg.injector_learning_rate (used by Stage 2b / Stage 3 injector group)")
    parser.add_argument("--additive-lr", type=float, default=None,
                        help="Override cfg.additive_learning_rate (used by Stage 2c / Stage 3 additive group)")
    parser.add_argument("--lambda-scc", type=float, default=None,
                        help="Override cfg.lambda_scc (sketch-cyclic-consistency weight)")
    parser.add_argument("--lambda-hole", type=float, default=None,
                        help="Weight of the direct in-hole L1 on the decoded estimate "
                             "(decoded vs GT inside the mask only). The most direct lever "
                             "on in-hole PSNR. 0.0 (default) = disabled")
    parser.add_argument("--mask-focus", type=float, default=None,
                        help="Multiply per-pixel loss inside the damaged mask by this factor "
                             "for L_diff / L_perc / L_hf (the hole is ~17%% of pixels, so the "
                             "uniform mean is dominated by the already-correct region). "
                             "1.0 (default) = original whole-image objective")
    parser.add_argument("--fast-fill", action="store_true",
                        help="Use the integral-image pyramid fill: numerically equivalent "
                             "to the original but O(H*W) per box filter instead of O(k^2), "
                             "cutting per-step time several-fold (pyramid_fill dominates "
                             "Stage-3 step time at ~1-2 s/step).")

    parser.add_argument("--fdp-enabled", action="store_true", default=None,
                        help="Enable the FDP-lite frequency sharpening module")
    parser.add_argument("--no-fdp", action="store_true",
                        help="Disable the FDP-lite frequency sharpening module")
    parser.add_argument("--fdp-grid", type=int, default=None,
                        help="Override cfg.fdp_grid (learnable frequency-gain grid resolution)")
    parser.add_argument("--resume", default=None, help="Checkpoint path to resume from (loads encoder+sketch+inpaint)")
    parser.add_argument(
        "--init-from", default=None,
        help="Warm-start ONLY encoder+sketch from this checkpoint, leaving the "
             "inpaint branch freshly initialized. Use when switching the inpaint "
             "branch architecture (e.g. to the SD UNet2DConditionModel) so the old "
             "inpaint weights are not loaded into an incompatible model.",
    )
    parser.add_argument("--cfe-ckpt", default=None, help="Pretrained CFE weights path (default: config.cfe_pretrained_path)")
    parser.add_argument("--out-dir", default="checkpoints", help="Directory for stage checkpoints of this run")
    parser.add_argument("--save-weights-only", action="store_true",
                        help="Save only model weights (drop optimizer/scaler state) -> ~3.5GB "
                             "checkpoints that load without a host-RAM spike on constrained boxes")
    parser.add_argument("--no-train-bf16", action="store_true",
                        help="keep master weights in fp32 (proper AMP): with train_bf16=True "
                             "the module params themselves are cast to bf16, whose ~2^-8 "
                             "relative resolution rounds away Adam updates of size ~lr, so "
                             "the model barely trains. fp32 master weights + bf16 autocast "
                             "gives correct updates at moderate memory cost.")
    parser.add_argument("--t-lo", type=int, default=None,
                        help="min training timestep (specialize the eps-predictor on the "
                             "noise regime used by the one-step Tweedie estimator; "
                             "default 0 = full range)")
    parser.add_argument("--t-hi", type=int, default=None,
                        help="max training timestep (default: diffusion_steps)")
    parser.add_argument("--fill-consistent", action="store_true",
                        help="Build z_t from the pyramid-filled image (as at inference) while "
                             "keeping the clean image as the Tweedie target, so training matches "
                             "the one-step restoration used at eval time")
    parser.add_argument("--crop-size", type=int, default=None,
                        help="Override the training crop/resolution (must be a multiple of 8). "
                             "Default 256. Set 512 to train the diffusion UNet at the VAE's "
                             "native latent resolution (64x64 latent) so fine mural texture "
                             "inside the damaged region is representable (the 256-latent model "
                             "cannot, which caps in-hole PSNR well below the VAE ceiling).")
    parser.add_argument("--unet-from-config", action="store_true",
                        help="Build the SD UNet from config.json only (random init) instead of "
                             "loading the ~3.4GB base weights via from_pretrained. Use when the "
                             "UNet weights will be filled by --resume (every recovery stage), to "
                             "avoid the 3.4GB eager read that OOMs/segfaults under host-RAM pressure")
    parser.add_argument("--diffusers-unet", action="store_true", help="Use SD UNet2DConditionModel")
    parser.add_argument("--no-grad-ckpt", action="store_true",
                        help="Disable gradient checkpointing in the inpaint UNet. Avoids a "
                             "torch.utils.checkpoint.pack_hook deadlock observed on Windows/WDDM "
                             "under bf16 autocast (stage-3 hangs before the first step). Costs "
                             "more activation VRAM but is required on affected builds.")
    parser.add_argument("--no-amp", action="store_true",
                        help="Disable AMP autocast entirely (train in full fp32). Works around a "
                             "bf16-autocast backward deadlock on Windows/WDDM where loss.backward() "
                             "stalls inside the autograd engine. Costs ~2x VRAM but is required on "
                             "affected builds; use with --no-train-bf16.")
    parser.add_argument("--no-clip-grad", action="store_true",
                        help="Skip gradient clipping (torch.nn.utils.clip_grad_norm_). Used to "
                             "isolate a Windows/WDDM clip_grad_norm_ deadlock that stalls training.")
    parser.add_argument("--fallback-vae", action="store_true", help="Use fallback VAE for debugging")
    # ---- LF/HF frequency decomposition ----
    parser.add_argument("--lambda-band-low", type=float, default=None,
                        help="Weight of the LOW-frequency restoration target: globally "
                             "smoothed decoded estimate vs GT (tone / coarse composition). "
                             "Normalised by the low band's own contrast. 0 = off")
    parser.add_argument("--lambda-band-high", type=float, default=None,
                        help="Weight of the HIGH-frequency restoration target: the "
                             "residual above the low-frequency band (lines / texture / "
                             "edges), vs GT. This is the band whose absence is visible as "
                             "'plausible but detail-free'. 0 = off")
    parser.add_argument("--band-mode", default=None, choices=["gaussian", "fft"],
                        help="Low-pass used for the LF/HF split: gaussian (smooth, no "
                             "ringing, default) or fft (hard circular cutoff)")
    parser.add_argument("--band-sigma", type=float, default=None,
                        help="Gaussian cutoff as a fraction of the image height "
                             "(default 0.01 -> 2.6px @256, 5.1px @512)")
    parser.add_argument("--band-radius", type=float, default=None,
                        help="FFT cutoff as a fraction of Nyquist (only for --band-mode fft)")
    parser.add_argument("--hf-refiner", action="store_true",
                        help="Enable the learned image-space HIGH-FREQUENCY generator "
                             "applied after the VAE decoder (the 8x-downsampled latent "
                             "cannot represent pixel-scale line detail). Zero-init -> "
                             "identity, safe to enable over a converged checkpoint")
    # ---- Bilateral contrastive loss (Dunhuang semantic ritual constraints) ----
    parser.add_argument("--lambda-contrast", type=float, default=None,
                        help="Weight of the bilateral contrastive loss (VGG feature-space "
                             "pull-toward-GT / push-away-from-wrong-restoration). 0 = off")
    parser.add_argument("--contrast-temperature", type=float, default=None,
                        help="Temperature for the contrastive InfoNCE (default 0.07)")
    parser.add_argument("--negative-dir", type=str, default="",
                        help="Directory of pre-generated negative restorations "
                             "(gen_negatives.py). If empty, the contrastive loss is disabled "
                             "regardless of --lambda-contrast.")
    parser.add_argument("--lambda-x0", type=float, default=None,
                        help="Weight of the direct latent-x0 regression: hole-focused "
                             "MSE between the Tweedie clean latent x0_hat and the GT latent "
                             "z_gt, SNR-gated to t <= --x0-t-max. Directly optimizes the "
                             "quantity that determines PSNR at inference. 0 = off")
    parser.add_argument("--x0-t-max", type=int, default=None,
                        help="Only supervise x0 at timesteps <= this (default 200, "
                             "matching the inference one-step Tweedie timestep)")
    args = parser.parse_args()

    # CLI --cfe-ckpt overrides the config default; fall back to config default
    # (weights/cfe_pretrained.pt) when not explicitly provided.
    cfg = DiffSketchConfig(
        cfe_pretrained_path=args.cfe_ckpt or "weights/cfe_pretrained.pt",
        use_diffusers_unet=args.diffusers_unet,
        allow_fallback_vae=args.fallback_vae,
    )
    cfg = DiffSketchConfig(
        cfe_pretrained_path=args.cfe_ckpt or "weights/cfe_pretrained.pt",
        use_diffusers_unet=args.diffusers_unet,
        allow_fallback_vae=args.fallback_vae,
    )
    if args.lr is not None:
        cfg.learning_rate = args.lr
    if args.injector_lr is not None:
        cfg.injector_learning_rate = args.injector_lr
    if args.additive_lr is not None:
        cfg.additive_learning_rate = args.additive_lr
    if args.lambda_scc is not None:
        cfg.lambda_scc = args.lambda_scc
    if args.lambda_hole is not None:
        cfg.lambda_hole = args.lambda_hole
    if args.mask_focus is not None:
        cfg.mask_focus_weight = args.mask_focus
    if args.fast_fill:
        cfg.fast_fill = True
    if args.fdp_grid is not None:
        cfg.fdp_grid = args.fdp_grid
    if args.fdp_enabled is True:
        cfg.fdp_enabled = True
    if args.no_fdp:
        cfg.fdp_enabled = False
    if args.no_train_bf16:
        cfg.train_bf16 = False
    if args.t_lo is not None:
        cfg.t_sample_lo = args.t_lo
    if args.t_hi is not None:
        cfg.t_sample_hi = args.t_hi
    if args.fill_consistent:
        cfg.fill_consistent = True
    if args.crop_size is not None:
        cfg.crop_size = args.crop_size
    if args.lambda_band_low is not None:
        cfg.lambda_band_low = args.lambda_band_low
    if args.lambda_band_high is not None:
        cfg.lambda_band_high = args.lambda_band_high
    if args.band_mode is not None:
        cfg.band_mode = args.band_mode
    if args.band_sigma is not None:
        cfg.band_sigma_frac = args.band_sigma
    if args.band_radius is not None:
        cfg.band_radius = args.band_radius
    if args.hf_refiner:
        cfg.hf_refiner = True
    if args.no_grad_ckpt:
        cfg.gradient_checkpointing = False
    if args.no_amp:
        cfg.amp = False
    if args.no_clip_grad:
        cfg.clip_grad = False
    if args.lambda_contrast is not None:
        cfg.lambda_contrast = args.lambda_contrast
    if args.contrast_temperature is not None:
        cfg.contrast_temperature = args.contrast_temperature
    if args.lambda_x0 is not None:
        cfg.lambda_x0 = args.lambda_x0
    if args.x0_t_max is not None:
        cfg.x0_t_max = args.x0_t_max
    # Contrastive loss needs pre-generated negatives; without them it's a no-op.
    negative_dir = args.negative_dir
    if negative_dir and cfg.lambda_contrast <= 0:
        print("[warn] --negative-dir given but --lambda-contrast is 0/None; "
              "contrastive loss disabled.")
    if cfg.lambda_contrast > 0 and not negative_dir:
        print("[warn] --lambda-contrast > 0 but no --negative-dir; falling back to "
              "damaged-image negatives (near-degenerate). Provide a --negative-dir.")
    dataset = MuralDataset(
        args.data,
        crop_size=cfg.crop_size,
        random_flip=cfg.random_flip,
        negative_dir=negative_dir,
    )
    # num_workers=0: on Windows, multiprocessing workers use a shared-memory
    # "event" that fails ("Couldn't open shared event") under host-RAM pressure
    # from concurrent jobs. Single-process data loading is reliable and is not
    # the bottleneck (the 859M frozen-UNet forward dominates per-step time).
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    encoder = SharedEncoder(in_channels=4, base_channels=cfg.base_channels)
    sketch_branch = SketchDiffusionBranch(
        feat_channels=encoder.out_channels,
        cond_channels=cfg.cfe_channels,
        unet_channels=cfg.sketch_unet_channels,
        timesteps=cfg.diffusion_steps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        sigmoid_temp=cfg.sketch_sigmoid_temp,
    )
    sketch_branch.load_pretrained_cfe(cfg.cfe_pretrained_path)

    # Warm-start encoder + sketch branch from a checkpoint WITHOUT touching the
    # (possibly architecture-changed) inpaint branch.
    if args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        # strict=False: if the CFE gained new submodules (global attention) since
        # the warm-start checkpoint was saved, those keys are absent here and keep
        # their (zero-init) values -- the rest of the warm-start still applies.
        missing_e, _ = encoder.load_state_dict(ck["encoder"], strict=False)
        missing_s, _ = sketch_branch.load_state_dict(ck["sketch"], strict=False)
        print(
            f"[init-from] loaded encoder+sketch from '{args.init_from}' "
            f"(stage={ck.get('stage')}, epoch={ck.get('epoch')}); "
            f"encoder missing_keys={len(missing_e)}, sketch missing_keys={len(missing_s)}; "
            "inpaint branch left freshly initialized."
        )

    inpaint_branch = InpaintDiffusionBranch(
        latent_channels=cfg.latent_channels,
        unet_channels=cfg.unet_channels,
        timesteps=cfg.diffusion_steps,
        beta_start=cfg.beta_start,
        beta_end=cfg.beta_end,
        vae_model=cfg.pretrained_vae,
        sd_model_id=cfg.sd_model_id,
        use_diffusers_unet=cfg.use_diffusers_unet,
        allow_fallback_vae=cfg.allow_fallback_vae,
        gradient_checkpointing=cfg.gradient_checkpointing,
        beta_init=cfg.modulator_beta_init,
        fdp_enabled=cfg.fdp_enabled,
        fdp_grid=cfg.fdp_grid,
        mem_efficient=cfg.mem_efficient,
        vae_cpu_offload=cfg.vae_cpu_offload,
        unet_pretrained=not args.unet_from_config,
        hf_refiner=cfg.hf_refiner,
        hf_refiner_base=cfg.hf_refiner_base,
        hf_refiner_blocks=cfg.hf_refiner_blocks,
        band_mode=cfg.band_mode,
        band_sigma_frac=cfg.band_sigma_frac,
        band_radius=cfg.band_radius,
    )

    trainer = Trainer(
        encoder, sketch_branch, inpaint_branch, cfg,
        checkpoint_dir=args.out_dir, save_weights_only=args.save_weights_only,
    )
    if args.stage == "check":
        trainer.check_gradients(loader)
    elif args.stage == "1":
        trainer.stage1_pretrain_sketch(loader, args.epochs, resume_from=args.resume)
    elif args.stage == "2":
        trainer.stage2_pretrain_inpaint(loader, args.epochs, resume_from=args.resume)
    elif args.stage == "2b":
        trainer.stage2b_adapter_injector(loader, args.epochs, resume_from=args.resume)
    elif args.stage == "2c":
        trainer.stage2c_additive_sketch(loader, args.epochs, resume_from=args.resume)
    elif args.stage == "3":
        trainer.stage3_joint_finetune(loader, args.epochs, resume_from=args.resume)


if __name__ == "__main__":
    main()
