"""Three-stage training loop with AMP, gradient accumulation, resume, and gradient check."""
import os
from typing import Dict, Optional, Union
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from losses import DiffSketchJointLoss


def pyramid_fill(img_t: torch.Tensor, mask_t: torch.Tensor,
                 ks=(63, 47, 31, 23, 15, 11, 7, 5, 3), reps: int = 6,
                 tail: int = 16) -> torch.Tensor:
    """Coarse-to-fine normalized-convolution fill (harmonic/limit interpolation).

    Mirrors visualize_test.py's anchor fill so that training sees exactly the
    same input the one-step Tweedie estimator sees at inference."""
    w = (1.0 - mask_t)
    x = img_t.clone()
    for k in list(ks) + [3] * tail:
        for _ in range(reps if k in ks else 1):
            wb = F.avg_pool2d(w, k, stride=1, padding=k // 2)
            xb = F.avg_pool2d(x * w, k, stride=1, padding=k // 2)
            est = xb / wb.clamp_min(1e-4)
            x = x * (1.0 - mask_t) + est * mask_t
            w = torch.ones_like(w)
    return img_t * (1.0 - mask_t) + x * mask_t


def _box_sum(x: torch.Tensor, k: int) -> torch.Tensor:
    """Sum over a k x k window (stride 1, zero padding) via an integral image.

    O(H*W) regardless of k. Equivalent to
    ``F.avg_pool2d(x, k, stride=1, padding=k//2, count_include_pad=True) * k * k``.
    """
    p = k // 2
    xp = F.pad(x, (p, p, p, p))
    cs = xp.cumsum(dim=-1).cumsum(dim=-2)
    cs = F.pad(cs, (1, 0, 1, 0))
    return (
        cs[..., k:, k:] - cs[..., :-k, k:] - cs[..., k:, :-k] + cs[..., :-k, :-k]
    )


def pyramid_fill_fast(img_t: torch.Tensor, mask_t: torch.Tensor,
                      ks=(63, 47, 31, 23, 15, 11, 7, 5, 3), reps: int = 6,
                      tail: int = 16) -> torch.Tensor:
    """Numerically equivalent to :func:`pyramid_fill` but ~5-10x cheaper.

    ``avg_pool2d(..., count_include_pad=True)`` divides by k*k, so the k*k factor
    cancels in ``est = avg(x*w) / avg(w)`` -> ``est = box_sum(x*w) / box_sum(w)``.
    Box sums are computed with integral images, so a 63x63 stride-1 filter costs
    the same as a 3x3 one. Accumulation is done in float64 to keep the
    prefix-sum cancellation error below fp32 epsilon.
    """
    w = (1.0 - mask_t)
    x = img_t.clone()
    wb_cache: Dict[int, torch.Tensor] = {}
    w_is_ones = False
    for k in list(ks) + [3] * tail:
        for _ in range(reps if k in ks else 1):
            if w_is_ones:
                wb = wb_cache.get(k)
                if wb is None:
                    wb = _box_sum(torch.ones_like(x, dtype=torch.float64), k)
                    wb_cache[k] = wb
            else:
                wb = _box_sum(w.double(), k)
            xb = _box_sum((x * w).double(), k)
            est = (xb / wb.clamp_min(1e-4)).to(x.dtype)
            x = x * (1.0 - mask_t) + est * mask_t
            if not w_is_ones:
                w = torch.ones_like(w)
                w_is_ones = True
    return img_t * (1.0 - mask_t) + x * mask_t


def compute_masked_psnr_t(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """PSNR outside the damaged mask, as a 0-dim GPU tensor (no host sync)."""
    valid = (1.0 - mask).expand_as(pred)
    mse = (((pred - target) * 0.5).square() * valid).sum() / valid.sum().clamp_min(1.0)
    return -10.0 * torch.log10(mse.clamp_min(1e-10))


def compute_masked_ssim_t(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """SSIM outside the damaged mask, as a 0-dim GPU tensor (no host sync)."""
    x = (pred + 1.0) * 0.5
    y = (target + 1.0) * 0.5
    valid = (1.0 - mask).expand_as(x)
    denom = valid.sum().clamp_min(1.0)

    mx = (x * valid).sum() / denom
    my = (y * valid).sum() / denom
    vx = (((x - mx) ** 2) * valid).sum() / denom
    vy = (((y - my) ** 2) * valid).sum() / denom
    cov = (((x - mx) * (y - my)) * valid).sum() / denom

    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return ((2 * mx * my + c1) * (2 * cov + c2)) / (
        (mx ** 2 + my ** 2 + c1) * (vx + vy + c2)
    )


def compute_masked_psnr(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> float:
    """Compute PSNR outside the damaged mask region (inputs in [-1, 1])."""
    return float(compute_masked_psnr_t(pred, target, mask).item())


def compute_masked_ssim(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> float:
    """Compute SSIM outside the damaged mask region (inputs in [-1, 1])."""
    return float(compute_masked_ssim_t(pred, target, mask).item())



class Trainer:
    """
    Manages the three-stage training pipeline and end-to-end gradient verification:
        - stage1_pretrain_sketch(loader, epochs, resume_from=None)
        - stage2_pretrain_inpaint(loader, epochs, resume_from=None)
        - stage3_joint_finetune(loader, epochs, resume_from=None)
        - check_gradients(batch_or_loader)
    """

    def __init__(
        self,
        encoder: torch.nn.Module,
        sketch_branch: torch.nn.Module,
        inpaint_branch: torch.nn.Module,
        config,
        checkpoint_dir: str = "checkpoints",
        save_weights_only: bool = False,
    ):
        self.cfg = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.encoder = encoder.to(self.device)
        self.sketch_branch = sketch_branch.to(self.device)
        self.inpaint_branch = inpaint_branch.to(self.device)

        # Memory: keep the model weights in bf16 during training. Combined with
        # the existing bf16 autocast, this ~halves the UNet's param + gradient +
        # optimizer-state footprint (~6-7GB saved on the 859M UNet). bf16 has the
        # same exponent range as fp32, so it is safe for SD fine-tuning at LR 1e-4.
        # AdamW then stores its momentum buffers in bf16 too. If training quality
        # regresses, set config.train_bf16 = False to fall back to fp32 weights.
        if getattr(config, "train_bf16", False):
            self.encoder = self.encoder.to(torch.bfloat16)
            self.sketch_branch = self.sketch_branch.to(torch.bfloat16)
            self.inpaint_branch = self.inpaint_branch.to(torch.bfloat16)
            # The SD VAE is numerically fragile under bf16 (collapses PSNR); keep
            # it in fp32. The encode/decode methods also force fp32 defensively.
            if getattr(self.inpaint_branch, "vae", None) is not None:
                self.inpaint_branch.vae = self.inpaint_branch.vae.float()

        # Memory: offload the frozen VAE to CPU between uses (saves ~1GB VRAM).
        # encode()/decode() move it back to the working device on demand.
        if getattr(config, "vae_cpu_offload", False) and self.inpaint_branch.vae is not None:
            self.inpaint_branch.vae.to("cpu")

        self.loss_fn = DiffSketchJointLoss(
            lambda_diff=config.lambda_diff,
            lambda_sketch=config.lambda_sketch,
            lambda_struct=config.lambda_struct,
            lambda_perc=getattr(config, "lambda_perc", 0.2),
            lambda_hf=getattr(config, "lambda_hf", 0.2),
            lambda_scc=getattr(config, "lambda_scc", 0.3),
            lambda_hole=getattr(config, "lambda_hole", 0.0),
            mask_focus_weight=getattr(config, "mask_focus_weight", 1.0),
            lambda_band_low=getattr(config, "lambda_band_low", 0.0),
            lambda_band_high=getattr(config, "lambda_band_high", 0.0),
            band_mode=getattr(config, "band_mode", "gaussian"),
            band_sigma_frac=getattr(config, "band_sigma_frac", 0.01),
            band_radius=getattr(config, "band_radius", 0.25),
            lambda_contrast=getattr(config, "lambda_contrast", 0.0),
            contrast_temperature=getattr(config, "contrast_temperature", 0.07),
            lambda_x0=getattr(config, "lambda_x0", 0.0),
            x0_t_max=getattr(config, "x0_t_max", 200),
        ).to(self.device)
        # pyramid_fill dominates per-step cost (~1-2 s at 256px): its 63x63
        # stride-1 box filters are O(k^2) each and run ~150 times per step.
        # pyramid_fill_fast is numerically equivalent but O(H*W).
        self._fill_fn = (
            pyramid_fill_fast
            if getattr(config, "fast_fill", False)
            else pyramid_fill
        )
        # Optional intra-epoch progress logging (env var, 0 = off). Needed because
        # this loop only reports at epoch end, which is opaque on a slow box.
        self.log_every = int(os.environ.get("LOG_EVERY", "0") or 0)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # When True, _save_checkpoint stores only the three model state dicts
        # (encoder/sketch/inpaint) and drops the bulky Adam/optimizer + scaler
        # state (~8GB for the 859M UNet). This keeps every checkpoint ~3.5GB so
        # it can be eager-loaded on a RAM-constrained box without OOM. Resuming a
        # weights-only checkpoint simply starts a fresh optimizer (same_stage is
        # False because the stage/epoch keys are absent), which is fine for the
        # small-LR finetune stages.
        self.save_weights_only = save_weights_only
        self.amp_enabled = bool(config.amp and self.device.type == "cuda")
        # bf16 autocast: same exponent range as fp32, so the large SD UNet's
        # activations cannot overflow to inf/nan the way fp16 did (which produced
        # the NaN losses in Stage 2). Keeps AMP's memory/speed benefit.
        self.amp_dtype = torch.bfloat16 if self.amp_enabled else torch.float32
        # Training-time timestep sampling window (config-driven). Narrowing it
        # specializes the eps-predictor on the noise regime used by the
        # inference-time one-step Tweedie (MMSE) estimator.
        lo = int(getattr(config, "t_sample_lo", 0) or 0)
        hi = int(getattr(config, "t_sample_hi", 0) or 0) or int(config.diffusion_steps)
        self.t_sample_lo = max(0, min(lo, hi - 1))
        self.t_sample_hi = max(self.t_sample_lo + 1, hi)
        # Train/inference consistency for one-step Tweedie restoration.
        self.fill_consistent = bool(getattr(config, "fill_consistent", False))
        # GradScaler (loss scaling) is only needed for fp16. bf16 shares fp32's
        # exponent range, so it neither needs nor supports loss scaling: PyTorch's
        # _amp_foreach_non_finite_check_and_unscale is NOT implemented for
        # BFloat16, so scaler.unscale_ on bf16 grads raises
        # ("_amp_foreach_non_finite_check_and_unscale_cuda not implemented for
        # 'BFloat16'"). Disable the scaler whenever we train in bf16; autocast is
        # still enabled so activations stay bf16. (This also matches how the
        # original fp32-trained best runs worked -- no loss scaling.)
        self.scaler = torch.amp.GradScaler(
            device=self.device.type,
            enabled=(self.amp_enabled and self.amp_dtype == torch.float16),
        )

    def _save_checkpoint(
        self, stage_name: str, epoch: int, optimizer: torch.optim.Optimizer
    ) -> str:
        # Keep a per-epoch copy (for rollback to the best epoch) AND a generic
        # latest file (convenience for downstream eval/resume).
        state = {
            "stage": stage_name,
            "epoch": epoch,
            "encoder": self.encoder.state_dict(),
            "sketch": self.sketch_branch.state_dict(),
            "inpaint": self.inpaint_branch.state_dict(),
        }
        if not self.save_weights_only:
            state["optimizer"] = optimizer.state_dict()
            state["scaler"] = self.scaler.state_dict()
        latest = os.path.join(self.checkpoint_dir, f"{stage_name}.pt")
        per_epoch = os.path.join(self.checkpoint_dir, f"{stage_name}_ep{epoch}.pt")
        torch.save(state, latest)
        # Copy (not move) so the generic latest still points at the most recent.
        import shutil as _shutil
        _shutil.copyfile(latest, per_epoch)
        return per_epoch

    def _resume_checkpoint(
        self,
        path: Optional[str],
        optimizer: Optional[torch.optim.Optimizer] = None,
        expected_stage: Optional[str] = None,
    ) -> int:
        if not path:
            return 0
        # mmap + map_location='cpu': lazily memory-map the (often 10+ GB)
        # checkpoint and never read it fully into host RAM. mmap+CPU is the
        # stable combo on this build (mmap+GPU segfaulted; eager reads OOM'd
        # under host-RAM pressure). load_state_dict then copies each mmap'd
        # CPU tensor into the model on GPU (one at a time -> tiny peak RAM).
        ckpt = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        # strict=False: a model with newer submodules (e.g. CFE global attention)
        # can resume from an older checkpoint; missing keys keep their init values.
        for name, module, sd in (
            ("encoder", self.encoder, ckpt["encoder"]),
            ("sketch", self.sketch_branch, ckpt["sketch"]),
            ("inpaint", self.inpaint_branch, ckpt["inpaint"]),
        ):
            missing, unexpected = module.load_state_dict(sd, strict=False)
            if missing or unexpected:
                print(
                    f"[resume:{name}] strict=False -> "
                    f"missing={len(missing)} unexpected={len(unexpected)}; "
                    f"missing_keys={missing[:8]}"
                )

        same_stage = (ckpt.get("stage") == expected_stage)
        if optimizer is not None and "optimizer" in ckpt and same_stage:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except ValueError:
                pass
        if "scaler" in ckpt and same_stage:
            self.scaler.load_state_dict(ckpt["scaler"])
        return (int(ckpt.get("epoch", -1)) + 1) if same_stage else 0

    def _estimate_clean_image(
        self, z_t: torch.Tensor, t: torch.Tensor, pred_noise: torch.Tensor,
        mask: Optional[torch.Tensor] = None, sketch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Differentiable clean latent reconstruction + VAE decoding.
        Uses smooth tanh bounding to prevent exploding gradients at large timesteps t.

        `mask` / `sketch` are forwarded to the optional learned HF refiner that
        runs inside decode(); leaving them None keeps the old behaviour exactly.
        """
        ab = self.inpaint_branch.alphas_cumprod[t].view(-1, 1, 1, 1).to(z_t.dtype)
        z0_raw = (z_t - torch.sqrt(1.0 - ab) * pred_noise) / torch.sqrt(ab).clamp_min(1e-3)
        z0_pred = 4.0 * torch.tanh(z0_raw / 4.0)
        return self.inpaint_branch.decode(z0_pred, mask=mask, sketch=sketch)

    def stage1_pretrain_sketch(
        self, loader: DataLoader, epochs: int, resume_from: Optional[str] = None
    ) -> None:
        """Stage 1: Freeze Branch 2, train SharedEncoder + Branch 1 with L_sketch."""
        self.inpaint_branch.requires_grad_(False)
        self.inpaint_branch.eval()
        self.encoder.requires_grad_(True)
        self.sketch_branch.requires_grad_(True)

        optimizer = torch.optim.AdamW(
            [
                {"params": self.encoder.parameters(), "lr": self.cfg.learning_rate},
                {"params": self.sketch_branch.parameters(), "lr": self.cfg.learning_rate},
            ],
            weight_decay=self.cfg.weight_decay,
        )
        start_epoch = self._resume_checkpoint(resume_from, optimizer, expected_stage="stage1")
        aux_w = getattr(self.cfg, "lambda_sketch_noise", 0.1)

        for epoch in range(start_epoch, epochs):
            self.encoder.train()
            self.sketch_branch.train()
            total_loss = 0.0
            tp = fp = fn = 0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(loader):
                damaged = batch["damaged"].to(self.device)
                mask = batch["mask"].to(self.device)
                sketch_gt = batch["sketch"].to(self.device)

                with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                    feats = self.encoder(torch.cat([damaged, mask], dim=1))
                    pred_noise_s, sketch_pred = self.sketch_branch(
                        feats, sketch_gt=sketch_gt, mode="train"
                    )
                    losses = self.loss_fn(
                        pred_noise=pred_noise_s,
                        noise=self.sketch_branch.last_target_noise,
                        sketch_pred=sketch_pred,
                        sketch_gt=sketch_gt,
                        image_pred=None,
                    )
                    loss = (losses["sketch"] + aux_w * losses["diff"]) / self.cfg.accum_steps

                if torch.isfinite(loss):
                    self.scaler.scale(loss).backward()
                if (step + 1) % self.cfg.accum_steps == 0 or (step + 1) == len(loader):
                    self.scaler.unscale_(optimizer)
                    if self.cfg.clip_grad:
                        torch.nn.utils.clip_grad_norm_(
                            [p for g in optimizer.param_groups for p in g["params"]],
                            1.0, foreach=False)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                total_loss += float(losses["sketch"].item())
                with torch.no_grad():
                    p_bin = sketch_pred.detach() > 0.5
                    g_bin = sketch_gt > 0.5
                    tp += int((p_bin & g_bin).sum().item())
                    fp += int((p_bin & ~g_bin).sum().item())
                    fn += int((~p_bin & g_bin).sum().item())

            f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
            avg_loss = total_loss / max(len(loader), 1)
            print(f"[Stage 1] Epoch {epoch + 1}/{epochs} | L_sketch={avg_loss:.4f} | F1={f1:.4f}")
            self._save_checkpoint("stage1", epoch, optimizer)

    def stage2_pretrain_inpaint(
        self, loader: DataLoader, epochs: int, resume_from: Optional[str] = None
    ) -> None:
        """Stage 2: Freeze Branch 1 (no_grad infer), train Branch 2 with L_diff + L_struct."""
        self.encoder.requires_grad_(False)
        self.sketch_branch.requires_grad_(False)
        self.encoder.eval()
        self.sketch_branch.eval()
        self.inpaint_branch.requires_grad_(True)
        # NOTE: the zero-init AdditiveSketchResidual heads are trained TOGETHER
        # with the UNet + injector here (no freeze). Zero-init keeps them at
        # identity at step 0, so co-training is safe (ControlNet-style) and lets
        # the UNet and sketch-modulation co-adapt -- this is what produced the
        # best XATTN4 result. Freezing them in Stage 2 (tried in XATTN7) starves
        # the additive heads of training and degrades both structural fidelity
        # and undamaged-region preservation.
        if self.inpaint_branch.vae is not None:
            self.inpaint_branch.vae.requires_grad_(False)

        trainable_params = [p for p in self.inpaint_branch.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params, lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay
        )
        start_epoch = self._resume_checkpoint(resume_from, optimizer, expected_stage="stage2")

        for epoch in range(start_epoch, epochs):
            self.inpaint_branch.train()
            if self.inpaint_branch.vae is not None:
                self.inpaint_branch.vae.eval()
            total_loss = 0.0
            perc_list, hf_list = [], []
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(loader):
                image = batch["image"].to(self.device)
                mask = batch["mask"].to(self.device)
                damaged = batch["damaged"].to(self.device)
                sketch_gt = batch["sketch"].to(self.device)

                # Wrap the frozen encoder/sketch inference in autocast so that,
                # under train_bf16, the bf16-cast encoder receives bf16 inputs
                # (otherwise: dtype mismatch fp32 input vs bf16 weight). stage3
                # already does this because its encoder call is inside the main
                # autocast block; stage2/2b/2c compute features in a separate
                # no_grad block and must opt in explicitly.
                with torch.no_grad(), torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled,
                ):
                    feats = self.encoder(torch.cat([damaged, mask], dim=1))
                    sketch_cond = self.sketch_branch(feats, mode="infer")

                z0 = self.inpaint_branch.encode(image)
                z_masked = self.inpaint_branch.encode(damaged)
                t = torch.randint(
                    self.t_sample_lo, self.t_sample_hi, (image.shape[0],), device=self.device
                ).long()
                noise = torch.randn_like(z0)
                z_t = self.inpaint_branch.q_sample(z0, t, noise)
                snr_weights = self.inpaint_branch.alphas_cumprod[t]

                with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                    pred_noise = self.inpaint_branch(
                        z_t, t, mask, sketch_cond, masked_image_latent=z_masked
                    )
                    pred_img = self._estimate_clean_image(z_t, t, pred_noise, mask=mask, sketch=sketch_cond)
                    losses = self.loss_fn(
                        pred_noise=pred_noise,
                        noise=noise,
                        sketch_pred=sketch_cond,
                        sketch_gt=sketch_gt,
                        image_pred=pred_img,
                        image_gt=image,
                        struct_sample_weights=snr_weights,
                        mask=mask,
                    )
                    loss = (
                        self.cfg.lambda_diff * losses["diff"]
                        + self.cfg.lambda_struct * losses["struct"]
                        + self.cfg.lambda_perc * losses["perc"]
                        + self.cfg.lambda_hf * losses["hf"]
                        + self.cfg.lambda_scc * losses["scc"]
                    ) / self.cfg.accum_steps

                if torch.isfinite(loss):
                    self.scaler.scale(loss).backward()
                if (step + 1) % self.cfg.accum_steps == 0 or (step + 1) == len(loader):
                    self.scaler.unscale_(optimizer)
                    if self.cfg.clip_grad:
                        torch.nn.utils.clip_grad_norm_(
                            [p for g in optimizer.param_groups for p in g["params"]],
                            1.0, foreach=False)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                total_loss += float(loss.item()) * self.cfg.accum_steps
                perc_list.append(float(losses["perc"].item()))
                hf_list.append(float(losses["hf"].item()))

            avg_loss = total_loss / max(len(loader), 1)
            avg_perc = sum(perc_list) / max(len(perc_list), 1)
            avg_hf = sum(hf_list) / max(len(hf_list), 1)
            print(f"[Stage 2] Epoch {epoch + 1}/{epochs} | Loss={avg_loss:.4f} "
                  f"| perc={avg_perc:.4f} | hf={avg_hf:.4f}")
            self._save_checkpoint("stage2", epoch, optimizer)

    def stage2b_adapter_injector(
        self, loader: DataLoader, epochs: int, resume_from: Optional[str] = None
    ) -> None:
        """
        Stage 2b: Adapter-only training of the CrossAttnInjector.

        Freezes the encoder, the sketch branch, and the ENTIRE inpaint branch
        (UNet / sketch_encoder / modulators / VAE) EXCEPT the CrossAttnInjector.
        Only the injection pathway learns to map the (fixed) sketch features to
        useful 768-dim cross-attention context.

        This is the stable recipe that avoids the Stage-2 divergence observed
        when the injector was co-trained jointly with the 859M UNet: with the
        UNet weights held fixed, the injector cannot drag them into a bad basin.
        """
        if self.inpaint_branch.cross_inject is None:
            raise RuntimeError(
                "cross_inject is None; Stage 2b requires use_diffusers_unet=True."
            )
        self.encoder.requires_grad_(False)
        self.sketch_branch.requires_grad_(False)
        self.inpaint_branch.requires_grad_(False)
        self.inpaint_branch.cross_inject.requires_grad_(True)
        self.encoder.eval()
        self.sketch_branch.eval()
        self.inpaint_branch.eval()  # frozen UNet in eval mode; cross_inject still trains
        if self.inpaint_branch.vae is not None:
            self.inpaint_branch.vae.requires_grad_(False)

        injector_params = [p for p in self.inpaint_branch.cross_inject.parameters()
                          if p.requires_grad]
        optimizer = torch.optim.AdamW(
            injector_params, lr=self.cfg.injector_learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        start_epoch = self._resume_checkpoint(resume_from, optimizer, expected_stage="stage2b")

        for epoch in range(start_epoch, epochs):
            self.inpaint_branch.cross_inject.train()  # only the adapter is live
            total_loss = 0.0
            perc_list, hf_list = [], []
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(loader):
                image = batch["image"].to(self.device)
                mask = batch["mask"].to(self.device)
                damaged = batch["damaged"].to(self.device)
                sketch_gt = batch["sketch"].to(self.device)

                # Wrap the frozen encoder/sketch inference in autocast so that,
                # under train_bf16, the bf16-cast encoder receives bf16 inputs
                # (otherwise: dtype mismatch fp32 input vs bf16 weight). stage3
                # already does this because its encoder call is inside the main
                # autocast block; stage2/2b/2c compute features in a separate
                # no_grad block and must opt in explicitly.
                with torch.no_grad(), torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled,
                ):
                    feats = self.encoder(torch.cat([damaged, mask], dim=1))
                    sketch_cond = self.sketch_branch(feats, mode="infer")

                z0 = self.inpaint_branch.encode(image)
                z_masked = self.inpaint_branch.encode(damaged)
                t = torch.randint(
                    self.t_sample_lo, self.t_sample_hi, (image.shape[0],), device=self.device
                ).long()
                noise = torch.randn_like(z0)
                z_t = self.inpaint_branch.q_sample(z0, t, noise)
                snr_weights = self.inpaint_branch.alphas_cumprod[t]

                with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                    pred_noise = self.inpaint_branch(
                        z_t, t, mask, sketch_cond, masked_image_latent=z_masked
                    )
                    pred_img = self._estimate_clean_image(z_t, t, pred_noise, mask=mask, sketch=sketch_cond)
                    losses = self.loss_fn(
                        pred_noise=pred_noise,
                        noise=noise,
                        sketch_pred=sketch_cond,
                        sketch_gt=sketch_gt,
                        image_pred=pred_img,
                        image_gt=image,
                        struct_sample_weights=snr_weights,
                        mask=mask,
                    )
                    loss = (
                        self.cfg.lambda_diff * losses["diff"]
                        + self.cfg.lambda_struct * losses["struct"]
                        + self.cfg.lambda_perc * losses["perc"]
                        + self.cfg.lambda_hf * losses["hf"]
                        + self.cfg.lambda_scc * losses["scc"]
                    ) / self.cfg.accum_steps

                if torch.isfinite(loss):
                    self.scaler.scale(loss).backward()
                if (step + 1) % self.cfg.accum_steps == 0 or (step + 1) == len(loader):
                    self.scaler.unscale_(optimizer)
                    if self.cfg.clip_grad:
                        torch.nn.utils.clip_grad_norm_(injector_params, 1.0, foreach=False)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                total_loss += float(loss.item()) * self.cfg.accum_steps
                perc_list.append(float(losses["perc"].item()))
                hf_list.append(float(losses["hf"].item()))

            avg_loss = total_loss / max(len(loader), 1)
            avg_perc = sum(perc_list) / max(len(perc_list), 1)
            avg_hf = sum(hf_list) / max(len(hf_list), 1)
            print(f"[Stage 2b] Epoch {epoch + 1}/{epochs} | Loss={avg_loss:.4f} "
                  f"| perc={avg_perc:.4f} | hf={avg_hf:.4f}")

            if avg_loss > 1.0:
                print(f"[Stage 2b] LOSS-EXPLOSION GUARD: avg_loss={avg_loss:.4f} >> "
                      f"healthy ~0.2; stopping early to avoid a broken checkpoint.")
                break
            self._save_checkpoint("stage2b", epoch, optimizer)

    def stage2c_additive_sketch(
        self, loader: DataLoader, epochs: int, resume_from: Optional[str] = None
    ) -> None:
        """
        Stage 2c: Adapter-only training of the AdditiveSketchResidual modules.

        Freezes the encoder, sketch branch, and the ENTIRE inpaint branch
        (UNet / sketch_encoder / modulators / cross_inject / VAE) EXCEPT the
        zero-init AdditiveSketchResidual heads. Only these new heads learn to
        inject sketch structure additively into each UNet block output.

        Because they are zero-init and the rest is frozen, the model at init is
        bit-identical to the converged baseline it resumes from, so this cannot
        diverge and can only improve fidelity (the diagnosed weak point).
        """
        if self.inpaint_branch.down_add is None:
            raise RuntimeError(
                "AdditiveSketchResidual modules are None; Stage 2c requires "
                "use_diffusers_unet=True."
            )
        self.encoder.requires_grad_(False)
        self.sketch_branch.requires_grad_(False)
        self.inpaint_branch.requires_grad_(False)
        # Unfreeze only the additive heads.
        for m in (self.inpaint_branch.down_add, self.inpaint_branch.mid_add,
                  self.inpaint_branch.up_add):
            for p in m.parameters():
                p.requires_grad_(True)
        self.encoder.eval()
        self.sketch_branch.eval()
        self.inpaint_branch.eval()  # frozen UNet; additive heads still train
        if self.inpaint_branch.vae is not None:
            self.inpaint_branch.vae.requires_grad_(False)

        additive_params = [p for p in
                           list(self.inpaint_branch.down_add.parameters())
                           + list(self.inpaint_branch.mid_add.parameters())
                           + list(self.inpaint_branch.up_add.parameters())
                           if p.requires_grad]
        optimizer = torch.optim.AdamW(
            additive_params, lr=self.cfg.injector_learning_rate,
            weight_decay=self.cfg.weight_decay,
        )
        start_epoch = self._resume_checkpoint(resume_from, optimizer, expected_stage="stage2c")

        for epoch in range(start_epoch, epochs):
            for m in (self.inpaint_branch.down_add, self.inpaint_branch.mid_add,
                      self.inpaint_branch.up_add):
                m.train()
            total_loss = 0.0
            perc_list, hf_list = [], []
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(loader):
                image = batch["image"].to(self.device)
                mask = batch["mask"].to(self.device)
                damaged = batch["damaged"].to(self.device)
                sketch_gt = batch["sketch"].to(self.device)

                # Wrap the frozen encoder/sketch inference in autocast so that,
                # under train_bf16, the bf16-cast encoder receives bf16 inputs
                # (otherwise: dtype mismatch fp32 input vs bf16 weight). stage3
                # already does this because its encoder call is inside the main
                # autocast block; stage2/2b/2c compute features in a separate
                # no_grad block and must opt in explicitly.
                with torch.no_grad(), torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.amp_enabled,
                ):
                    feats = self.encoder(torch.cat([damaged, mask], dim=1))
                    sketch_cond = self.sketch_branch(feats, mode="infer")

                z0 = self.inpaint_branch.encode(image)
                z_masked = self.inpaint_branch.encode(damaged)
                t = torch.randint(
                    self.t_sample_lo, self.t_sample_hi, (image.shape[0],), device=self.device
                ).long()
                noise = torch.randn_like(z0)
                z_t = self.inpaint_branch.q_sample(z0, t, noise)
                snr_weights = self.inpaint_branch.alphas_cumprod[t]

                with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                    pred_noise = self.inpaint_branch(
                        z_t, t, mask, sketch_cond, masked_image_latent=z_masked
                    )
                    pred_img = self._estimate_clean_image(z_t, t, pred_noise, mask=mask, sketch=sketch_cond)
                    losses = self.loss_fn(
                        pred_noise=pred_noise,
                        noise=noise,
                        sketch_pred=sketch_cond,
                        sketch_gt=sketch_gt,
                        image_pred=pred_img,
                        image_gt=image,
                        struct_sample_weights=snr_weights,
                        mask=mask,
                    )
                    loss = (
                        self.cfg.lambda_diff * losses["diff"]
                        + self.cfg.lambda_struct * losses["struct"]
                        + self.cfg.lambda_perc * losses["perc"]
                        + self.cfg.lambda_hf * losses["hf"]
                        + self.cfg.lambda_scc * losses["scc"]
                    ) / self.cfg.accum_steps

                if torch.isfinite(loss):
                    self.scaler.scale(loss).backward()
                if (step + 1) % self.cfg.accum_steps == 0 or (step + 1) == len(loader):
                    self.scaler.unscale_(optimizer)
                    if self.cfg.clip_grad:
                        torch.nn.utils.clip_grad_norm_(additive_params, 1.0, foreach=False)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                total_loss += float(loss.item()) * self.cfg.accum_steps
                perc_list.append(float(losses["perc"].item()))
                hf_list.append(float(losses["hf"].item()))

            avg_loss = total_loss / max(len(loader), 1)
            avg_perc = sum(perc_list) / max(len(perc_list), 1)
            avg_hf = sum(hf_list) / max(len(hf_list), 1)
            print(f"[Stage 2c] Epoch {epoch + 1}/{epochs} | Loss={avg_loss:.4f} "
                  f"| perc={avg_perc:.4f} | hf={avg_hf:.4f}")

            if avg_loss > 1.0:
                print(f"[Stage 2c] LOSS-EXPLOSION GUARD: avg_loss={avg_loss:.4f} >> "
                      f"healthy ~0.2; stopping early to avoid a broken checkpoint.")
                break
            self._save_checkpoint("stage2c", epoch, optimizer)

    def stage3_joint_finetune(
        self, loader: DataLoader, epochs: int, resume_from: Optional[str] = None
    ) -> None:
        """
        Stage 3: Unfreeze all branches for end-to-end joint optimization.
        Branch 1 LR = 0.1 * main LR. Evaluates PSNR & SSIM outside mask per epoch.
        """
        self.encoder.requires_grad_(True)
        self.sketch_branch.requires_grad_(True)
        self.inpaint_branch.requires_grad_(True)
        if self.inpaint_branch.vae is not None:
            self.inpaint_branch.vae.requires_grad_(False)

        # Split the inpaint branch into the CrossAttnInjector adapter (its own,
        # typically smaller LR), the zero-init AdditiveSketchResidual heads (their
        # own small LR), and the rest (UNet / sketch_encoder / modulators).
        inj_ids = (
            {id(p) for p in self.inpaint_branch.cross_inject.parameters()}
            if self.inpaint_branch.cross_inject is not None
            else set()
        )
        add_ids = set()
        for m in (self.inpaint_branch.down_add, self.inpaint_branch.mid_add,
                  self.inpaint_branch.up_add):
            if m is not None:
                add_ids.update(id(p) for p in m.parameters())
        injector_params = [
            p for p in self.inpaint_branch.parameters()
            if p.requires_grad and id(p) in inj_ids
        ]
        additive_params = [
            p for p in self.inpaint_branch.parameters()
            if p.requires_grad and id(p) in add_ids
        ]
        inpaint_base_params = [
            p for p in self.inpaint_branch.parameters()
            if p.requires_grad and id(p) not in inj_ids and id(p) not in add_ids
        ]
        param_groups = [
            {
                "params": list(self.encoder.parameters())
                + list(self.sketch_branch.parameters()),
                "lr": self.cfg.learning_rate * self.cfg.sketch_learning_rate_ratio,
            },
            {"params": inpaint_base_params, "lr": self.cfg.learning_rate},
        ]
        if injector_params:
            param_groups.append(
                {"params": injector_params, "lr": self.cfg.injector_learning_rate}
            )
        if additive_params:
            param_groups.append(
                {"params": additive_params, "lr": self.cfg.additive_learning_rate}
            )
        optimizer = torch.optim.AdamW(param_groups, weight_decay=self.cfg.weight_decay)
        start_epoch = self._resume_checkpoint(resume_from, optimizer, expected_stage="stage3")
        aux_w = getattr(self.cfg, "lambda_sketch_noise", 0.1)
        print(
            f"[Stage 3] damaged-region focus: mask_focus_weight="
            f"{getattr(self.cfg, 'mask_focus_weight', 1.0):.2f} "
            f"lambda_hole={getattr(self.cfg, 'lambda_hole', 0.0):.3f} "
            f"(1.0 / 0.0 = original whole-image objective)"
        )

        for epoch in range(start_epoch, epochs):
            self.encoder.train()
            self.sketch_branch.train()
            self.inpaint_branch.train()
            if self.inpaint_branch.vae is not None:
                self.inpaint_branch.vae.eval()

            total_loss = torch.zeros((), device=self.device)
            psnr_acc = torch.zeros((), device=self.device)
            ssim_acc = torch.zeros((), device=self.device)
            perc_acc = torch.zeros((), device=self.device)
            hf_acc = torch.zeros((), device=self.device)
            hole_acc = torch.zeros((), device=self.device)
            blo_acc = torch.zeros((), device=self.device)
            bhi_acc = torch.zeros((), device=self.device)
            cls_acc = torch.zeros((), device=self.device)
            scc_acc = torch.zeros((), device=self.device)
            con_acc = torch.zeros((), device=self.device)
            x0_acc = torch.zeros((), device=self.device)
            n_logged = 0
            optimizer.zero_grad(set_to_none=True)

            for step, batch in enumerate(loader):
                image = batch["image"].to(self.device)
                mask = batch["mask"].to(self.device)
                damaged = batch["damaged"].to(self.device)
                sketch_gt = batch["sketch"].to(self.device)
                negative = batch.get("negative")
                if negative is not None:
                    negative = negative.to(self.device)

                z0 = self.inpaint_branch.encode(image)
                if self.fill_consistent:
                    # Inference feeds the one-step estimator a noised latent of the
                    # classical pyramid fill. Train on exactly that input, but keep
                    # the eps target that Tweedie-inverts to the CLEAN image, so the
                    # model learns to remove the fill residual in one step.
                    with torch.no_grad():
                        z_base = self.inpaint_branch.encode(self._fill_fn(image, mask))
                    z_masked = z_base
                else:
                    z_base = z0
                    z_masked = self.inpaint_branch.encode(damaged)
                t = torch.randint(
                    self.t_sample_lo, self.t_sample_hi, (image.shape[0],), device=self.device
                ).long()
                noise = torch.randn_like(z0)
                z_t = self.inpaint_branch.q_sample(z_base, t, noise)
                if self.fill_consistent:
                    ab = self.inpaint_branch.alphas_cumprod[t].to(z_t.dtype).view(-1, 1, 1, 1)
                    noise = (z_t - ab.sqrt() * z0) / (1.0 - ab).sqrt().clamp_min(1e-4)
                snr_weights = self.inpaint_branch.alphas_cumprod[t]

                with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_enabled):
                    feats = self.encoder(torch.cat([damaged, mask], dim=1))
                    pred_noise_s, sketch_pred = self.sketch_branch(
                        feats, sketch_gt=sketch_gt, mode="train"
                    )
                    pred_noise = self.inpaint_branch(
                        z_t, t, mask, sketch_pred, masked_image_latent=z_masked
                    )
                    pred_img = self._estimate_clean_image(z_t, t, pred_noise, mask=mask, sketch=sketch_pred)
                    # Direct latent-x0 supervision: reconstruct the Tweedie clean
                    # latent from the eps prediction and compare it to the GT
                    # latent z0 inside the hole. This is the exact quantity that
                    # determines PSNR at inference. Only computed when lambda_x0>0
                    # so the default path stays bit-identical.
                    x0_hat = None
                    if self.loss_fn.lambda_x0 > 0:
                        ab = self.inpaint_branch.alphas_cumprod[t].view(-1, 1, 1, 1).to(z_t.dtype)
                        x0_hat = (z_t - (1.0 - ab).sqrt() * pred_noise) / ab.sqrt().clamp_min(1e-3)
                    losses = self.loss_fn(
                        pred_noise=pred_noise,
                        noise=noise,
                        sketch_pred=sketch_pred,
                        sketch_gt=sketch_gt,
                        image_pred=pred_img,
                        image_gt=image,
                        struct_sample_weights=snr_weights,
                        mask=mask,
                        negative=negative,
                        x0_hat=x0_hat,
                        z_gt=z0 if x0_hat is not None else None,
                        timestep=t if x0_hat is not None else None,
                    )
                    # Retain auxiliary DDPM noise supervision on sketch MLE to prevent drift
                    sketch_noise_mse = F.mse_loss(
                        pred_noise_s.float(), self.sketch_branch.last_target_noise.float()
                    )
                    loss = (losses["total"] + aux_w * sketch_noise_mse) / self.cfg.accum_steps

                # `if torch.isfinite(loss)` costs a host<->device sync every step
                # (it drains the CUDA queue and kills cross-step overlap). Check
                # every 25 steps instead: a blow-up is still caught long before it
                # could corrupt a saved checkpoint (epoch-end guard + per-epoch save).
                if (step % 25 == 0) or (step + 1) == len(loader):
                    if not torch.isfinite(loss):
                        print(f"[Stage 3] non-finite loss at step {step}; aborting epoch.")
                        break
                self.scaler.scale(loss).backward()
                if (step + 1) % self.cfg.accum_steps == 0 or (step + 1) == len(loader):
                    self.scaler.unscale_(optimizer)
                    if self.cfg.clip_grad:
                        torch.nn.utils.clip_grad_norm_(
                            [p for g in optimizer.param_groups for p in g["params"]],
                            1.0, foreach=False)
                    self.scaler.step(optimizer)
                    self.scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                # Accumulate on the GPU and read back ONCE per epoch: calling
                # .item() per step forced ~7 host<->device syncs per iteration,
                # which serialised the pipeline (GPU util dropped to ~40% and an
                # epoch took ~1.5 h instead of ~15 min) without changing any value.
                with torch.no_grad():
                    total_loss += losses["total"].detach()
                    perc_acc += losses["perc"].detach()
                    hf_acc += losses["hf"].detach()
                    hole_acc += losses["hole"].detach()
                    blo_acc += losses["band_low"].detach()
                    bhi_acc += losses["band_high"].detach()
                    cls_acc += losses["diff"].detach()
                    scc_acc += losses["scc"].detach()
                    con_acc += losses["contrast"].detach()
                    x0_acc += losses["x0"].detach()
                    psnr_acc += compute_masked_psnr_t(pred_img.detach(), image, mask)
                    ssim_acc += compute_masked_ssim_t(pred_img.detach(), image, mask)
                n_logged += 1
            if self.log_every and (step + 1) % self.log_every == 0:
                print(
                    f"[Stage 3] step {step + 1}/{len(loader)} "
                    f"| loss={float(losses['total'].item()):.4f} "
                    f"| diff={float(cls_acc.item()) / n_logged:.4f} "
                    f"| perc={float(perc_acc.item()) / n_logged:.4f} "
                    f"| hf={float(hf_acc.item()) / n_logged:.4f} "
                    f"| scc={float(scc_acc.item()) / n_logged:.4f} "
                    f"| con={float(con_acc.item()) / n_logged:.4f} "
                    f"| x0={float(x0_acc.item()) / n_logged:.4f} "
                    f"| bandL={float(blo_acc.item()) / n_logged:.4f} "
                    f"| bandH={float(bhi_acc.item()) / n_logged:.4f} "
                    f"| psnr_known={float(psnr_acc.item()) / n_logged:.2f}dB",
                    flush=True,
                )

            avg_loss = float(total_loss.item()) / max(n_logged, 1)
            # NOTE: with fill_consistent the eps target carries the fill residual on
            # top of the DDPM noise, so its variance (and hence the healthy MSE) is
            # ~1.05x-1.3x the plain-noise value, not ~0.2. Raise the ceiling.
            guard = 4.0 if self.fill_consistent else 1.0
            if avg_loss > guard:
                print(f"[Stage 3] LOSS-EXPLOSION GUARD: avg_loss={avg_loss:.4f} >> "
                      f"healthy ~0.2 (guard={guard}); stopping Stage 3 early to avoid "
                      f"overwriting good weights with a diverged checkpoint.")
                break
            denom = max(n_logged, 1)
            avg_psnr = float(psnr_acc.item()) / denom
            avg_ssim = float(ssim_acc.item()) / denom
            avg_perc = float(perc_acc.item()) / denom
            avg_hf = float(hf_acc.item()) / denom
            avg_hole = float(hole_acc.item()) / denom
            avg_blo = float(blo_acc.item()) / denom
            avg_bhi = float(bhi_acc.item()) / denom
            print(
                f"[Stage 3] Epoch {epoch + 1}/{epochs} | Loss={avg_loss:.4f} "
                f"| PSNR_out={avg_psnr:.2f}dB | SSIM_out={avg_ssim:.4f} "
                f"| perc={avg_perc:.4f} | hf={avg_hf:.4f} | hole={avg_hole:.4f} "
                f"| bandL={avg_blo:.4f} | bandH={avg_bhi:.4f}"
            )
            self._save_checkpoint("stage3", epoch, optimizer)

    def check_gradients(
        self, batch_or_loader: Union[Dict[str, torch.Tensor], DataLoader]
    ) -> Dict[str, float]:
        """
        Two-step gradient verification:
          Step A: Backward ONLY (L_diff + L_struct) (with L_sketch=0) to strictly prove
                  that gradients from Branch 2 flow through ConditionModulator into Branch 1.
          Step B: Backward full joint loss (L_diff + L_sketch + L_struct).
        """
        batch = (
            next(iter(batch_or_loader))
            if isinstance(batch_or_loader, DataLoader)
            else batch_or_loader
        )

        self.encoder.requires_grad_(True)
        self.sketch_branch.requires_grad_(True)
        self.inpaint_branch.requires_grad_(True)
        if self.inpaint_branch.vae is not None:
            self.inpaint_branch.vae.requires_grad_(False)

        self.encoder.train()
        self.sketch_branch.train()
        self.inpaint_branch.train()

        image = batch["image"].to(self.device)
        mask = batch["mask"].to(self.device)
        damaged = batch["damaged"].to(self.device)
        sketch_gt = batch["sketch"].to(self.device)

        # --- Step A: Cross-branch gradient flow test (L_diff + L_struct only) ---
        self.encoder.zero_grad(set_to_none=True)
        self.sketch_branch.zero_grad(set_to_none=True)
        self.inpaint_branch.zero_grad(set_to_none=True)

        feats = self.encoder(torch.cat([damaged, mask], dim=1))
        _, sketch_pred = self.sketch_branch(feats, sketch_gt=sketch_gt, mode="train")

        z0 = self.inpaint_branch.encode(image)
        z_masked = self.inpaint_branch.encode(damaged)
        t = torch.randint(
            self.t_sample_lo, self.t_sample_hi, (image.shape[0],), device=self.device
        ).long()
        noise = torch.randn_like(z0)
        z_t = self.inpaint_branch.q_sample(z0, t, noise)

        pred_noise = self.inpaint_branch(
            z_t, t, mask, sketch_pred, masked_image_latent=z_masked
        )
        pred_img = self._estimate_clean_image(z_t, t, pred_noise, mask=mask, sketch=sketch_pred)

        cross_losses = self.loss_fn(
            pred_noise=pred_noise,
            noise=noise,
            sketch_pred=sketch_pred,
            sketch_gt=None,
            image_pred=pred_img,
        )
        cross_losses["total"].backward(retain_graph=True)

        cross_sketch_grad = sum(
            p.grad.norm().item()
            for p in self.sketch_branch.parameters()
            if p.requires_grad and p.grad is not None
        )
        print(
            f"[GradCheck - CrossBranch Only (L_diff + L_struct -> SketchBranch)] "
            f"grad_norm_sum={cross_sketch_grad:.6e}"
        )
        if cross_sketch_grad <= 0.0:
            raise RuntimeError(
                "Gradient check failed: L_diff/L_struct did not backpropagate to SketchDiffusionBranch!"
            )

        # --- Step B: Full Joint Loss backward ---
        self.encoder.zero_grad(set_to_none=True)
        self.sketch_branch.zero_grad(set_to_none=True)
        self.inpaint_branch.zero_grad(set_to_none=True)

        full_losses = self.loss_fn(
            pred_noise=pred_noise,
            noise=noise,
            sketch_pred=sketch_pred,
            sketch_gt=sketch_gt,
            image_pred=pred_img,
        )
        full_losses["total"].backward()

        stats: Dict[str, float] = {"cross_branch_sketch_grad": float(cross_sketch_grad)}
        for name, module in [
            ("shared_encoder", self.encoder),
            ("sketch_branch", self.sketch_branch),
            ("inpaint_branch", self.inpaint_branch),
        ]:
            norms = [
                p.grad.norm().item()
                for p in module.parameters()
                if p.requires_grad and p.grad is not None
            ]
            grad_sum = float(sum(norms))
            stats[name] = grad_sum
            print(
                f"[GradCheck - Full Joint] {name}: grad_norm_sum={grad_sum:.6e} "
                f"({len(norms)} tensors with grad)"
            )
        return stats