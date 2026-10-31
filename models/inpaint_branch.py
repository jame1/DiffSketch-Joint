"""Branch 2: Latent Diffusion Inpainting Branch with SketchEncoder & ConditionModulator."""
from typing import Tuple, List, Optional
import os
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from .blocks import ConvBlock, ConditionModulator, ModulatedUNet
from .freqband import BandSplit, HighFreqRefiner


def resolve_vae_path(vae_model: str) -> str:
    """
    Prefer a local copy of the VAE shipped with the repo (weights/sd-vae-ft-mse)
    so the pipeline runs fully offline. Falls back to the original HF id only
    when no local copy is found (then it will attempt a download).

    This avoids repeated Hugging Face HEAD requests / SSL retries on machines
    that are offline or behind a restricted network.
    """
    # Already an explicit path or a local directory -> use as-is.
    if vae_model.startswith((".", "/", "\\")) or os.path.isdir(vae_model):
        return vae_model
    local = Path(__file__).resolve().parents[1] / "weights" / "sd-vae-ft-mse"
    return str(local) if local.is_dir() else vae_model


def resolve_unet_path(sd_model_id: str) -> str:
    """
    Prefer a local copy of the SD inpainting UNet shipped with the repo
    (weights/sd-unet-inpainting, mirroring the HF `unet/` subfolder layout:
        weights/sd-unet-inpainting/unet/config.json
        weights/sd-unet-inpainting/unet/diffusion_pytorch_model.safetensors
    ) so training runs fully offline. Falls back to the original HF id only
    when no local copy is found (then from_pretrained attempts a download).
    """
    if sd_model_id.startswith((".", "/", "\\")) or os.path.isdir(sd_model_id):
        return sd_model_id
    local = Path(__file__).resolve().parents[1] / "weights" / "sd-unet-inpainting"
    return str(local) if local.is_dir() else sd_model_id


class CrossAttnInjector(nn.Module):
    """
    Replaces the dead all-zero cross-attention token of the SD UNet with real
    sketch-derived context, so the UNet's 768-dim cross-attention actually
    conditions on the mural line-drawing instead of being a broadcast constant.

    It takes the inpaint branch's own sketch-encoder feature map (the SAME
    sketch feature the per-layer ConditionModulators use, at latent resolution),
    pools it to a fixed low-res token grid, and projects 320 -> 256 -> 768.

    The FINAL 1x1 conv is zero-initialized (ControlNet zero-conv pattern): at
    initialization the output is exactly zero, i.e. encoder_hidden_states is the
    all-zero tensor the model already trains with, so loading this module on top
    of an existing checkpoint is a no-op until it is actually trained. This keeps
    the (already converged) 859M UNet stable on the first Stage-2 steps.

    Note: we deliberately reuse the inpaint branch's internal sketch encoder
    rather than threading the sketch branch's CFE across modules -- it is
    available in both `forward` (training) and `generate` (inference) with no
    plumbing, and it trains jointly with the UNet during Stage 2, which is
    exactly when the injection must take effect.
    """

    def __init__(self, in_ch: int = 320, mid_ch: int = 256, out_ch: int = 768, tok: int = 32):
        super().__init__()
        self.tok = tok
        # Fixed low-res token grid -> bounded cross-attention cost regardless of
        # input resolution (32x32 = 1024 tokens; the QK^T matmul is ~16MB).
        self.pool = nn.AdaptiveAvgPool2d((tok, tok))
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=1),
        )
        # Zero-init the gate (final conv) -> identity (all-zero output) at init.
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, sketch_feat: torch.Tensor) -> torch.Tensor:
        # sketch_feat: [B, in_ch, H/8, W/8] (latent-resolution sketch feature)
        x = self.pool(sketch_feat)                       # [B, in_ch, tok, tok]
        x = self.net(x)                                  # [B, out_ch, tok, tok]
        return x.flatten(2).transpose(1, 2).contiguous()  # [B, tok*tok, out_ch]


class AdditiveSketchResidual(nn.Module):
    """
    ControlNet-lite ADDITIVE structural injection (the missing piece).

    The existing ConditionModulator only does multiplicative gating
        F_out = F_in * (1 + beta * sigmoid(proj(F_s)))
    which can RESCALE features but cannot ADD new structural content -- so the
    UNet still has to *imagine* the missing lines instead of being told where
    they are. That is the diagnosed "sketch under-utilized" bottleneck and a
    main reason the model loses fidelity to a trivial blur-fill.

    This module adds a zero-initialized trainable residual derived from the
    sketch feature directly into the UNet block output:
        F_out = mod(F_in, F_s) + zero_conv(F_s)
    so the sketch can INJECT structure (line/figure layout) at every resolution.
    The FINAL conv is zero-initialized (ControlNet zero-conv): at init the
    residual is exactly zero, i.e. the module is the identity and the model
    behaves exactly like the converged baseline until it is actually trained.
    """

    def __init__(self, in_ch: int, out_ch: int, mid_ch: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
        )
        # Zero-init the gate (final conv) -> identity (all-zero output) at init.
        nn.init.zeros_(self.net[2].weight)
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x: torch.Tensor, f_s: torch.Tensor) -> torch.Tensor:
        if f_s.shape[-2:] != x.shape[-2:]:
            f_s = F.interpolate(f_s, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x + self.net(f_s)


class FrequencySharpen(nn.Module):
    """
    Learnable frequency-domain high-frequency enhancer (FDP-lite), inspired by
    DiffuMural's FDP module (arXiv:2504.09513). Applied to the decoded clean
    image to fight the *blur* that plagues latent-diffusion mural restoration
    (our diagnosed low SSIM_dmg / "loses to blur-fill" failure).

    Mechanism (zero-init SAFE -- exact identity at initialization):
        x01  = (x + 1) / 2                      # map [-1,1] -> [0,1]
        X    = fft2(x01)
        G    = exp(log_gain)                    # learnable smooth gain, init 0 -> G=1
        X_f  = X * G
        x_f  = ifft2(X_f).real * 2 - 1          # back to [-1,1]
        out  = x_f
    At init G = 1 -> x_f = x (exact identity), so wrapping a converged checkpoint
    with this module is a no-op until it is actually trained. The gain is a small
    learnable grid (fdp_grid x fdp_grid) bilinearly upsampled to the image size,
    which keeps it smooth (enhances broad frequency bands, not per-pixel noise)
    and robust to the small 1564-sample dataset. The gradient w.r.t. G is
    non-zero even at G=1 (d x_f / dG = ifft(X) != 0), so the module learns to
    sharpen under the hf / perceptual / SCC losses without a separate warm-up.

    Computed in float32 for FFT numerical safety regardless of autocast dtype.
    """

    def __init__(self, grid: int = 16):
        super().__init__()
        self.grid = int(grid)
        # log-gain grid (smooth, upsample to image size at forward). init 0 -> G=1 -> identity.
        self.log_gain = nn.Parameter(torch.zeros(1, 1, self.grid, self.grid))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for FFT numerical safety regardless of autocast dtype.
        x_dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape
        x01 = (x + 1.0) * 0.5
        # Upsample the smooth learnable gain to the image resolution.
        # Clamp log_gain to [-3, 3] so G in [e^-3, e^3] ~ [0.05, 20]: keeps the
        # module expressive (can still sharpen or suppress broad bands) but
        # prevents exp() overflow -> inf -> NaN that would poison a whole batch.
        G = torch.exp(torch.clamp(self.log_gain, -3.0, 3.0))
        if (H, W) != (self.grid, self.grid):
            G = F.interpolate(G, size=(H, W), mode="bilinear", align_corners=False)
        G = G.expand(B, C, H, W)
        X = torch.fft.fft2(x01, norm="ortho")
        X_f = X * G.to(X.dtype)
        x_f = torch.fft.ifft2(X_f, norm="ortho").real
        x_f = x_f * 2.0 - 1.0
        # Numerical-safety net: collapse any residual NaN/Inf (e.g. from a
        # degenerate FFT) to a safe finite value, then clamp to [-1, 1].
        x_f = torch.nan_to_num(x_f, nan=0.0, posinf=1.0, neginf=-1.0)
        x_f = x_f.clamp(-1.0, 1.0)
        return x_f.to(x_dtype)


def expand_conv_in_to_9channels(unet: "UNet2DConditionModel") -> "UNet2DConditionModel":
    """
    Adapt a *base* Stable Diffusion UNet (4-channel conv_in) into the 9-channel
    inpainting layout our branch feeds it:
        [noisy_latent(4), mask(1), masked_image_latent(4)]  -> 9 channels.
    The first 4 conv_in weights (the original RGB/latent projection) are copied
    verbatim; the extra 5 channels (mask + masked-image) are zero-initialised so
    the network initially ignores them and behaves like base SD, then Stage 2/3
    fine-tunes them on inpainting. This is exactly how RunwayML's
    stable-diffusion-inpainting was derived from a base SD checkpoint.
    No-op (returns unchanged) if the UNet already has 9 input channels.
    """
    cfg = unet.config
    if int(getattr(cfg, "in_channels", 4)) == 9:
        return unet
    old_conv = unet.conv_in  # Conv2d(4, *, 3, padding=1)
    new_conv = torch.nn.Conv2d(
        9, old_conv.out_channels, old_conv.kernel_size,
        stride=old_conv.stride, padding=old_conv.padding, bias=(old_conv.bias is not None),
    )
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, :4, :, :] = old_conv.weight  # keep pretrained latent projection
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    unet.conv_in = new_conv
    cfg.in_channels = 9
    return unet


class SketchEncoder(nn.Module):
    """
    Lightweight convolutional network that encodes a full-resolution single-channel
    sketch probability map S [B, 1, H, W] into a multi-scale feature list F_s
    aligned with the spatial resolutions and channel dimensions of the LDM U-Net.

    Input:  sketch [B, 1, H, W] in [0, 1]
    Output: List[Tensor] -> [F_s^0 [B, C_0, H/8, W/8],
                             F_s^1 [B, C_1, H/16, W/16], ...]
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: Tuple[int, ...] = (64, 128, 256),
        vae_scale_factor: int = 8,
    ):
        super().__init__()
        self.out_channels = out_channels

        num_stem_downs = int(round(torch.log2(torch.tensor(float(vae_scale_factor))).item()))
        stem_layers: List[nn.Module] = []
        c_curr = in_channels
        c_stem = max(16, out_channels[0] // 2)
        for i in range(num_stem_downs):
            c_next = out_channels[0] if i == num_stem_downs - 1 else c_stem
            stem_layers.append(ConvBlock(c_curr, c_next, stride=2))
            c_curr = c_next
        self.stem = nn.Sequential(*stem_layers)

        self.stages = nn.ModuleList()
        for idx, c_out in enumerate(out_channels):
            if idx == 0:
                self.stages.append(ConvBlock(out_channels[0], c_out, stride=1))
            else:
                self.stages.append(ConvBlock(out_channels[idx - 1], c_out, stride=2))

    def forward(self, sketch: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(sketch)
        feats: List[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class InpaintDiffusionBranch(nn.Module):
    """
    LDM-based inpainting branch with frozen pretrained SD VAE and per-layer
    residual gating modulation:
        F_out = F_in * (1 + beta * sigmoid(F_s))

    Interfaces:
        - encode(image [-1,1]) -> latent [B, 4, H/8, W/8]
        - decode(latent) -> image [-1,1] [B, 3, H, W]
        - forward(latent, timestep, mask, sketch, masked_image_latent=None) -> pred_noise
        - generate(image, mask, sketch, num_steps=50) -> restored_image [-1,1]
    """

    def __init__(
        self,
        latent_channels: int = 4,
        unet_channels: Tuple[int, ...] = (64, 128, 256),
        timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        vae_model: str = "stabilityai/sd-vae-ft-mse",
        sd_model_id: str = "runwayml/stable-diffusion-inpainting",
        use_diffusers_unet: bool = False,
        allow_fallback_vae: bool = False,
        gradient_checkpointing: bool = True,
        beta_init: float = 0.5,
        fdp_enabled: bool = True,
        fdp_grid: int = 16,
        mem_efficient: bool = True,
        vae_cpu_offload: bool = False,
        unet_pretrained: bool = True,
        # ---- Frequency-domain (LF/HF) decomposition add-ons ----
        hf_refiner: bool = False,      # learned image-space HF generator in decode()
        hf_refiner_base: int = 32,
        hf_refiner_blocks: int = 4,
        band_mode: str = "gaussian",   # how the loss splits LF/HF
        band_sigma_frac: float = 0.01,  # Gaussian cutoff as a fraction of H
        band_radius: float = 0.25,      # FFT cutoff as a fraction of Nyquist
    ):
        super().__init__()
        self.latent_channels = latent_channels
        self.use_diffusers_unet = use_diffusers_unet
        self.mem_efficient = bool(mem_efficient)
        self.vae_cpu_offload = bool(vae_cpu_offload)
        self._current_sketch_feats: List[torch.Tensor] = []

        # 1. Load frozen pretrained Stable Diffusion VAE
        self.vae: Optional[AutoencoderKL] = None
        try:
            self.vae = AutoencoderKL.from_pretrained(resolve_vae_path(vae_model))
            self.vae.requires_grad_(False)
            self.vae.eval()
            if gradient_checkpointing and hasattr(self.vae, "enable_gradient_checkpointing"):
                self.vae.enable_gradient_checkpointing()
            # Memory: slice the VAE decode/encode so it processes the latent in
            # tiles instead of all-at-once -> much lower peak VRAM (matters most
            # at 512+ eval resolution). Pure runtime optimization, no accuracy loss.
            if self.mem_efficient and hasattr(self.vae, "enable_slicing"):
                self.vae.enable_slicing()
            self.scale_factor = float(getattr(self.vae.config, "scaling_factor", 0.18215))
        except Exception as exc:
            if not allow_fallback_vae:
                raise RuntimeError(
                    f"Failed to load pretrained VAE '{vae_model}': {exc}. "
                    "Download the weights locally or set allow_fallback_vae=True for debugging."
                ) from exc
            self.scale_factor = 0.18215
            self.fallback_encoder = nn.Sequential(
                ConvBlock(3, 32, stride=2),
                ConvBlock(32, 64, stride=2),
                nn.Conv2d(64, latent_channels, kernel_size=3, stride=2, padding=1),
            )
            self.fallback_decoder = nn.Sequential(
                nn.ConvTranspose2d(latent_channels, 64, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
                nn.Tanh(),
            )
            self.fallback_encoder.requires_grad_(False)
            self.fallback_decoder.requires_grad_(False)

        # 2. Unified DDPM Noise Scheduler
        self.scheduler = DDPMScheduler(
            num_train_timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="linear",
            clip_sample=False,
        )
        self.timesteps = timesteps
        self.register_buffer("alphas_cumprod", self.scheduler.alphas_cumprod.clone())
        # Inference sampler: deterministic DDIM. The training scheduler above is
        # DDPMScheduler; sampling with *strided* DDPM (few steps, e.g. 25) leaves
        # residual noise speckles inside the filled holes. DDIM produces clean
        # reconstructions at 20-50 steps from the same trained eps-predictor.
        self.ddim_scheduler = DDIMScheduler(
            num_train_timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
            beta_schedule="linear",
            clip_sample=False,
            set_alpha_to_one=True,
            steps_offset=0,
        )

        # 3. U-Net backbone + SketchEncoder + per-layer ConditionModulator
        in_ch = latent_channels * 2 + 1  # 4 (noisy latent) + 1 (mask) + 4 (masked latent) = 9
        if self.use_diffusers_unet:
            unet_dir = os.path.join(resolve_unet_path(sd_model_id), "unet")
            if unet_pretrained:
                # Load the real SD-inpainting UNet weights (~3.4GB safetensors).
                self.unet = UNet2DConditionModel.from_pretrained(unet_dir)
            else:
                # Build the architecture from config.json ONLY (random weights);
                # the subsequent checkpoint resume / eval load fills the UNet
                # weights. This avoids the ~3.4GB eager safetensors read that
                # OOMs ("page file too small") or segfaults under host-RAM
                # pressure from concurrent jobs on this box. Safe because every
                # resume checkpoint in the recovery pipeline (stage2/stage2c/
                # stage3) already contains the full trained UNet state.
                config = UNet2DConditionModel.load_config(unet_dir)
                self.unet = UNet2DConditionModel.from_config(config)
            # A base SD UNet ships with a 4-channel conv_in; adapt it to the
            # 9-channel inpainting input our branch concatenates.
            self.unet = expand_conv_in_to_9channels(self.unet)
            if gradient_checkpointing:
                self.unet.enable_gradient_checkpointing()
            # Memory: attention slicing splits the QKV attention computation so
            # the [seq x seq] attention matrix is never materialized all at once.
            # Big peak-VRAM reduction at 512 eval (latent 64x64 -> 4096 tokens).
            if self.mem_efficient and hasattr(self.unet, "enable_attention_slicing"):
                self.unet.enable_attention_slicing()
            # Optional: xformers memory-efficient attention (needs the xformers
            # package; silently skipped if unavailable). Further reduces peak.
            if self.mem_efficient:
                try:
                    self.unet.enable_xformers_memory_efficient_attention()
                except Exception:
                    pass
            block_channels: Tuple[int, ...] = tuple(self.unet.config.block_out_channels)
            self.sketch_encoder = SketchEncoder(
                in_channels=1, out_channels=block_channels, vae_scale_factor=8
            )
            self.down_modulators = nn.ModuleList(
                [ConditionModulator(c, c, beta_init=beta_init) for c in block_channels]
            )
            self.mid_modulator = ConditionModulator(
                block_channels[-1], block_channels[-1], beta_init=beta_init
            )
            self.up_modulators = nn.ModuleList(
                [ConditionModulator(c, c, beta_init=beta_init) for c in reversed(block_channels)]
            )
            # ControlNet-lite ADDITIVE sketch residual at each resolution: lets
            # the sketch inject structure, not merely scale features. Zero-init
            # -> identity at init (safe warm-start over a converged checkpoint).
            self.down_add = nn.ModuleList(
                [AdditiveSketchResidual(c, c) for c in block_channels]
            )
            self.mid_add = AdditiveSketchResidual(
                block_channels[-1], block_channels[-1]
            )
            self.up_add = nn.ModuleList(
                [AdditiveSketchResidual(c, c) for c in reversed(block_channels)]
            )
            # Cross-attention injection: replace the dead all-zero cross-attn
            # token with real sketch-derived context (see CrossAttnInjector).
            cross_dim = getattr(self.unet.config, "cross_attention_dim", 768)
            self.cross_inject = CrossAttnInjector(
                in_ch=block_channels[0], mid_ch=256, out_ch=cross_dim, tok=32
            )
            self._register_diffusers_modulation_hooks()
        else:
            self.sketch_encoder = SketchEncoder(
                in_channels=1, out_channels=unet_channels, vae_scale_factor=8
            )
            self.unet = ModulatedUNet(
                in_ch=in_ch,
                out_ch=latent_channels,
                channels=unet_channels,
                beta_init=beta_init,
            )
            self.cross_inject = None  # small custom UNet uses direct sketch_feats
            self.down_add = self.mid_add = self.up_add = None

        # Frequency-domain sharpening (DiffuMural FDP-lite). Independent of the
        # UNet type; applied inside decode() to fight restoration blur.
        self.fdp_enabled = bool(fdp_enabled)
        self.fdp = FrequencySharpen(grid=fdp_grid) if self.fdp_enabled else None

        # ---- LF/HF decomposition (models/freqband.py) ------------------------
        # Lossless band split used to supervise the low-frequency (tone /
        # composition) and high-frequency (lines / texture / edges) restoration
        # targets separately. The learned HF refiner runs AFTER the VAE decoder
        # because the 8x-downsampled latent cannot represent pixel-scale detail.
        self.band_split = BandSplit(
            mode=band_mode, sigma_frac=band_sigma_frac, radius=band_radius
        )
        self.hf_refiner = (
            HighFreqRefiner(base=hf_refiner_base, blocks=hf_refiner_blocks)
            if hf_refiner else None
        )

    def _register_diffusers_modulation_hooks(self) -> None:
        """
        Safely attach ConditionModulator hooks to diffusers UNet2DConditionModel.
        Modulates the primary hidden state of each down_block, mid_block, and up_block.
        A zero-init AdditiveSketchResidual is applied right after each modulator so
        the sketch can inject structure (not only scale it).
        """

        def make_down_hook(scale_idx: int):
            def hook(_module, _inp, output):
                f_s = self._current_sketch_feats[scale_idx]
                mod = self.down_modulators[scale_idx]
                add = self.down_add[scale_idx]
                if isinstance(output, tuple):
                    hidden = add(mod(output[0], f_s), f_s)
                    return (hidden,) + output[1:]
                return add(mod(output, f_s), f_s)
            return hook

        def mid_hook(_module, _inp, output):
            f_s = self._current_sketch_feats[-1]
            return self.mid_add(self.mid_modulator(output, f_s), f_s)

        def make_up_hook(up_idx: int):
            def hook(_module, _inp, output):
                scale_idx = len(self._current_sketch_feats) - 1 - up_idx
                f_s = self._current_sketch_feats[scale_idx]
                mod = self.up_modulators[up_idx]
                add = self.up_add[up_idx]
                if isinstance(output, tuple):
                    hidden = add(mod(output[0], f_s), f_s)
                    return (hidden,) + output[1:]
                return add(mod(output, f_s), f_s)
            return hook

        for idx, block in enumerate(self.unet.down_blocks):
            block.register_forward_hook(make_down_hook(idx))
        if self.unet.mid_block is not None:
            self.unet.mid_block.register_forward_hook(mid_hook)
        for idx, block in enumerate(self.unet.up_blocks):
            block.register_forward_hook(make_up_hook(idx))

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """Encode RGB image [-1, 1] [B, 3, H, W] into scaled latent [B, 4, H/8, W/8].

        The VAE is forced to fp32 regardless of the model's working dtype: the
        SD VAE is numerically fragile under bf16/fp16 (it produces severe
        artifacts that collapse whole-image PSNR by ~15 dB). Running it in fp32
        costs only the small VAE weight memory; the much larger UNet stays in
        the reduced dtype, so the training/inference memory win is preserved.
        """
        if self.vae is not None:
            if self.vae_cpu_offload:
                self.vae.to(image.device)
            with torch.no_grad(), torch.autocast(device_type=image.device.type, enabled=False):
                vae = self.vae.float()
                posterior = vae.encode(image.float()).latent_dist
                z = posterior.sample()
            if self.vae_cpu_offload:
                self.vae.to("cpu")
            return z * self.scale_factor
        with torch.no_grad():
            return self.fallback_encoder(image.float()) * self.scale_factor

    def decode(self, latent: torch.Tensor, mask: Optional[torch.Tensor] = None,
               sketch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode scaled latent [B, 4, H/8, W/8] into RGB image [-1, 1] [B, 3, H, W].

        `mask` / `sketch` are only consumed by the optional learned HF refiner;
        leaving them None (the default) keeps the behaviour bit-identical to
        the pre-refiner implementation.
        """
        z = latent / self.scale_factor
        if self.vae is not None:
            if self.vae_cpu_offload:
                self.vae.to(latent.device)
            with torch.autocast(device_type=latent.device.type, enabled=False):
                vae = self.vae.float()
                image = vae.decode(z.float()).sample
            if self.vae_cpu_offload:
                self.vae.to("cpu")
        else:
            image = self.fallback_decoder(z)
        image = image.clamp(-1.0, 1.0)
        # FDP-lite sharpening: identity at init, learns to de-blur during training.
        # FDP self-casts to fp32 internally, so it is safe under any model dtype.
        if self.fdp is not None:
            image = self.fdp(image)
        # Learned image-space high-frequency generator (identity at init).
        if self.hf_refiner is not None:
            image = self.hf_refiner(image, mask, sketch)
        return image.clamp(-1.0, 1.0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Standard DDPM forward noise addition via DDPMScheduler."""
        return self.scheduler.add_noise(x0, noise, t)

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        mask: torch.Tensor,
        sketch: torch.Tensor,
        masked_image_latent: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict noise epsilon_theta.

        Inputs:
            latent:              [B, 4, h, w] noisy latent z_t
            timestep:            [B] diffusion step t
            mask:                [B, 1, H, W] binary damage mask (1 = damaged)
            sketch:              [B, 1, H, W] continuous probability map S in [0, 1]
            masked_image_latent: Optional [B, 4, h, w] latent of damaged image
        Output:
            pred_noise:          [B, 4, h, w]
        """
        mask_latent = F.interpolate(mask, size=latent.shape[-2:], mode="nearest")
        if masked_image_latent is None:
            masked_image_latent = latent * (1.0 - mask_latent)

        latent_input = torch.cat([latent, mask_latent, masked_image_latent], dim=1)
        sketch_feats = self.sketch_encoder(sketch)

        if self.use_diffusers_unet:
            self._current_sketch_feats = sketch_feats
            cross_dim = getattr(self.unet.config, "cross_attention_dim", 768)
            if self.cross_inject is not None:
                # Real sketch-derived cross-attn context (latent-res sketch
                # feature) replaces the formerly all-zero token.
                encoder_hidden_states = self.cross_inject(sketch_feats[0])
            else:
                bsz = latent.shape[0]
                encoder_hidden_states = torch.zeros(
                    (bsz, 1, cross_dim), device=latent.device, dtype=latent.dtype
                )
            return self.unet(
                latent_input,
                timestep,
                encoder_hidden_states=encoder_hidden_states,
            ).sample

        return self.unet(latent_input, timestep, sketch_feats)

    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        sketch: torch.Tensor,
        num_steps: int = 50,
        tt: "Optional[object]" = None,
        sampler: str = "ddim",
        init_image: Optional[torch.Tensor] = None,
        t_start: int = 0,
        one_step: int = 0,
        one_step_multi: Optional[list] = None,
        cascade: bool = False,
        cond_init: bool = False,
        refine_ts: Optional[list] = None,
        return_latent: bool = False,
        latent_refiner: "Optional[object]" = None,
    ) -> "Union[torch.Tensor, tuple]":
        """
        Reverse diffusion sampling for mural restoration.
        Input:  image [B, 3, H, W], mask [B, 1, H, W], sketch [B, 1, H, W]
        Output: restored_image [B, 3, H, W] in [-1, 1]

        The sampler anchors the KNOWN (unmasked) region with its correctly
        noised latent at every step (standard LDM / SD-inpainting procedure),
        so the model only has to fill the damaged (masked) hole. This keeps the
        training/inference condition consistent in the known region and
        dramatically stabilizes hole-filling for a small, lightly-trained model.

        `tt` is an optional TTGuidance (models/tt_prior.py). When given, the
        single-step x0 estimate is nudged toward its rank-R Tensor-Train
        projection (or toward a pre-fitted global TT prior) inside the hole,
        with strength w(t). The correction is applied to the *signal* component
        only:  z_prev += sqrt(alpha_bar_{t-1}) * delta,  so the noise
        realisation is untouched and w(t)=0 is an exact no-op.
        """
        damaged = image * (1.0 - mask)
        z_masked = self.encode(damaged)
        mask_latent = F.interpolate(mask, size=z_masked.shape[-2:], mode="nearest")

        # SDEdit-style partial init: start the hole from a (partially noised)
        # classical fill instead of pure noise. Anchoring generation to a
        # color-correct init massively improves hole PSNR and prevents the
        # "dark speckle" failure mode on scattered small holes.
        z0_init = self.encode(init_image) if init_image is not None else None
        if cond_init and z0_init is not None:
            # Condition the 9th input channel on the CLASSICAL FILL instead of the
            # gray-hole damaged image. Matches `fill_consistent` training and gives
            # the eps-predictor exact knowledge of the latent it must correct.
            z_masked = z0_init
        if z0_init is not None and t_start > 0:
            t0 = int(min(t_start, self.timesteps - 1))
            t0_b = torch.full((z_masked.shape[0],), t0, device=z_masked.device, dtype=torch.long)
            z = self.q_sample(z0_init, t0_b, torch.randn_like(z0_init))
        else:
            z = torch.randn_like(z_masked)
            t0 = 0

        # One-step Tweedie estimate (MMSE / conditional-mean inpainting):
        #   x0_hat = (z_t - sqrt(1-ab_t) * eps_hat) / sqrt(ab_t)
        # Uses the trained eps-predictor once, instead of drawing a *sample* from
        # the reverse process. For PSNR the conditional MEAN is optimal, whereas
        # iterative sampling drifts to a plausible-but-different sample.
        if one_step > 0 or one_step_multi:
            ts_list = sorted((one_step_multi if one_step_multi else [one_step]), reverse=True)
            ab = self.alphas_cumprod.to(z.device, dtype=torch.float32)
            base0 = z0_init if z0_init is not None else z_masked
            x0_prev = base0.float()
            x0_sum, n = None, 0
            for t_os in ts_list:
                t_os = int(min(t_os, self.timesteps - 1))
                t_b = torch.full((z.shape[0],), t_os, device=z.device, dtype=torch.long)
                if not cascade:
                    # independent estimates: always re-noise the ORIGINAL init
                    z_t = z if (z0_init is not None and t_start == t_os) else \
                        self.q_sample(base0, t_b, torch.randn_like(base0))
                else:
                    # cascaded refinement: re-noise the PREVIOUS estimate
                    z_t = self.q_sample(x0_prev.to(z.dtype), t_b, torch.randn_like(base0))
                eps = self.forward(z_t, t_b, mask, sketch, masked_image_latent=z_masked)
                ab_t = ab[t_os]
                x0_hat = (z_t.float() - (1.0 - ab_t).sqrt() * eps.float()) / ab_t.sqrt().clamp_min(1e-3)
                x0_prev = x0_hat
                x0_sum = x0_hat if x0_sum is None else x0_sum + x0_hat
                n += 1
            # Multi-t Tweedie: average MMSE estimates across noise levels --
            # each t gives a differently-biased conditional-mean approximation;
            # their mean has lower variance than any single t.
            x0_hat = x0_sum / n

            # ---- Iterative Tweedie refinement (descending t ladder) ---------
            # Deliberately NOT the same as `cascade` (which AVERAGES every
            # level, including the high-error high-t ones). Here we keep the
            # FINAL, lowest-t estimate: the one-step Tweedie error is
            #   x0_hat - x0 = (eps_pred_error) / c(t),  c(t) = sqrt(ab/(1-ab))
            # and c grows as t falls, so each rung of the ladder has strictly
            # less error amplification than the one above it. Re-noising the
            # current estimate and re-estimating therefore *contracts* toward
            # x0 rather than merely averaging differently-biased guesses.
            # Default None == exact no-op (bit-identical to before).
            if refine_ts:
                x0_cur = x0_hat.float()
                for t_r in sorted(refine_ts, reverse=True):
                    t_r = int(min(max(int(t_r), 0), self.timesteps - 1))
                    t_b = torch.full((z.shape[0],), t_r, device=z.device, dtype=torch.long)
                    z_t = self.q_sample(x0_cur.to(z.dtype), t_b, torch.randn_like(z))
                    eps_r = self.forward(z_t, t_b, mask, sketch, masked_image_latent=z_masked)
                    ab_r = ab[t_r]
                    x0_cur = (z_t.float()
                              - (1.0 - ab_r).sqrt() * eps_r.float()) / ab_r.sqrt().clamp_min(1e-3)
                x0_hat = x0_cur

            if latent_refiner is not None:
                with torch.amp.autocast(device_type=z.device.type, enabled=False):
                    x0_hat = latent_refiner(x0_hat.float(), mask_latent)
            pred_img = self.decode(x0_hat.to(z.dtype), mask=mask, sketch=sketch)
            out = pred_img * mask + image * (1.0 - mask)
            return (out, x0_hat) if return_latent else out

        if tt is not None:
            tt.prepare(z_masked, mask_latent, vae_encode=self.encode,
                       image=damaged, mask=mask)

        # Strided-DDPM (few steps) leaves residual noise speckles in the filled
        # holes; deterministic DDIM is the default inference sampler.
        sched = self.ddim_scheduler if sampler == "ddim" else self.scheduler
        sched.set_timesteps(min(num_steps, self.timesteps), device=z.device)
        sched_ts = sched.timesteps
        if t0 > 0:
            sched_ts = sched_ts[sched_ts <= t0]
            if len(sched_ts) == 0:  # degenerate: t0 below the coarsest step
                sched_ts = sched.timesteps[-1:]
        for i, t in enumerate(sched_ts):
            t_batch = torch.full((z.shape[0],), int(t.item()), device=z.device, dtype=torch.long)
            # Anchor known region: z_in = model_guess(hole) + noised_known_latent(known)
            noise_known = torch.randn_like(z_masked)
            z_t_known = self.q_sample(z_masked, t_batch, noise_known)
            z_in = z * mask_latent + z_t_known * (1.0 - mask_latent)
            pred_noise = self.forward(z_in, t_batch, mask, sketch, masked_image_latent=z_masked)
            z = sched.step(pred_noise, t, z_in).prev_sample

            # ---- Low-rank (Tensor-Train) structure guidance -------------------
            # x0-space correction, propagated back to z through the SIGNAL
            # component only: z_prev += sqrt(alpha_bar_{t-1}) * delta.
            if tt is not None:
                ab = self.alphas_cumprod.to(z.device, dtype=torch.float32)
                t_idx = int(t.item())
                ab_t = ab[t_idx]
                x0_hat = (z_in.float()
                          - (1.0 - ab_t).sqrt() * pred_noise.float()) / ab_t.sqrt().clamp_min(1e-3)
                delta = tt.delta(x0_hat, t_idx / float(self.timesteps), mask_latent)
                if delta is not None:
                    prev_idx = int(sched_ts[i + 1].item()) if i + 1 < len(sched_ts) else 0
                    sq = ab[prev_idx].sqrt().clamp_min(0.0)
                    if getattr(tt.cfg, "scale_mode", "signal") == "boost":
                        # un-attenuated: sqrt(alpha_bar) at t=T is ~0.007, which
                        # would silently turn an "early" schedule into a no-op.
                        sq = 1.0 / sq.clamp_min(getattr(tt.cfg, "boost_floor", 0.2))
                        c = getattr(tt.cfg, "delta_clamp", 3.0)
                        delta = delta.clamp(-c, c)
                    z = z + (sq * delta.to(z.dtype)).to(z.device)

        if latent_refiner is not None:
            with torch.amp.autocast(device_type=z.device.type, enabled=False):
                z = latent_refiner(z.float(), mask_latent)
        pred_img = self.decode(z, mask=mask, sketch=sketch)
        out = pred_img * mask + image * (1.0 - mask)
        return (out, z) if return_latent else out