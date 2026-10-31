"""Configuration dataclass for DiffSketch-Joint."""
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DiffSketchConfig:
    """All model, diffusion, and training hyperparameters."""
    # Image & dataset settings
    image_size: int = 256
    crop_size: int = 256
    image_channels: int = 3
    sketch_channels: int = 1
    mask_channels: int = 1
    random_flip: bool = True

    # Architecture channels
    base_channels: int = 32
    unet_channels: Tuple[int, ...] = (64, 128, 256)
    sketch_unet_channels: Tuple[int, ...] = (16, 32, 64)
    cfe_channels: int = 64
    latent_channels: int = 4
    vae_scale_factor: int = 8

    # Diffusion settings
    diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    sketch_sigmoid_temp: float = 5.0
    modulator_beta_init: float = 0.5
    # Training-time timestep sampling range [t_sample_lo, t_sample_hi).
    # Default [0, 1000). Narrowing it (e.g. [100, 500]) specializes the
    # eps-predictor on the noise regime used by the one-step Tweedie (MMSE)
    # estimator at inference, improving the PSNR-optimal fill.
    t_sample_lo: int = 0
    t_sample_hi: int = 0  # 0 = use diffusion_steps
    # Train/inference consistency for the one-step Tweedie estimator.
    # At inference the noisy latent is built from the *pyramid-filled* image,
    # not from the ground truth. With fill_consistent=True the training pair
    # matches that: z_t = q_sample(z_fill, t) while the eps target is the one
    # that Tweedie-inverts to the *clean* image, i.e. the model is trained to
    # remove both the injected noise and the fill residual in one step.
    fill_consistent: bool = False

    # Pretrained model paths / IDs
    pretrained_vae: str = "stabilityai/sd-vae-ft-mse"
    sd_model_id: str = "runwayml/stable-diffusion-inpainting"
    use_diffusers_unet: bool = False  # Set True to load full SD UNet2DConditionModel
    allow_fallback_vae: bool = False  # Set True only for offline unit-test debugging
    cfe_pretrained_path: str = "weights/cfe_pretrained.pt"
    gradient_checkpointing: bool = True

    # Optimization & loss weights
    learning_rate: float = 1e-4
    injector_learning_rate: float = 1e-4  # dedicated LR for the CrossAttnInjector adapter
    additive_learning_rate: float = 2e-4  # dedicated LR for the AdditiveSketchResidual heads
    sketch_learning_rate_ratio: float = 0.1
    weight_decay: float = 1e-4
    lambda_diff: float = 1.0
    lambda_sketch: float = 0.5
    lambda_struct: float = 0.1
    lambda_perc: float = 0.4  # VGG19 perceptual loss (decoded vs GT clean image)
    lambda_hf: float = 0.2    # Laplacian high-frequency loss (decoded vs GT)
    lambda_scc: float = 0.3   # Sketch-Cyclic-Consistency: edges of restored region must
                               # match the input sketch lines inside the mask (DiffuMural
                               # reward/cyclic-consistency loss L(c^v, D(x0')) -- fixes the
                               # "sketch under-utilized -> blur-fill wins" failure mode).
    lambda_sketch_noise: float = 0.1  # Auxiliary DDPM noise loss weight for MLE
    # ---- Damaged-region focus (P1 fix) ----
    # The mask covers only ~17% of the image, so whole-image means in L_diff /
    # L_perc / L_hf are dominated by the 83% that is already correct (anchored
    # known latent). That dilutes the only gradient that matters. mask_focus_weight
    # multiplies the per-pixel loss inside the hole by this factor (>1 = focus);
    # 1.0 reproduces the original whole-image behaviour exactly.
    mask_focus_weight: float = 1.0
    # Direct L1 between the decoded estimate and GT *inside the hole only*.
    # This is the most direct lever on in-hole PSNR; SNR-weighted so high-noise
    # timesteps (whose estimate is nearly noise) do not dominate.
    lambda_hole: float = 0.0
    # ---- LF/HF frequency decomposition of the restoration target ----
    # Evidence (probe_freqband.py): in the LATENT domain there is no band whose
    # restoration signal beats white noise better than average, so re-weighting
    # the eps target by band buys nothing. What IS strongly band-dependent is
    # the IMAGE-DOMAIN target: the LF part (tone / composition) is roughly an
    # order of magnitude larger in amplitude than the HF part (lines / texture
    # / edges), so one unnormalised pixel loss is dominated by LF and the
    # visibly-deficient HF band gets almost no gradient. These two terms split
    # the decoded target per band and normalise each by its own contrast.
    # 0.0 = disabled (default) -> bit-identical to the previous objective.
    lambda_band_low: float = 0.0    # low band: overall tone + coarse composition
    lambda_band_high: float = 0.0   # high band: lines / texture / edge detail
    band_mode: str = "gaussian"     # gaussian (smooth, no ringing) | fft (hard cutoff)
    band_sigma_frac: float = 0.01   # Gaussian cutoff = frac * H (resolution-relative)
    band_radius: float = 0.25       # FFT cutoff as a fraction of Nyquist
    # ---- Bilateral contrastive loss (Dunhuang semantic ritual constraints) ----
    # Pulls the restored image toward GT and pushes it away from a wrong
    # restoration in VGG feature space. 0.0 = disabled (bit-identical to before).
    lambda_contrast: float = 0.0
    contrast_temperature: float = 0.07
    # ---- Direct latent-x0 regression (the PSNR-determining quantity) ----
    # The one-step Tweedie estimator at inference inverts the eps prediction to
    # a clean latent x0_hat = (z_t - sqrt(1-ab)*eps)/sqrt(ab). PSNR is decided
    # *entirely* by how close x0_hat is to the GT latent z_gt inside the hole.
    # No existing term supervises this directly (L_diff is on eps, the image
    # terms are on the decoded image). This adds a hole-focused latent MSE
    # between x0_hat and z_gt, SNR-gated to low-noise timesteps, so the UNet is
    # optimized for exactly the quantity that matters at eval. 0.0 = disabled.
    lambda_x0: float = 0.0
    # Only supervise x0 at timesteps t <= x0_t_max (low noise -> the Tweedie
    # estimate is meaningful; high-t estimates are near-pure noise). Default 200
    # matches the inference one-step timestep.
    x0_t_max: int = 200
    # Learned image-space high-frequency generator applied after the VAE decoder.
    # The 8x-downsampled latent cannot represent pixel-scale line detail, so this
    # gives the model a direct route to generate that band (zero-init = identity
    # at init, safe over a converged checkpoint).
    hf_refiner: bool = False
    hf_refiner_base: int = 32
    hf_refiner_blocks: int = 4
    # Use the integral-image pyramid fill (numerically equivalent, ~5-10x faster).
    # Off by default so old checkpoints/recipes stay bit-comparable; turn on for
    # any new long training run.
    fast_fill: bool = False
    fdp_enabled: bool = True  # Frequency-domain sharpening (DiffuMural FDP-lite) to fight
                              # restoration blur; applied inside decode(), identity at init.
    fdp_grid: int = 16        # resolution of the learnable frequency-gain grid (upsampled)
    accum_steps: int = 1
    amp: bool = True
    clip_grad: bool = True  # False disables torch.nn.utils.clip_grad_norm_ (Windows/WDDM
                            # fused-foreach clip deadlock workaround; use with --no-clip-grad)
    device: str = "cuda"

    # ---- Memory optimization (safe, dependency-free) ----
    mem_efficient: bool = True       # diffusers attention slicing + VAE slicing
    train_bf16: bool = True          # keep model weights in bf16 during training:
                                     # ~halves UNet param+grad+optimizer memory
                                     # (~6-7GB on the 859M UNet). Standard for SD fine-tune.
    vae_cpu_offload: bool = False    # offload the frozen VAE to CPU between uses
                                     # (saves ~1GB VRAM; minor speed cost). Opt-in.

    # ---- Low-rank Tensor-Train (TT) structure guidance ----
    # Inference-time, TRAINING-FREE, OFF by default. See models/tt_prior.py.
    # Probe (output/tt_probe) on held-out murals: a rank-4/8 TT completion of
    # the damaged image scores SSIM_dmg ~0.62 / PSNR_dmg ~17 dB inside the hole
    # (vs mean-fill 0.49/13.6) -- i.e. the global low-rank prior is real, but it
    # is a *smooth layout* prior, not a line-topology prior: plug it in only for
    # the structural (high-t) part of the reverse process.
    tt_enabled: bool = False
    tt_mode: str = "dynamic"     # dynamic = rank-R self-projection | static = fitted prior
    tt_rank: int = 4             # probe sweet spot 4-8; >=16 overfits the observed pixels
    tt_lambda: float = 0.5       # peak strength w(t)
    tt_schedule: str = "early"   # early | mid | late | constant
    tt_t_lo: float = 0.15        # w = 0 below this fraction of T (texture stage)
    tt_t_hi: float = 1.00        # w = lambda at this fraction of T
    tt_p: float = 1.0            # ramp exponent
    tt_fit_iters: int = 200      # static mode: Adam steps for the completion fit
    tt_fit_lr: float = 0.05
    tt_fit_space: str = "latent" # latent (cheap) | image (needs an extra VAE encode)

    def tt_config(self):
        """Build a models.tt_prior.TTConfig from these fields."""
        from models.tt_prior import TTConfig
        return TTConfig(
            enabled=self.tt_enabled, mode=self.tt_mode, rank=self.tt_rank,
            lam=self.tt_lambda, schedule=self.tt_schedule, t_lo=self.tt_t_lo,
            t_hi=self.tt_t_hi, p=self.tt_p, fit_iters=self.tt_fit_iters,
            fit_lr=self.tt_fit_lr, fit_space=self.tt_fit_space,
        )