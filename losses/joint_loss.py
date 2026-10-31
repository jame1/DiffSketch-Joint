"""Joint loss functions for DiffSketch-Joint.

    Components:
    1. L_diff:   DDPM noise-prediction MSE (on the latent noise).
    2. L_sketch: Binary Cross-Entropy between predicted S and S_gt.
    3. L_struct: Sobel-edge L1 between decoded image and predicted sketch,
                 optionally weighted per-sample by SNR (alpha_bar_t).
    4. L_perc:   VGG19 perceptual loss (ImageNet, from local cache -> offline-safe)
                 between the decoded clean image and the GT clean image.
    5. L_hf:     Laplacian high-frequency L1 between decoded and GT image.
    7. L_band_low / L_band_high:
                 LF/HF decomposition of the *decoded restoration target*. The
                 low band carries overall tone and coarse composition, the high
                 band carries line / texture / edge detail. Each band is scored
                 against GT **and normalised by that band's own contrast**, which
                 is the whole point: their amplitudes differ by roughly an order
                 of magnitude, so a single unnormalised pixel loss is dominated
                 by the low band and the visibly-deficient high band receives
                 almost no gradient.

L_perc and L_hf directly supervise the *decoded* image toward the ground truth,
which the original noise-MSE objective only does indirectly. They are the main
lever against the "plausible but blurry" restoration (low damaged-region SSIM).
Both are SNR-weighted (alpha_bar_t) so high-timestep single-step estimates --
which are essentially noise -- do not dominate the gradient.
"""
from typing import Dict, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from models.freqband import BandSplit, band_reference


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """
    Differentiable Sobel edge detector.
    Input:  x [B, C, H, W] (RGB in [-1, 1] or grayscale in [0, 1])
    Output: edge_map [B, 1, H, W] normalized to [0, 1]
    """
    if x.shape[1] == 3:
        x_gray = ((x + 1.0) * 0.5).mean(dim=1, keepdim=True)
    else:
        x_gray = x

    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x_gray.dtype,
    ).view(1, 1, 3, 3)
    ky = kx.transpose(2, 3)

    gx = F.conv2d(x_gray, kx, padding=1)
    gy = F.conv2d(x_gray, ky, padding=1)
    grad_mag = torch.sqrt(gx.square() + gy.square() + 1e-8)
    return (grad_mag / 5.656854).clamp(0.0, 1.0)


def laplacian(x: torch.Tensor) -> torch.Tensor:
    """
    Laplacian high-pass filter (per channel).
    Input/output: x [B, C, H, W].
    """
    c = x.shape[1]
    k = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3).repeat(c, 1, 1, 1)
    return F.conv2d(x, k, padding=1, groups=c)


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _imagenet_norm(x: torch.Tensor) -> torch.Tensor:
    """Map RGB in [-1, 1] -> ImageNet channel-normalized input."""
    x = (x + 1.0) * 0.5
    mean = x.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


class VGGPerceptual(nn.Module):
    """
    Frozen VGG19 feature extractor for perceptual loss.

    Loads ImageNet weights from the *local torch hub cache* (vgg19-*.pth),
    so it works fully offline. Only early/mid ReLU activations are kept
    (default relu1_2, relu2_2, relu3_3) -- these carry most of the
    texture / edge detail that matters for sharpness, while keeping the
    activation memory small.
    """

    def __init__(self, layer_ids: Tuple[int, ...] = (3, 8, 15)):
        super().__init__()
        from torchvision.models import vgg19, VGG19_Weights

        feats = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features
        slices = nn.ModuleList()
        prev = 0
        for lid in sorted(layer_ids):
            slices.append(feats[prev:lid + 1])
            prev = lid + 1
        for p in slices.parameters():
            p.requires_grad_(False)
        slices.eval()
        self.slices = slices

    def forward(self, x: torch.Tensor) -> list:
        x = _imagenet_norm(x)
        outs = []
        for s in self.slices:
            x = s(x)
            outs.append(x)
        return outs


class DiffSketchJointLoss(nn.Module):
    """
    Computes the weighted sum of:
        1. L_diff:   DDPM noise prediction MSE.
        2. L_sketch: Binary Cross-Entropy between predicted S and S_gt (if provided).
        3. L_struct: L1 distance between Sobel edges of decoded image and predicted S,
                     optionally weighted per-sample by SNR (alpha_bar_t).
        4. L_perc:   VGG19 perceptual L1 between decoded image and GT image (if image_gt).
        5. L_hf:     Laplacian high-frequency L1 between decoded image and GT image.

    L_perc / L_hf are SNR-weighted by struct_sample_weights (alpha_bar_t) so that
    only low-noise (meaningful) single-step clean-image estimates contribute.
    """

    def __init__(
        self,
        lambda_diff: float = 1.0,
        lambda_sketch: float = 0.5,
        lambda_struct: float = 0.1,
        lambda_perc: float = 0.2,
        lambda_hf: float = 0.2,
        lambda_scc: float = 0.3,
        lambda_hole: float = 0.0,
        mask_focus_weight: float = 1.0,
        perc_layers: Tuple[int, ...] = (3, 8, 15),
        lambda_band_low: float = 0.0,
        lambda_band_high: float = 0.0,
        band_mode: str = "gaussian",
        band_sigma_frac: float = 0.01,
        band_radius: float = 0.25,
        lambda_contrast: float = 0.0,
        contrast_temperature: float = 0.07,
        lambda_x0: float = 0.0,
        x0_t_max: float = 200,
    ):
        super().__init__()
        self.lambda_diff = lambda_diff
        self.lambda_sketch = lambda_sketch
        self.lambda_struct = lambda_struct
        self.lambda_perc = lambda_perc
        self.lambda_hf = lambda_hf
        self.lambda_scc = lambda_scc
        self.lambda_hole = lambda_hole
        self.mask_focus_weight = float(mask_focus_weight)
        self._perc_layers = perc_layers
        # LF/HF decomposition of the restoration target (default weights 0 ->
        # disabled, i.e. bit-identical to the pre-decomposition objective).
        self.lambda_band_low = float(lambda_band_low)
        self.lambda_band_high = float(lambda_band_high)
        self.lambda_contrast = float(lambda_contrast)
        self.contrast_temperature = float(contrast_temperature)
        # Direct latent-x0 regression weight (default 0 = disabled).
        self.lambda_x0 = float(lambda_x0)
        self.x0_t_max = float(x0_t_max)
        # Lazily created on first use (so Stage 1 never pays the contrast cost).
        self._contrast: Optional["BilateralContrastiveLoss"] = None
        self._band = BandSplit(
            mode=band_mode, sigma_frac=band_sigma_frac, radius=band_radius
        )
        # Lazily created on first use (so Stage 1 never pays the VGG cost).
        self._vgg: Optional[VGGPerceptual] = None

    def _hole_weight_map(
        self, mask: Optional[torch.Tensor], size, ref: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Per-pixel weight map [B, 1, h, w] = 1 + (focus - 1) * mask, resampled to
        `size`. Returns None when no focusing is requested (focus == 1 or no mask),
        in which case callers fall back to the plain uniform mean.
        """
        if mask is None or abs(self.mask_focus_weight - 1.0) < 1e-6:
            return None
        m = mask.float()
        if m.shape[-2:] != tuple(size):
            m = F.interpolate(m, size=tuple(size), mode="area")
        return 1.0 + (self.mask_focus_weight - 1.0) * m

    def _get_vgg(self, device: torch.device) -> VGGPerceptual:
        if self._vgg is None:
            self._vgg = VGGPerceptual(self._perc_layers).to(device)
            self._vgg.eval()
        return self._vgg

    def forward(
        self,
        pred_noise: Optional[torch.Tensor],
        noise: Optional[torch.Tensor],
        sketch_pred: Optional[torch.Tensor],
        sketch_gt: Optional[torch.Tensor],
        image_pred: Optional[torch.Tensor],
        image_gt: Optional[torch.Tensor] = None,
        struct_sample_weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        negative: Optional[torch.Tensor] = None,
        x0_hat: Optional[torch.Tensor] = None,
        z_gt: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        ref = next(
            t for t in (pred_noise, sketch_pred, image_pred) if t is not None
        )
        zero = ref.new_zeros(())

        # 1. Inpainting diffusion noise MSE loss (optionally hole-focused)
        if pred_noise is not None and noise is not None:
            diff_sq = (pred_noise.float() - noise.float()).square()
            w_lat = self._hole_weight_map(mask, pred_noise.shape[-2:], ref)
            if w_lat is not None:
                denom = w_lat.sum() * diff_sq.shape[1]
                l_diff = (diff_sq * w_lat).sum() / denom.clamp_min(1e-6)
            else:
                l_diff = diff_sq.mean()
        else:
            l_diff = zero

        # 2. Sketch supervision BCE loss (disable autocast for float32 numerical safety)
        if sketch_pred is not None and sketch_gt is not None:
            with torch.amp.autocast(device_type=ref.device.type, enabled=False):
                s_prob = sketch_pred.float().clamp(1e-6, 1.0 - 1e-6)
                s_gt = sketch_gt.float().clamp(0.0, 1.0)
                l_sketch = F.binary_cross_entropy(s_prob, s_gt)
        else:
            l_sketch = zero

        # 3. Structural consistency Sobel L1 loss on decoded image
        if image_pred is not None and sketch_pred is not None:
            pred_edges = sobel_edges(image_pred)
            if pred_edges.shape[-2:] != sketch_pred.shape[-2:]:
                pred_edges = F.interpolate(
                    pred_edges,
                    size=sketch_pred.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            per_pixel_l1 = (pred_edges.float() - sketch_pred.float()).abs()
            if struct_sample_weights is not None:
                w = struct_sample_weights.view(-1, 1, 1, 1).float()
                l_struct = (per_pixel_l1 * w).mean()
            else:
                l_struct = per_pixel_l1.mean()
        else:
            l_struct = zero

        # 4 + 5. Perceptual (VGG) and high-frequency (Laplacian) losses on the
        # decoded image vs. the clean GT image. Computed in fp32 (autocast off)
        # for numerical stability; gradients flow back through the VAE decode.
        l_perc = zero
        l_hf = zero
        l_hole = zero
        if (
            image_pred is not None
            and image_gt is not None
            and (self.lambda_perc > 0 or self.lambda_hf > 0)
        ):
            with torch.amp.autocast(device_type=ref.device.type, enabled=False):
                pred_f = image_pred.float()
                gt_f = image_gt.float()
                if struct_sample_weights is not None:
                    snr_w = struct_sample_weights.view(-1).float()  # [B] = alpha_bar_t
                else:
                    snr_w = None

                if self.lambda_perc > 0:
                    vgg = self._get_vgg(pred_f.device)
                    fp = vgg(pred_f)
                    with torch.no_grad():
                        ft = vgg(gt_f)
                    perc_layers = []
                    for a, b in zip(fp, ft):
                        d = (a - b).abs()
                        # Weight in-hole features up: the hole is ~17% of pixels,
                        # so a uniform feature mean is dominated by the intact region.
                        w_l = self._hole_weight_map(
                            mask, d.shape[-2:], ref
                        ) if mask is not None else None
                        if w_l is not None:
                            denom = w_l.sum(dim=(1, 2, 3)) * d.shape[1]
                            perc_layers.append(
                                (d * w_l).sum(dim=(1, 2, 3)) / denom.clamp_min(1e-6)
                            )
                        else:
                            perc_layers.append(d.mean(dim=(1, 2, 3)))
                    perc_sample = torch.stack(perc_layers, 0).mean(0)  # [B]
                    l_perc = (
                        (perc_sample * snr_w).mean()
                        if snr_w is not None
                        else perc_sample.mean()
                    )

                if self.lambda_hf > 0:
                    d_hf = (laplacian(pred_f) - laplacian(gt_f)).abs()
                    w_hf = self._hole_weight_map(mask, d_hf.shape[-2:], ref)
                    if w_hf is not None:
                        denom = w_hf.sum(dim=(1, 2, 3)) * d_hf.shape[1]
                        hf_sample = (d_hf * w_hf).sum(dim=(1, 2, 3)) / denom.clamp_min(1e-6)
                    else:
                        hf_sample = d_hf.mean(dim=(1, 2, 3))
                    l_hf = (
                        (hf_sample * snr_w).mean()
                        if snr_w is not None
                        else hf_sample.mean()
                    )

                # 4b. Direct in-hole pixel L1 (the most direct lever on in-hole PSNR).
                if (
                    self.lambda_hole > 0
                    and mask is not None
                    and image_pred is not None
                    and image_gt is not None
                ):
                    m = mask.float()
                    if m.shape[-2:] != pred_f.shape[-2:]:
                        m = F.interpolate(
                            m, size=pred_f.shape[-2:], mode="area"
                        )
                    d_pix = (pred_f - gt_f).abs()
                    denom = m.sum(dim=(1, 2, 3)) * d_pix.shape[1]
                    hole_sample = (d_pix * m).sum(dim=(1, 2, 3)) / denom.clamp_min(1e-6)
                    l_hole = (
                        (hole_sample * snr_w).mean()
                        if snr_w is not None
                        else hole_sample.mean()
                    )

        # 4c + 4d. LF / HF decomposition of the RESTORATION TARGET.
        #   low  band -> overall tone and coarse composition
        #   high band -> line / texture / edge detail
        # Each band is normalised by its own GT contrast (see
        # freqband.band_reference); without that the low band's much larger
        # amplitude would dominate and the high band would receive almost no
        # gradient -- which is precisely the "plausible but detail-free" failure
        # we are trying to fix. Same mask-focus and SNR weighting as the other
        # image-domain terms so it composes with the existing machinery.
        l_band_low = zero
        l_band_high = zero
        if (
            image_pred is not None
            and image_gt is not None
            and (self.lambda_band_low > 0 or self.lambda_band_high > 0)
        ):
            with torch.amp.autocast(device_type=ref.device.type, enabled=False):
                bp_l, bp_h = self._band(image_pred.float())
                bg_l, bg_h = self._band(image_gt.float())
                if struct_sample_weights is not None:
                    snr_w2 = struct_sample_weights.view(-1).float()
                else:
                    snr_w2 = None

                def _term(pb: torch.Tensor, gb: torch.Tensor) -> torch.Tensor:
                    d = (pb - gb).abs()
                    wb = self._hole_weight_map(mask, d.shape[-2:], ref)
                    if wb is not None:
                        v = (d * wb).sum(dim=(1, 2, 3)) / (
                            wb.sum(dim=(1, 2, 3)) * d.shape[1]
                        ).clamp_min(1e-6)
                    else:
                        v = d.mean(dim=(1, 2, 3))
                    v = v / band_reference(gb)
                    return (v * snr_w2).mean() if snr_w2 is not None else v.mean()

                if self.lambda_band_low > 0:
                    l_band_low = _term(bp_l, bg_l)
                if self.lambda_band_high > 0:
                    l_band_high = _term(bp_h, bg_h)

        # 6. Sketch-Cyclic-Consistency (DiffuMural reward/cyclic loss L(c^v, D(x0'))).
        # Within the damaged mask, the restored image's Sobel edges must line up
        # with the INPUT sketch lines. This forces the network to actually honour
        # the sketch rather than ignore it (the blur-fill baseline does the latter).
        l_scc = zero
        if (
            image_pred is not None
            and sketch_gt is not None
            and mask is not None
            and self.lambda_scc > 0
        ):
            with torch.amp.autocast(device_type=ref.device.type, enabled=False):
                pred_e = sobel_edges(image_pred.float())      # [B,1,H,W]
                skt_e = sobel_edges(sketch_gt.float())        # sketch is already line-like
                if pred_e.shape[-2:] != skt_e.shape[-2:]:
                    pred_e = F.interpolate(
                        pred_e, size=skt_e.shape[-2:], mode="bilinear",
                        align_corners=False,
                    )
                # Focus supervision only where a sketch LINE exists, inside the mask,
                # so we require "restored region contains the sketch lines" without
                # over-penalising legitimate mural texture that has no sketch line.
                skt_line = (skt_e > 0.05).float()            # binary line locations
                m = mask.float()
                if m.shape[-2:] != skt_e.shape[-2:]:
                    m = F.interpolate(m, size=skt_e.shape[-2:], mode="nearest")
                focus = skt_line * m                          # [B,1,H,W]
                denom = focus.sum().clamp_min(1.0)
                l_scc = ((pred_e - skt_e).abs() * focus).sum() / denom

        # 7. Bilateral contrastive loss (Dunhuang semantic ritual constraints).
        #    In VGG feature space, pull the restored image toward the GT
        #    "positive" and push it away from a wrong-restoration "negative".
        #    Only the small projection head is trainable; the VGG backbone is
        #    frozen. Gradients flow back through the VAE decode.
        l_contrast = zero
        if (
            image_pred is not None
            and image_gt is not None
            and negative is not None
            and self.lambda_contrast > 0
        ):
            with torch.amp.autocast(device_type=ref.device.type, enabled=False):
                if self._contrast is None:
                    from losses.contrastive import BilateralContrastiveLoss

                    self._contrast = BilateralContrastiveLoss(
                        perc_layers=self._perc_layers,
                        temperature=self.contrast_temperature,
                    ).to(ref.device)
                if struct_sample_weights is not None:
                    snr_w3 = struct_sample_weights.view(-1).float()
                else:
                    snr_w3 = None
                l_contrast = self._contrast(
                    image_pred.float(),
                    image_gt.float(),
                    negative.float(),
                    mask=mask.float() if mask is not None else None,
                    sample_weights=snr_w3,
                )

        # 8. Direct latent-x0 regression (the PSNR-determining quantity).
        #    x0_hat = (z_t - sqrt(1-ab)*eps)/sqrt(ab) is what the one-step
        #    Tweedie estimator decodes at inference. PSNR is decided entirely by
        #    ||x0_hat - z_gt|| inside the hole. Supervise it directly (hole-
        #    weighted, SNR-gated to low-noise t) so the UNet optimizes the exact
        #    quantity that matters at eval, rather than only the eps proxy.
        l_x0 = zero
        if (
            x0_hat is not None
            and z_gt is not None
            and self.lambda_x0 > 0
        ):
            with torch.amp.autocast(device_type=ref.device.type, enabled=False):
                x0_f = x0_hat.float()
                zg_f = z_gt.float()
                # hole weight in LATENT space
                if mask is not None:
                    m_lat = mask.float()
                    if m_lat.shape[-2:] != x0_f.shape[-2:]:
                        m_lat = F.interpolate(
                            m_lat, size=x0_f.shape[-2:], mode="nearest"
                        )
                else:
                    m_lat = torch.ones_like(x0_f[:, :1])
                d = (x0_f - zg_f).square()          # [B, C, h, w]
                # weight the hole more so the ~15% hole pixels are not drowned
                # by the intact-region (already-correct) latent
                w = m_lat.expand_as(d)               # {0,1}
                denom = w.sum().clamp_min(1e-6) * d.shape[1]
                l_x0 = (d * w).sum() / denom

                # SNR gate: only supervise low-noise timesteps (t <= x0_t_max),
                # where the Tweedie estimate is meaningful. High-t estimates are
                # near-pure noise and would inject a huge, uninformative gradient.
                if timestep is not None:
                    gate = (timestep.float() <= float(self.x0_t_max)).float()  # [B]
                    if gate.sum() > 0:
                        # per-sample gate -> only low-t samples contribute
                        d_s = (d * w).sum(dim=(1, 2, 3)) / (
                            w.sum(dim=(1, 2, 3)).clamp_min(1e-6) * d.shape[1]
                        )
                        l_x0 = (d_s * gate).sum() / gate.sum().clamp_min(1.0)

        total = (
            self.lambda_diff * l_diff
            + self.lambda_sketch * l_sketch
            + self.lambda_struct * l_struct
            + self.lambda_perc * l_perc
            + self.lambda_hf * l_hf
            + self.lambda_scc * l_scc
            + self.lambda_hole * l_hole
            + self.lambda_band_low * l_band_low
            + self.lambda_band_high * l_band_high
            + self.lambda_contrast * l_contrast
            + self.lambda_x0 * l_x0
        )
        return {
            "total": total,
            "diff": l_diff,
            "sketch": l_sketch,
            "struct": l_struct,
            "perc": l_perc,
            "hf": l_hf,
            "scc": l_scc,
            "hole": l_hole,
            "band_low": l_band_low,
            "band_high": l_band_high,
            "contrast": l_contrast,
            "x0": l_x0,
        }
