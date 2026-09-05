"""Nodes in AlcheminePack/Image."""

import torch
import torch.nn.functional as F

import comfy.utils

UPSCALE_METHODS = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]


#################################################################
# Base class
#################################################################
class BaseImage:
    """Base class for Image nodes."""



#################################################################
# Helpers
#################################################################
def _rgb_to_luminance(image: torch.Tensor) -> torch.Tensor:
    """Return per-pixel luminance with a trailing channel dim kept.

    image: (..., H, W, 3) in [0, 1]
    """
    weights = torch.tensor([0.299, 0.587, 0.114], device=image.device, dtype=image.dtype)
    return (image[..., :3] * weights).sum(dim=-1, keepdim=True)


def _conv2d_per_channel(rgb: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Apply a 2D kernel to each channel independently (reflect padding).

    rgb:    (B, H, W, C) in [0, 1]
    kernel: (kh, kw)
    """
    x = rgb.movedim(-1, 1)  # (B, C, H, W)
    c = x.shape[1]
    kh, kw = kernel.shape
    weight = kernel.to(device=x.device, dtype=x.dtype).expand(c, 1, kh, kw)
    padded = F.pad(x, (kw // 2, kw // 2, kh // 2, kh // 2), mode="reflect")
    out = F.conv2d(padded, weight, groups=c)
    return out.movedim(1, -1)


def _cas(rgb: torch.Tensor, sharpness: float) -> torch.Tensor:
    """Contrast Adaptive Sharpening (AMD FidelityFX CAS).

    Adaptive unsharp: sharpens flat regions more and already-sharp edges less,
    so it avoids the crunchy over-sharpening of a fixed kernel.

    rgb:       (B, H, W, C) in [0, 1]
    sharpness: 0.0 = softest sharpen ... 1.0 = strongest
    """
    x = rgb.movedim(-1, 1)  # (B, C, H, W)
    p = F.pad(x, (1, 1, 1, 1), mode="reflect")
    up = p[..., :-2, 1:-1]
    down = p[..., 2:, 1:-1]
    left = p[..., 1:-1, :-2]
    right = p[..., 1:-1, 2:]
    center = x

    mn = torch.minimum(torch.minimum(torch.minimum(up, down), torch.minimum(left, right)), center)
    mx = torch.maximum(torch.maximum(torch.maximum(up, down), torch.maximum(left, right)), center)

    # Adaptive weight: less sharpening where local contrast is already high.
    d = torch.clamp(torch.minimum(mn, 1.0 - mx) / (mx + 1e-6), 0.0, 1.0)
    amp = torch.sqrt(d)
    # sharpness lerps the base weight between gentle (-0.125) and strong (-0.2).
    w = amp * (-0.125 - 0.075 * sharpness)

    out = (center + (up + down + left + right) * w) / (1.0 + 4.0 * w)
    return out.movedim(1, -1)


def _local_contrast(rgb: torch.Tensor, amount: float, radius: int = 15) -> torch.Tensor:
    """Large-radius unsharp mask ("clarity"): boosts mid-scale volume/depth."""
    kernel = torch.ones(radius, radius) / float(radius * radius)
    blurred = _conv2d_per_channel(rgb, kernel)
    return rgb + amount * (rgb - blurred)


def _bilateral(rgb: torch.Tensor, strength: float, radius: int = 2) -> torch.Tensor:
    """Edge-preserving denoise (bilateral filter approximation).

    Smooths flat noise/banding while keeping lines and edges crisp.
    strength: 0.0 = off ... 1.0 = strong smoothing.
    """
    x = rgb.movedim(-1, 1)  # (B, C, H, W)
    p = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    _, _, ph, pw = p.shape
    h, w = x.shape[-2:]

    sigma_space = radius
    # Stronger strength => wider color sigma => smooths across bigger differences.
    sigma_color = 0.05 + 0.35 * strength

    acc = torch.zeros_like(x)
    wsum = torch.zeros_like(x)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            neighbor = p[..., radius + dy:radius + dy + h, radius + dx:radius + dx + w]
            spatial = torch.exp(torch.tensor(-(dy * dy + dx * dx) / (2.0 * sigma_space**2)))
            color = torch.exp(-((neighbor - x) ** 2) / (2.0 * sigma_color**2))
            weight = spatial * color
            acc += neighbor * weight
            wsum += weight
    smoothed = acc / (wsum + 1e-8)
    smoothed = smoothed.movedim(1, -1)
    return rgb * (1.0 - strength) + smoothed * strength


#################################################################
# Nodes
#################################################################
class AdjustImage(BaseImage):
    """Adjust brightness, contrast, saturation, gamma and a suite of sharpen/
    denoise filters (edge enhance, CAS, local contrast, bilateral), plus
    optional upscaling of an image.

    All adjustments run as torch ops on the image's own device (CPU or GPU),
    so this works on GPU tensors without a numpy round-trip. RGB channels only;
    an existing alpha channel (RGBA) is passed through untouched. Every knob
    defaults to a no-op so the node is a pass-through until a slider is moved.
    """

    INPUT_TYPES = lambda: {
        "required": {
            "image": ("IMAGE",),
            "brightness": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "1.0 = no change, <1.0 = darker, >1.0 = brighter.",
                },
            ),
            "contrast": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "1.0 = no change, <1.0 = less contrast, >1.0 = more contrast.",
                },
            ),
            "saturation": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                    "tooltip": "1.0 = no change, 0.0 = grayscale, >1.0 = more vivid.",
                },
            ),
            "gamma": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 5.0,
                    "step": 0.01,
                    "tooltip": "1.0 = no change, <1.0 = brighter mid-tones, >1.0 = darker mid-tones.",
                },
            ),
            "edge_enhance": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Blend factor for edge enhancement. 0.0 = off, 1.0 = full.",
                },
            ),
            "cas": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Contrast Adaptive Sharpening. 0.0 = off. Adaptive, avoids crunchy edges. Great for anime.",
                },
            ),
            "local_contrast": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.01,
                    "tooltip": "Large-radius clarity. 0.0 = off. Adds mid-scale depth/volume.",
                },
            ),
            "denoise": (
                "FLOAT",
                {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Edge-preserving denoise (bilateral). 0.0 = off. Removes flat noise/banding, keeps lines.",
                },
            ),
            "upscale_method": (
                UPSCALE_METHODS,
                {
                    "default": "lanczos",
                    "tooltip": "Resampling method used when scale_by != 1.0.",
                },
            ),
            "scale_by": (
                "FLOAT",
                {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 8.0,
                    "step": 0.05,
                    "tooltip": "Scale factor. 1.0 = no resize, 1.2 = 20% larger, <1.0 = smaller.",
                },
            ),
        }
    }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "AlcheminePack/Image"

    @classmethod
    def execute(
        cls,
        image: torch.Tensor,
        brightness: float = 1.0,
        contrast: float = 1.0,
        saturation: float = 1.0,
        gamma: float = 1.0,
        edge_enhance: float = 0.0,
        cas: float = 0.0,
        local_contrast: float = 0.0,
        denoise: float = 0.0,
        upscale_method: str = "lanczos",
        scale_by: float = 1.0,
    ) -> tuple[torch.Tensor]:
        has_alpha = image.shape[-1] == 4
        alpha = image[..., 3:] if has_alpha else None
        rgb = image[..., :3].clone()

        # Brightness
        if brightness != 1.0:
            rgb = rgb * brightness

        # Contrast (around mid-gray 0.5)
        if contrast != 1.0:
            rgb = (rgb - 0.5) * contrast + 0.5

        # Saturation (interpolate between luminance and color)
        if saturation != 1.0:
            lum = _rgb_to_luminance(rgb)
            rgb = lum + (rgb - lum) * saturation

        rgb = rgb.clamp(0.0, 1.0)

        # Gamma (needs non-negative input, so applied after clamp)
        if gamma != 1.0:
            rgb = rgb.pow(gamma)

        # Denoise first (edge-preserving), so later sharpening doesn't amplify noise.
        if denoise > 0.0:
            rgb = _bilateral(rgb, denoise)

        # Edge enhance (EDGE_ENHANCE_MORE kernel, blended by factor)
        if edge_enhance > 0.0:
            kernel = torch.tensor(
                [[-1.0, -1.0, -1.0], [-1.0, 9.0, -1.0], [-1.0, -1.0, -1.0]]
            )
            enhanced = _conv2d_per_channel(rgb, kernel)
            rgb = rgb * (1.0 - edge_enhance) + enhanced * edge_enhance

        # Contrast Adaptive Sharpening (adaptive, edge-safe)
        if cas > 0.0:
            rgb = _cas(rgb, cas)

        # Local contrast / clarity (large-radius unsharp)
        if local_contrast > 0.0:
            rgb = _local_contrast(rgb, local_contrast)

        rgb = rgb.clamp(0.0, 1.0)

        out = torch.cat([rgb, alpha], dim=-1) if has_alpha else rgb

        # Upscale / resize
        if scale_by != 1.0:
            _, h, w, _ = out.shape
            new_w = max(1, round(w * scale_by))
            new_h = max(1, round(h * scale_by))
            samples = out.movedim(-1, 1)  # (B, H, W, C) -> (B, C, H, W)
            samples = comfy.utils.common_upscale(
                samples, new_w, new_h, upscale_method, "disabled"
            )
            out = samples.movedim(1, -1).clamp(0.0, 1.0)

        return (out,)
