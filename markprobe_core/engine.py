from pathlib import Path
import csv
import json
import math
import random
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler, Subset, random_split
from tqdm import tqdm

from .data import ImageFolderDataset, random_bits
from .model import MarkProbe


CHECKPOINT_FORMAT = "markprobe-core-v6-robust-training"
LEGACY_CHECKPOINT_FORMATS = {"markprobe-core-v3-stegastamp", "markprobe-core-v4-stegastamp-training", "markprobe-core-v5-stegastamp-training"}


def _device(device=None):
    return torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))


def quantize_8bit_ste(image):
    """Simulate PNG's 8-bit RGB quantization with a straight-through gradient."""
    clipped = image.clamp(0.0, 1.0)
    quantized = torch.round(clipped * 255.0) / 255.0
    return clipped + (quantized - clipped).detach()


_JPEG_LUMA = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99],
], dtype=torch.float32)
_JPEG_CHROMA = torch.tensor([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
], dtype=torch.float32)


def _dct_matrix(device, dtype):
    x = torch.arange(8, device=device, dtype=dtype)
    u = x[:, None]
    alpha = torch.full((8,), math.sqrt(2.0 / 8.0), device=device, dtype=dtype)
    alpha[0] = math.sqrt(1.0 / 8.0)
    return alpha[:, None] * torch.cos((2.0 * x[None, :] + 1.0) * u * (math.pi / 16.0))


def _symmetric_pad_right_bottom(image, pad_h, pad_w):
    """TensorFlow SYMMETRIC padding on the bottom and right edges."""
    height, width = image.shape[-2:]
    y = torch.arange(height + pad_h, device=image.device)
    x = torch.arange(width + pad_w, device=image.device)
    if height > 0:
        y = y.remainder(2 * height)
        y = torch.where(y < height, y, 2 * height - 1 - y)
    if width > 0:
        x = x.remainder(2 * width)
        x = torch.where(x < width, x, 2 * width - 1 - x)
    return image.index_select(-2, y).index_select(-1, x)


def _jpeg_blocks(component, quality, table):
    """Port StegaStamp's DCT quantization and round-only-near-zero function."""
    n, channels, height, width = component.shape
    blocks = component.contiguous().view(n, channels, height // 8, 8, width // 8, 8)
    blocks = blocks.permute(0, 1, 2, 4, 3, 5)
    dct = _dct_matrix(component.device, component.dtype)
    coeff = torch.matmul(torch.matmul(dct, blocks), dct.t())
    quality = float(quality)
    if quality < 50.0:
        scale = 5000.0 / max(1.0, quality)
    else:
        scale = 200.0 - 2.0 * quality
    factor = scale / 100.0 + 0.0001
    # StegaStamp stores the JPEG tables transposed relative to the conventional
    # display and uses round_only_at_0 rather than hard JPEG integer rounding.
    qtable = table.to(device=component.device, dtype=component.dtype).t() * factor
    scaled = coeff / qtable
    rounded = torch.where(scaled.abs() < 0.5, scaled.pow(3), scaled)
    restored = torch.matmul(torch.matmul(dct.t(), rounded * qtable), dct)
    restored = restored.permute(0, 1, 2, 4, 3, 5).contiguous()
    return restored.view(n, channels, height, width)


def jpeg_compress_decompress(image, quality=80, chroma_subsampling=True):
    """Differentiable PyTorch port of StegaStamp's JPEG simulation path.

    The transform, quantization tables, chroma subsampling, and smooth
    round-only-near-zero surrogate follow the reference; backend arithmetic can
    still differ slightly from TensorFlow.
    """
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("Expected RGB image tensor with shape NCHW")
    quality = float(max(1.0, min(100.0, quality)))
    n, _, height, width = image.shape
    pad_h = (-height) % 16
    pad_w = (-width) % 16
    source = _symmetric_pad_right_bottom(image, pad_h, pad_w) if (pad_h or pad_w) else image
    r, g, b = source[:, 0:1], source[:, 1:2], source[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    if chroma_subsampling:
        cb = F.avg_pool2d(cb, 2, 2)
        cr = F.avg_pool2d(cr, 2, 2)

    luma = y * 255.0 - 128.0
    cb = cb * 255.0
    cr = cr * 255.0
    y = _jpeg_blocks(luma, quality, _JPEG_LUMA)
    cb = _jpeg_blocks(cb, quality, _JPEG_CHROMA)
    cr = _jpeg_blocks(cr, quality, _JPEG_CHROMA)
    y = (y + 128.0) / 255.0
    cb = (cb + 128.0) / 255.0
    cr = (cr + 128.0) / 255.0
    if chroma_subsampling:
        # StegaStamp's JPEG helper upsamples 4:2:0 chroma by exact 2x repeat.
        cb = F.interpolate(cb, size=y.shape[-2:], mode="nearest")
        cr = F.interpolate(cr, size=y.shape[-2:], mode="nearest")
    chroma_center = 128.0 / 255.0
    cb, cr = cb - chroma_center, cr - chroma_center
    red = y + 1.402 * cr
    green = y - 0.344136 * cb - 0.714136 * cr
    blue = y + 1.772 * cb
    decoded = torch.cat([red, green, blue], dim=1).clamp(0.0, 1.0)
    return decoded[..., :height, :width]


def attack(image, strength=1.0):
    """PNG/JPEG, color, blur, noise and affine distortions for robust training."""
    strength = float(max(0.0, min(1.0, strength)))
    image = quantize_8bit_ste(image)
    if strength <= 0.0:
        return image

    n, _, h, w = image.shape
    if random.random() < 0.65 * strength:
        # Per-image brightness and contrast, similar to StegaStamp's transform net.
        contrast = torch.empty((n, 1, 1, 1), device=image.device).uniform_(
            1.0 - 0.22 * strength, 1.0 + 0.22 * strength
        )
        brightness = torch.empty((n, 1, 1, 1), device=image.device).uniform_(
            -0.08 * strength, 0.08 * strength
        )
        image = ((image - 0.5) * contrast + 0.5 + brightness).clamp(0.0, 1.0)
    if random.random() < 0.5 * strength:
        saturation = torch.empty((n, 1, 1, 1), device=image.device).uniform_(
            1.0 - 0.3 * strength, 1.0 + 0.3 * strength
        )
        luminance = (0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3])
        image = (luminance + saturation * (image - luminance)).clamp(0.0, 1.0)
    if random.random() < 0.3 * strength:
        sigma = random.uniform(0.0, 0.018 * strength)
        image = (image + torch.randn_like(image) * sigma).clamp(0.0, 1.0)
    if random.random() < 0.25 * strength:
        image = F.avg_pool2d(image, 3, stride=1, padding=1).clamp(0.0, 1.0)
    if random.random() < 0.55 * strength:
        angles = torch.empty(n, device=image.device).uniform_(-3.0, 3.0) * (math.pi / 180.0) * strength
        scales = torch.empty(n, device=image.device).uniform_(1.0 - 0.12 * strength, 1.0)
        tx = torch.empty(n, device=image.device).uniform_(-0.08, 0.08) * strength
        ty = torch.empty(n, device=image.device).uniform_(-0.08, 0.08) * strength
        theta = torch.zeros((n, 2, 3), device=image.device, dtype=image.dtype)
        theta[:, 0, 0] = scales * torch.cos(angles)
        theta[:, 0, 1] = -torch.sin(angles)
        theta[:, 1, 0] = torch.sin(angles)
        theta[:, 1, 1] = scales * torch.cos(angles)
        theta[:, 0, 2] = tx
        theta[:, 1, 2] = ty
        grid = F.affine_grid(theta, image.size(), align_corners=False)
        image = F.grid_sample(image, grid, mode="bilinear", padding_mode="border",
                              align_corners=False).clamp(0.0, 1.0)
    if random.random() < 0.65 * strength:
        quality = random.randint(max(35, int(70 - 15 * strength)), 98)
        image = jpeg_compress_decompress(image, quality=quality)
    return image.clamp(0.0, 1.0)


def _rgb_yuv(image):
    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    return torch.cat([y, u, v], dim=1)



def _linear_ramp(step, duration):
    if duration <= 0:
        return 1.0
    return min(1.0, max(0.0, float(step) / float(duration)))


def _homography_from_points(source, destination):
    """Batched 4-point projective transform mapping source pixels to destination."""
    batch = source.shape[0]
    x, y = source[..., 0], source[..., 1]
    u, v = destination[..., 0], destination[..., 1]
    zero = torch.zeros_like(x)
    one = torch.ones_like(x)
    row_u = torch.stack((x, y, one, zero, zero, zero, -u * x, -u * y), dim=-1)
    row_v = torch.stack((zero, zero, zero, x, y, one, -v * x, -v * y), dim=-1)
    matrix = torch.stack((row_u, row_v), dim=2).reshape(batch, 8, 8)
    rhs = torch.stack((u, v), dim=2).reshape(batch, 8, 1)
    coeffs = torch.linalg.solve(matrix, rhs).squeeze(-1)
    return torch.cat((coeffs, torch.ones((batch, 1), device=source.device, dtype=source.dtype)), dim=1).view(batch, 3, 3)


def _warp_projective(image, source_to_destination):
    """Differentiable projective warp, with source_to_destination in pixel coordinates."""
    n, _, height, width = image.shape
    ys, xs = torch.meshgrid(
        torch.arange(height, device=image.device, dtype=image.dtype),
        torch.arange(width, device=image.device, dtype=image.dtype), indexing="ij")
    output_points = torch.stack((xs, ys, torch.ones_like(xs)), dim=-1).view(1, height * width, 3)
    destination_to_source = torch.linalg.inv(source_to_destination)
    source_points = torch.bmm(output_points.expand(n, -1, -1), destination_to_source.transpose(1, 2))
    denom = source_points[..., 2].clamp(min=1e-6)
    source_x = source_points[..., 0] / denom
    source_y = source_points[..., 1] / denom
    if width > 1:
        source_x = source_x * (2.0 / (width - 1)) - 1.0
    else:
        source_x = torch.zeros_like(source_x)
    if height > 1:
        source_y = source_y * (2.0 / (height - 1)) - 1.0
    else:
        source_y = torch.zeros_like(source_y)
    grid = torch.stack((source_x, source_y), dim=-1).view(n, height, width, 2)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def _reference_geometry_embed(model, covers, messages, step, geometry_ramp_steps=10000,
                              max_perspective=0.10, border="white"):
    """StegaStamp's warp-cover -> embed -> inverse-warp-residual pipeline."""
    n, _, height, width = covers.shape
    amount = random.random() * max_perspective * _linear_ramp(step, geometry_ramp_steps)
    # Match StegaStamp's get_rand_transform_matrix: use d=floor(size * rnd)
    # and corners at (0, 0), (width, 0), (width, height), (0, height).
    # Its reference trainer uses square 400x400 images; keep the PyTorch port
    # well-defined for rectangular smoke tests by scaling each axis separately.
    dx = math.floor(width * amount)
    dy = math.floor(height * amount)
    base = torch.tensor([[0.0, 0.0], [float(width), 0.0],
                         [float(width), float(height)], [0.0, float(height)]],
                        device=covers.device, dtype=covers.dtype)
    jitter = torch.empty((n, 4, 2), device=covers.device, dtype=covers.dtype)
    jitter[..., 0].uniform_(-dx, dx)
    jitter[..., 1].uniform_(-dy, dy)
    source = base.unsqueeze(0) + jitter
    destination = base.unsqueeze(0).expand(n, -1, -1)
    homography = _homography_from_points(source, destination)

    # StegaStamp passes H (jittered-source -> regular-destination) to a
    # sampler whose matrix maps output -> input. grid_sample has the same
    # sampling convention, while _warp_projective accepts a forward
    # source -> destination matrix, so use H^{-1} for this first forward warp.
    inverse = torch.linalg.inv(homography)
    warped_cover = _warp_projective(covers, inverse)
    warped_mask = _warp_projective(torch.ones_like(covers[:, :1]), inverse)
    warped_cover = warped_cover + (1.0 - warped_mask) * covers
    residual_warped = model.encoder(warped_cover, messages)
    encoded_warped = warped_cover + residual_warped

    # The released implementation uses H^{-1} for the inverse sampler here;
    # as a forward transform this corresponds to H.
    residual = _warp_projective(residual_warped, homography)
    if border == "no_edge":
        # Reference no_edge mode restores only the residual and adds it
        # directly to the original cover, avoiding an inverse-warped cover.
        encoded = covers + residual
    elif border in {"black", "white", "random", "randomrgb", "image"}:
        # Reference crop modes inverse-warp the complete marked crop. The
        # sampler's zero padding is the black border; the remaining modes
        # explicitly composite their configured fill behind the warped mask.
        encoded = _warp_projective(encoded_warped, homography)
        if border != "black":
            restored_mask = _warp_projective(
                torch.ones_like(covers[:, :1]), homography
            ).clamp(0.0, 1.0)
            if border == "white":
                fill = torch.ones((1, 3, 1, 1), device=covers.device, dtype=covers.dtype)
            elif border == "random":
                # The reference samples one grayscale color shared across batch.
                fill = torch.rand((1, 1, 1, 1), device=covers.device, dtype=covers.dtype)
            elif border == "randomrgb":
                fill = torch.rand((1, 3, 1, 1), device=covers.device, dtype=covers.dtype)
            else:  # image
                fill = torch.roll(covers, shifts=1, dims=0)
            encoded = encoded + (1.0 - restored_mask) * fill
    else:
        raise ValueError("border must be one of: no_edge, black, random, randomrgb, image, white")
    return encoded, warped_cover, encoded_warped


def _random_blur_kernel(device, dtype):
    """StegaStamp 7x7 mixture: no blur, line blur, or Gaussian blur."""
    coords = torch.arange(7, device=device, dtype=dtype) - 3.0
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    manhattan = xx.abs() + yy.abs()
    sample = random.random()
    if sample < 0.25:
        sigma = random.uniform(1.0, 3.0)
        kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))
    elif sample < 0.5:
        theta = random.uniform(0.0, 2.0 * math.pi)
        sigma = random.uniform(0.25, 1.0)
        width = random.uniform(3.0, 0.5 * (7 - 1) + 0.1)
        distance = xx * math.cos(theta) + yy * math.sin(theta)
        kernel = torch.exp(-distance.square() / (2.0 * sigma * sigma)) * (manhattan < width)
    else:
        kernel = torch.zeros_like(xx)
        kernel[3, 3] = 1.0
    kernel = kernel / kernel.sum().clamp_min(1e-8)
    return kernel.view(1, 1, 7, 7).expand(3, 1, 7, 7).contiguous()


def stegastamp_transform(image, step, *, brightness=0.3, hue=0.1, noise=0.02,
                         saturation=1.0, contrast_low=0.5, contrast_high=1.5,
                         jpeg_quality=50, brightness_ramp=1000, hue_ramp=1000,
                         noise_ramp=1000, saturation_ramp=1000, contrast_ramp=1000,
                         jpeg_ramp=1000, use_jpeg=True):
    """Differentiable port of the released StegaStamp transform_net."""
    n = image.shape[0]
    kernel = _random_blur_kernel(image.device, image.dtype)
    image = F.conv2d(image, kernel, padding=3, groups=3)

    noise_std = random.random() * _linear_ramp(step, noise_ramp) * noise
    if noise_std:
        image = image + torch.randn_like(image) * noise_std
    image = image.clamp(0.0, 1.0)

    contrast_scale = _linear_ramp(step, contrast_ramp)
    low = 1.0 - (1.0 - contrast_low) * contrast_scale
    high = 1.0 + (contrast_high - 1.0) * contrast_scale
    contrast = torch.empty((n, 1, 1, 1), device=image.device, dtype=image.dtype).uniform_(low, high)
    image = image * contrast

    bri_scale = _linear_ramp(step, brightness_ramp) * brightness
    hue_scale = _linear_ramp(step, hue_ramp) * hue
    brightness_shift = torch.empty((n, 1, 1, 1), device=image.device, dtype=image.dtype).uniform_(-bri_scale, bri_scale)
    hue_shift = torch.empty((n, 3, 1, 1), device=image.device, dtype=image.dtype).uniform_(-hue_scale, hue_scale)
    image = (image + brightness_shift + hue_shift).clamp(0.0, 1.0)

    sat_scale = random.random() * _linear_ramp(step, saturation_ramp) * saturation
    luminance = (image * image.new_tensor([0.3, 0.6, 0.1]).view(1, 3, 1, 1)).sum(dim=1, keepdim=True)
    image = (1.0 - sat_scale) * image + sat_scale * luminance

    if use_jpeg:
        ramp = _linear_ramp(step, jpeg_ramp)
        sampled_quality = 100.0 - random.random() * ramp * (100.0 - jpeg_quality)
        image = jpeg_compress_decompress(image, quality=sampled_quality, chroma_subsampling=True)
    return image.clamp(0.0, 1.0)


def _edge_falloff(height, width, edge_gain, device, dtype):
    """StegaStamp's cosine attenuation map, strongest at image borders."""
    if edge_gain <= 0:
        # The reference starts from zero edge gain; returning ones doubles the YUV
        # delta and quadruples the loss before the edge-penalty ramp begins.
        return torch.zeros((1, 1, height, width), device=device, dtype=dtype)
    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)
    # Equivalent to the reference's cosine falloff over the outer quarter.
    dist_y = torch.minimum(y, (height - 1) - y).clamp_min(0)
    dist_x = torch.minimum(x, (width - 1) - x).clamp_min(0)
    fy = ((torch.cos(4.0 * math.pi * dist_y / max(1, height) + math.pi) + 1.0) * 0.5)
    fx = ((torch.cos(4.0 * math.pi * dist_x / max(1, width) + math.pi) + 1.0) * 0.5)
    fy = torch.where(dist_y < height / 4.0, fy, torch.ones_like(fy))
    fx = torch.where(dist_x < width / 4.0, fx, torch.ones_like(fx))
    return (1.0 - fy.view(1, 1, height, 1) * fx.view(1, 1, 1, width)) * edge_gain


def _stegastamp_yuv_loss(encoded, cover, yuv_scales=(1.0, 1.0, 1.0), edge_gain=0.0):
    delta = _rgb_yuv(encoded) - _rgb_yuv(cover)
    edge = _edge_falloff(*delta.shape[-2:], edge_gain, delta.device, delta.dtype)
    delta = delta * (1.0 + edge)
    per_channel = delta.square().mean(dim=(0, 2, 3))
    scales = torch.as_tensor(yuv_scales, device=delta.device, dtype=delta.dtype)
    return torch.sum(per_channel * scales)


class _StegaStampDiscriminator(torch.nn.Module):
    """Training-only discriminator from the StegaStamp release."""
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 8, 3, stride=2, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(8, 16, 3, stride=2, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 32, 3, stride=2, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 64, 3, stride=2, padding=1), torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(64, 1, 3, padding=1),
        )
        for layer in self.layers:
            if isinstance(layer, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, image):
        heatmap = self.layers(image - 0.5)
        return heatmap.mean(), heatmap


def _make_lpips(device):
    try:
        import lpips
    except ImportError as exc:
        raise RuntimeError("LPIPS is enabled for StegaStamp-aligned training, but the 'lpips' package is missing. Install requirements.txt or set lpips_weight=0 for a smoke test.") from exc
    metric = lpips.LPIPS(net="alex").to(device).eval()
    for parameter in metric.parameters():
        parameter.requires_grad_(False)
    return metric


def _perceptual_loss(metric, encoded, cover):
    # The PyTorch LPIPS implementation expects RGB in [-1, 1].
    return metric(encoded * 2.0 - 1.0, cover * 2.0 - 1.0).mean()

def _bit_accuracy(logits, messages):
    return (logits >= 0).eq(messages >= 0.5).float().mean().item()


@contextmanager
def _seeded_rng(seed):
    """Temporarily seed Python/PyTorch randomness without perturbing training RNG."""
    python_state = random.getstate()
    cpu_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        yield
    finally:
        random.setstate(python_state)
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


def crop_resize_attack(image, min_scale=0.65, max_scale=0.90):
    """Differentiable random crop-and-resize view, sampled independently per image."""
    if not (0.0 < min_scale <= max_scale <= 1.0):
        raise ValueError("crop scales must satisfy 0 < min_scale <= max_scale <= 1")
    batch = image.shape[0]
    scale = torch.empty((batch, 1, 1), device=image.device, dtype=image.dtype).uniform_(min_scale, max_scale)
    max_shift = 1.0 - scale[:, 0, 0]
    tx = (torch.rand(batch, device=image.device, dtype=image.dtype) * 2.0 - 1.0) * max_shift
    ty = (torch.rand(batch, device=image.device, dtype=image.dtype) * 2.0 - 1.0) * max_shift
    theta = torch.zeros((batch, 2, 3), device=image.device, dtype=image.dtype)
    theta[:, 0, 0] = scale[:, 0, 0]
    theta[:, 1, 1] = scale[:, 0, 0]
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, image.shape, align_corners=False)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=False).clamp(0.0, 1.0)


def _gradient_norms(model):
    grouped = {"encoder": [], "decoder": [], "other": []}
    finite = True
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        finite = finite and bool(torch.isfinite(grad).all().item())
        group = "encoder" if name.startswith("encoder.") else ("decoder" if name.startswith("decoder.") else "other")
        grouped[group].append(torch.linalg.vector_norm(grad))
    result = {}
    for name, norms in grouped.items():
        result[name] = float(torch.linalg.vector_norm(torch.stack(norms)).item()) if norms else 0.0
    result["total"] = math.sqrt(sum(result[name] ** 2 for name in grouped))
    result["finite"] = finite
    return result


def _ssim_batch(x, y):
    window = min(11, x.shape[-2], x.shape[-1])
    if window % 2 == 0:
        window -= 1
    if window < 3:
        return x.new_tensor(1.0) if torch.equal(x, y) else x.new_tensor(0.0)
    mu_x, mu_y = F.avg_pool2d(x, window, stride=1), F.avg_pool2d(y, window, stride=1)
    var_x = (F.avg_pool2d(x.square(), window, stride=1) - mu_x.square()).clamp_min(0)
    var_y = (F.avg_pool2d(y.square(), window, stride=1) - mu_y.square()).clamp_min(0)
    cov = F.avg_pool2d(x * y, window, stride=1) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_x * mu_y + c1) * (2 * cov + c2)) /
            ((mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2))).mean()


def _new_bit_meter():
    return {"bce_sum": 0.0, "correct": 0, "confidence_sum": 0.0, "bits": 0}


def _update_bit_meter(meter, logits, messages):
    probs = logits.detach().sigmoid()
    bits = messages.numel()
    meter["bce_sum"] += float(F.binary_cross_entropy_with_logits(logits, messages).item()) * bits
    meter["correct"] += int(((logits >= 0) == (messages >= 0.5)).sum().item())
    meter["confidence_sum"] += float(torch.maximum(probs, 1.0 - probs).sum().item())
    meter["bits"] += bits


def _finish_bit_meter(meter):
    count = max(1, meter["bits"])
    accuracy = meter["correct"] / count
    return {"bce": meter["bce_sum"] / count, "bit_accuracy": accuracy,
            "ber": 1.0 - accuracy, "mean_confidence": meter["confidence_sum"] / count,
            "bits": meter["bits"]}


def _flatten_numeric(value, prefix=""):
    """Return scalar numeric leaves from an epoch record for curve export."""
    flattened = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = "{}.{}".format(prefix, key) if prefix else str(key)
            flattened.update(_flatten_numeric(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        flattened[prefix] = value
    return flattened


def _write_curve_files(curves_path, records):
    """Write column-oriented JSON and CSV files without rendering a plot."""
    curves_path = Path(curves_path)
    curves_path.parent.mkdir(parents=True, exist_ok=True)
    flattened_records = [_flatten_numeric(record) for record in records]
    keys = sorted({key for record in flattened_records for key in record})
    columns = {key: [record.get(key) for record in flattened_records] for key in keys}
    length = len(flattened_records)
    curves_path.write_text(json.dumps({"format": "markprobe-curves-v1", **columns},
                                      ensure_ascii=False, allow_nan=False, indent=2),
                           encoding="utf-8")
    csv_path = curves_path.with_suffix(".csv")
    fieldnames = list(columns)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row_index in range(length):
            writer.writerow({key: columns[key][row_index] for key in fieldnames})


def _load_metric_records(metrics_file):
    if not metrics_file.is_file():
        return []
    records = []
    for line in metrics_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _subset_source_index(dataset, index):
    """Resolve a nested Subset index back to the original dataset."""
    while isinstance(dataset, Subset):
        index = dataset.indices[index]
        dataset = dataset.dataset
    return dataset, int(index)


def _tensor_to_pil(image):
    array = (image.detach().clamp(0.0, 1.0)[0].permute(1, 2, 0).cpu().numpy() * 255.0).round()
    return Image.fromarray(array.astype(np.uint8), "RGB")


def _residual_to_pil(residual, scale=4.0):
    """Visualize a signed image residual with neutral gray representing zero."""
    # A direct residual-to-PNG conversion would clip negative values to black.
    # Mapping zero to mid-gray keeps both brightening and darkening visible.
    visual = (residual.detach() * scale + 0.5).clamp(0.0, 1.0)
    return _tensor_to_pil(visual)


def _export_epoch_inference(model, dataset, eval_set, epoch, global_step, output_dir,
                            message_bits, seed, device, sample_count):
    """Save deterministic cover/encoded/decode artifacts for the current epoch."""
    if sample_count <= 0:
        return {"enabled": False, "sample_count": 0}
    source_set = eval_set if len(eval_set) else dataset
    count = min(int(sample_count), len(source_set))
    epoch_dir = Path(output_dir) / "epoch_{:04d}".format(epoch)
    epoch_dir.mkdir(parents=True, exist_ok=True)
    records = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for sample_index in range(count):
            image = source_set[sample_index].unsqueeze(0).to(device)
            _, source_index = _subset_source_index(source_set, sample_index)
            source_dataset, _ = _subset_source_index(source_set, sample_index)
            source_name = (str(source_dataset.paths[source_index].name)
                           if hasattr(source_dataset, "paths") else "sample_{:04d}".format(sample_index))
            generator = torch.Generator().manual_seed(seed + 730001 + source_index)
            message = torch.randint(0, 2, (1, message_bits), generator=generator).float().to(device)
            encoded = model.encode(image, message).clamp(0.0, 1.0)
            encoded_png = quantize_8bit_ste(encoded)
            probabilities = model.decode(encoded_png).sigmoid()[0].detach().cpu().tolist()
            decoded_bits = [int(value >= 0.5) for value in probabilities]
            expected_bits = [int(value) for value in message[0].cpu().tolist()]
            correct = sum(expected == actual for expected, actual in zip(expected_bits, decoded_bits))

            sample_dir = epoch_dir / "sample_{:04d}".format(sample_index)
            sample_dir.mkdir(parents=True, exist_ok=True)
            cover_path = sample_dir / "cover.png"
            encoded_path = sample_dir / "encoded.png"
            residual_path = sample_dir / "residual.png"
            residual = encoded_png - image
            residual_abs = residual.abs()
            residual_stats = {
                "mean_signed": float(residual.mean().item()),
                "mean_abs": float(residual_abs.mean().item()),
                "p95_abs": float(torch.quantile(residual_abs.flatten(), 0.95).item()),
                "max_abs": float(residual_abs.max().item()),
                "visual_scale": 4.0,
                "visual_zero_level": 0.5,
            }
            _tensor_to_pil(image).save(cover_path)
            _tensor_to_pil(encoded_png).save(encoded_path)
            _residual_to_pil(residual).save(residual_path)
            decoded_record = {
                "epoch": int(epoch), "global_step": int(global_step),
                "source_name": source_name, "message_bits": expected_bits,
                "decoded_bits": decoded_bits, "probabilities": probabilities,
                "bit_accuracy": correct / max(1, message_bits),
                "ber": 1.0 - correct / max(1, message_bits),
                "cover": cover_path.name, "encoded": encoded_path.name,
                "residual": residual_path.name, "residual_stats": residual_stats,
            }
            (sample_dir / "decoded.json").write_text(
                json.dumps(decoded_record, ensure_ascii=False, indent=2), encoding="utf-8")
            (sample_dir / "message.txt").write_text("".join(map(str, expected_bits)), encoding="utf-8")
            (sample_dir / "decoded.txt").write_text("".join(map(str, decoded_bits)), encoding="utf-8")
            records.append({**decoded_record, "sample": sample_index,
                            "directory": str(sample_dir.relative_to(epoch_dir))})
    if was_training:
        model.train()
    summary = {"enabled": True, "epoch": int(epoch), "global_step": int(global_step),
               "sample_count": count, "directory": str(epoch_dir), "samples": records}
    (epoch_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"enabled": True, "sample_count": count, "directory": str(epoch_dir),
            "mean_bit_accuracy": sum(item["bit_accuracy"] for item in records) / max(1, count),
            "mean_ber": sum(item["ber"] for item in records) / max(1, count)}


def train_model(data_dir, output, image_size=400, message_bits=96, width=32,
                epochs=10, batch_size=4, lr=1e-4, image_weight=1.5,
                val_fraction=0.1, workers=0, seed=2026, device=None, limit=None,
                max_steps=None,
                progress=True, progress_interval=5.0, resume=None, use_stn=True,
                residual_scale=0.1, no_im_loss_steps=1500,
                l2_loss_ramp_steps=15000, lpips_loss_ramp_steps=15000,
                geometry_ramp_steps=10000, jpeg_ramp_steps=1000,
                lpips_weight=1.0, yuv_scales=(1.0, 1.0, 1.0),
                edge_gain=10.0, edge_delay_steps=80000, edge_ramp_steps=10000,
                secret_loss_scale=1.0, secret_loss_ramp_steps=1,
                gan_weight=1.0, gan_ramp_steps=15000,
                use_gan=True, use_jpeg=True, border="white",
                dual_branch=True, crop_scale_min=0.65, crop_scale_max=0.90,
                adaptive_image_loss=False, adaptive_msg_threshold=0.30,
                adaptive_ber_threshold=0.25, adaptive_patience=2,
                adaptive_ramp_steps=15000, validation_samples=256,
                gradient_log_interval=100, metrics_path=None, curves_path=None,
                epoch_export_dir=None, epoch_export_samples=4):
    """Train the unchanged StegaStamp-topology model.

    ``dual_branch`` and ``adaptive_image_loss`` are optional Mark-Probe training
    extensions. They leave encoder/decoder layers and activations untouched.
    Epoch metrics are written as JSON Lines next to the checkpoint by default.
    Numeric metrics are also exported as column-oriented JSON/CSV curves, and a
    small deterministic inference set is saved after every completed epoch.
    """
    l2_loss_scale = float(image_weight) if image_weight is not None else 2.0
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must be in [0, 1)")
    if validation_samples is not None:
        validation_samples = int(validation_samples)
    if int(epoch_export_samples) < 0:
        raise ValueError("epoch_export_samples must be non-negative")
    if max_steps is not None and int(max_steps) < 1:
        raise ValueError("max_steps must be a positive integer or None")
    if max_steps is not None:
        max_steps = int(max_steps)
    if adaptive_patience < 1 or (validation_samples is not None and validation_samples < 1) or gradient_log_interval < 1:
        raise ValueError("adaptive_patience, validation_samples (when set), and gradient_log_interval must be positive")
    if not (0.0 <= adaptive_ber_threshold <= 1.0) or adaptive_msg_threshold < 0.0:
        raise ValueError("adaptive BER threshold must be in [0,1] and BCE threshold must be non-negative")
    if adaptive_ramp_steps < 1:
        raise ValueError("adaptive_ramp_steps must be positive")
    if not (0.0 < crop_scale_min <= crop_scale_max <= 1.0):
        raise ValueError("crop scales must satisfy 0 < min <= max <= 1")

    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = _device(device)
    dataset = ImageFolderDataset(data_dir, image_size, limit=limit)
    if len(dataset) < 1:
        raise ValueError("Training dataset is empty")
    n_val = max(1, int(len(dataset) * val_fraction)) if val_fraction > 0 and len(dataset) > 1 else 0
    if adaptive_image_loss and n_val == 0:
        raise ValueError("adaptive_image_loss requires a held-out validation split (val_fraction > 0)")
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)
    if n_val and validation_samples is not None:
        val_set = Subset(val_set, range(min(len(val_set), int(validation_samples))))
    # Match the reference sampler: sample with replacement and always use full batches.
    reference_steps_per_epoch = len(train_set) // batch_size + 1
    train_sampler = RandomSampler(train_set, replacement=True,
                                  num_samples=reference_steps_per_epoch * batch_size)
    train_loader = DataLoader(train_set, batch_size=batch_size, sampler=train_sampler,
                              drop_last=False, num_workers=workers,
                              pin_memory=device.type == "cuda", persistent_workers=workers > 0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False,
                            num_workers=workers, pin_memory=device.type == "cuda",
                            persistent_workers=workers > 0) if n_val else []
    print("dataset loaded: {} images from {} (train={}, validation={}, validation_eval={}, device={})".format(
        len(dataset), dataset.root.resolve(), n_train, n_val, len(val_set) if n_val else 0, device))

    model = MarkProbe(message_bits, width, use_stn=use_stn,
                      residual_scale=residual_scale, image_size=image_size).to(device)
    secret_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    full_optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    discriminator = _StegaStampDiscriminator().to(device) if use_gan else None
    discriminator_optimizer = (torch.optim.RMSprop(discriminator.parameters(), lr=1e-5,
                               alpha=0.9, eps=1e-10) if use_gan else None)
    perceptual = _make_lpips(device) if lpips_weight > 0 else None
    start_epoch, global_step, best = 0, 0, float("inf")
    adaptive_ready_epochs, adaptive_start_step = 0, None
    if resume:
        state = _read_checkpoint(resume, device)
        if state.get("format") != CHECKPOINT_FORMAT:
            raise ValueError("This checkpoint has weights but not compatible v6 training state; start a new run or use it for inference.")
        expected = (message_bits, width, bool(use_stn), float(residual_scale), int(image_size))
        found = (state["message_bits"], state["width"], state["use_stn"],
                 state["residual_scale"], state["image_size"])
        if expected != found:
            raise ValueError(
                "Resume checkpoint architecture mismatch: requested {} but checkpoint has {}. "
                "The message-bit count determines encoder/decoder layer dimensions and cannot "
                "be converted when resuming. Set message_bits to {} to continue this checkpoint, "
                "or start a fresh run with the standardized 96-bit configuration."
                .format(expected, found, state["message_bits"])
            )
        model.load_state_dict(state["model"])
        train_state = state.get("training_state", {})
        if train_state.get("secret_optimizer"):
            secret_optimizer.load_state_dict(train_state["secret_optimizer"])
        if train_state.get("full_optimizer"):
            full_optimizer.load_state_dict(train_state["full_optimizer"])
        if use_gan and train_state.get("discriminator"):
            discriminator.load_state_dict(train_state["discriminator"])
            discriminator_optimizer.load_state_dict(train_state["discriminator_optimizer"])
        adaptive_ready_epochs = int(train_state.get("adaptive_ready_epochs", 0))
        adaptive_start_step = train_state.get("adaptive_start_step")
        start_epoch = int(state.get("epoch", 0))
        global_step = int(state.get("global_step", 0))
        best = float(state.get("best_val_loss", best))
        _restore_rng_state(state)
        print("resumed checkpoint {} at epoch {}, step {}".format(resume, start_epoch, global_step))

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    latest_output = output.with_name(output.stem + ".last" + output.suffix)
    metrics_file = Path(metrics_path) if metrics_path else output.with_name(output.stem + ".metrics.jsonl")
    curves_file = Path(curves_path) if curves_path else output.with_name(output.stem + ".curves.json")
    epoch_export_root = (Path(epoch_export_dir) if epoch_export_dir is not None
                         else output.with_name(output.stem + ".epochs"))
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    if not resume:
        metrics_file.write_text("", encoding="utf-8")
        curve_records = []
    else:
        curve_records = _load_metric_records(metrics_file)
    current_train_config = {
        "batch_size": int(batch_size), "lr": float(lr), "seed": int(seed),
        "val_fraction": float(val_fraction),
        "no_im_loss_steps": int(no_im_loss_steps), "l2_loss_scale": l2_loss_scale,
        "l2_loss_ramp_steps": int(l2_loss_ramp_steps),
        "lpips_loss_ramp_steps": int(lpips_loss_ramp_steps),
        "geometry_ramp_steps": int(geometry_ramp_steps), "jpeg_ramp_steps": int(jpeg_ramp_steps),
        "lpips_weight": float(lpips_weight), "secret_loss_scale": float(secret_loss_scale),
        "secret_loss_ramp_steps": int(secret_loss_ramp_steps),
        "yuv_scales": tuple(float(value) for value in yuv_scales),
        "edge_gain": float(edge_gain), "edge_delay_steps": int(edge_delay_steps),
        "edge_ramp_steps": int(edge_ramp_steps), "gan_weight": float(gan_weight),
        "gan_ramp_steps": int(gan_ramp_steps), "use_gan": bool(use_gan),
        "use_jpeg": bool(use_jpeg), "border": str(border),
        "dual_branch": bool(dual_branch), "crop_scale_min": float(crop_scale_min),
        "crop_scale_max": float(crop_scale_max),
        "adaptive_image_loss": bool(adaptive_image_loss),
        "adaptive_msg_threshold": float(adaptive_msg_threshold),
        "adaptive_ber_threshold": float(adaptive_ber_threshold),
        "adaptive_patience": int(adaptive_patience), "adaptive_ramp_steps": int(adaptive_ramp_steps),
        "validation_samples": int(validation_samples) if validation_samples is not None else None,
    }
    if resume:
        saved_config = state.get("training_state", {}).get("train_config", {})
        if saved_config and saved_config != current_train_config:
            changed = sorted(key for key in set(saved_config) | set(current_train_config)
                             if saved_config.get(key) != current_train_config.get(key))
            raise ValueError(
                "Resume training objective/schedule mismatch in: {}. Resume with the same "
                "training settings that created the checkpoint (including dual-branch, adaptive "
                "loss, ramps, validation sampling, GAN/JPEG, crop, and border settings). "
                "The CLI does not expose every notebook setting; use the notebook to resume a "
                "notebook-created checkpoint, or start a fresh CLI run."
                .format(", ".join(changed))
            )

    steps_per_epoch = max(1, len(train_loader))
    if max_steps is None:
        total_steps = max(1, epochs * (len(train_loader) + len(val_loader)))
        progress_initial = start_epoch * (len(train_loader) + len(val_loader))
    else:
        remaining_train_steps = max(0, max_steps - global_step)
        planned_train_steps = min(remaining_train_steps, max(0, epochs - start_epoch) * steps_per_epoch)
        planned_epochs = math.ceil(planned_train_steps / steps_per_epoch) if planned_train_steps else 0
        total_steps = planned_train_steps + planned_epochs * len(val_loader)
        progress_initial = 0
    progress_bar = tqdm(total=total_steps, initial=progress_initial,
                        desc="MarkProbe train/validate", unit="batch", disable=not progress,
                        mininterval=progress_interval, miniters=1, dynamic_ncols=True)
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            if discriminator is not None:
                discriminator.train()
            train_sums = {key: 0.0 for key in ("loss", "photo_bce", "crop_bce", "yuv", "lpips", "gan", "quality_ramp", "weighted_message", "weighted_yuv", "weighted_lpips", "weighted_gan")}
            train_correct = train_bits = train_batches = 0
            gradient_sums = {"total": 0.0, "encoder": 0.0, "decoder": 0.0, "other": 0.0, "finite": True}
            gradient_samples = 0

            for images in train_loader:
                if max_steps is not None and global_step >= max_steps:
                    break
                images = images.to(device, non_blocking=device.type == "cuda")
                messages = random_bits(images.shape[0], message_bits, device)
                current_step = global_step
                encoded, warped_cover, encoded_warped = _reference_geometry_embed(
                    model, images, messages, current_step,
                  geometry_ramp_steps=geometry_ramp_steps, border=border)

                photo_view = stegastamp_transform(encoded, current_step,
                    jpeg_ramp=jpeg_ramp_steps, use_jpeg=use_jpeg)
                photo_logits = model.decode(photo_view)
                photo_bce = F.binary_cross_entropy_with_logits(photo_logits, messages)
                if dual_branch:
                    crop_view = crop_resize_attack(encoded, crop_scale_min, crop_scale_max)
                    crop_view = stegastamp_transform(crop_view, current_step,
                        jpeg_ramp=jpeg_ramp_steps, use_jpeg=use_jpeg)
                    crop_logits = model.decode(crop_view)
                    crop_bce = F.binary_cross_entropy_with_logits(crop_logits, messages)
                    secret_loss = 0.5 * (photo_bce + crop_bce)
                else:
                    crop_logits = None
                    crop_bce = photo_bce.new_zeros(())
                    secret_loss = photo_bce

                yuv_loss = _stegastamp_yuv_loss(encoded, images, yuv_scales,
                    edge_gain=(edge_gain * _linear_ramp(current_step - edge_delay_steps, edge_ramp_steps)
                               if current_step > edge_delay_steps else 0.0))
                lpips_loss = (secret_loss.new_zeros(()) if perceptual is None else
                            _perceptual_loss(perceptual, encoded, images))
                if adaptive_image_loss:
                    quality_ramp = (0.0 if adaptive_start_step is None else
                                    _linear_ramp(current_step - adaptive_start_step, adaptive_ramp_steps))
                else:
                    quality_ramp = 1.0
                l2_scale = l2_loss_scale * _linear_ramp(current_step, l2_loss_ramp_steps) * quality_ramp
                lpips_scale = lpips_weight * _linear_ramp(current_step, lpips_loss_ramp_steps) * quality_ramp
                secret_scale = secret_loss_scale * _linear_ramp(current_step, secret_loss_ramp_steps)
                gan_scale = gan_weight * _linear_ramp(current_step, gan_ramp_steps) * quality_ramp
                warmup_only = (current_step < no_im_loss_steps or
                               (adaptive_image_loss and adaptive_start_step is None))

                if border == "no_edge":
                    gan_real_image, gan_fake_image = images, encoded
                else:
                    gan_real_image, gan_fake_image = warped_cover, encoded_warped
                if discriminator is not None:
                    for parameter in discriminator.parameters():
                        parameter.requires_grad_(False)
                    fake_score, _ = discriminator(gan_fake_image)
                else:
                    fake_score = secret_loss.new_zeros(())
                if warmup_only:
                    loss, optimizer = secret_loss, secret_optimizer
                    weighted_message, weighted_yuv = secret_loss, secret_loss.new_zeros(())
                    weighted_lpips, weighted_gan = secret_loss.new_zeros(()), secret_loss.new_zeros(())
                else:
                    weighted_message = secret_scale * secret_loss
                    weighted_yuv = l2_scale * yuv_loss
                    weighted_lpips = lpips_scale * lpips_loss
                    weighted_gan = gan_scale * fake_score
                    loss = weighted_message + weighted_yuv + weighted_lpips + weighted_gan
                    optimizer = full_optimizer

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if ((global_step + 1) % gradient_log_interval == 0) or gradient_samples == 0:
                    grad_stats = _gradient_norms(model)
                    if not grad_stats["finite"]:
                        raise FloatingPointError("Non-finite model gradient detected at optimizer step {}".format(global_step + 1))
                    for key in ("total", "encoder", "decoder", "other"):
                        gradient_sums[key] += grad_stats[key]
                    gradient_samples += 1
                optimizer.step()

                if discriminator is not None:
                    for parameter in discriminator.parameters():
                        parameter.requires_grad_(True)
                    if not warmup_only:
                        discriminator_optimizer.zero_grad(set_to_none=True)
                        real_score, _ = discriminator(gan_real_image.detach())
                        fake_score_detached, _ = discriminator(gan_fake_image.detach())
                        (real_score - fake_score_detached).backward()
                        for parameter in discriminator.parameters():
                            if parameter.grad is not None:
                                parameter.grad.clamp_(-0.25, 0.25)
                        discriminator_optimizer.step()
                        with torch.no_grad():
                            for parameter in discriminator.parameters():
                                parameter.clamp_(-0.01, 0.01)

                train_sums["loss"] += float(loss.detach().item())
                train_sums["photo_bce"] += float(photo_bce.detach().item())
                train_sums["crop_bce"] += float(crop_bce.detach().item())
                train_sums["yuv"] += float(yuv_loss.detach().item())
                train_sums["lpips"] += float(lpips_loss.detach().item())
                train_sums["gan"] += float(fake_score.detach().item())
                train_sums["quality_ramp"] += quality_ramp
                train_sums["weighted_message"] += float(weighted_message.detach().item())
                train_sums["weighted_yuv"] += float(weighted_yuv.detach().item())
                train_sums["weighted_lpips"] += float(weighted_lpips.detach().item())
                train_sums["weighted_gan"] += float(weighted_gan.detach().item())
                train_correct += int(((photo_logits.detach() >= 0) == (messages >= 0.5)).sum().item())
                if crop_logits is not None:
                    train_correct += int(((crop_logits.detach() >= 0) == (messages >= 0.5)).sum().item())
                    train_bits += messages.numel()
                train_bits += messages.numel()
                train_batches += 1
                global_step += 1
                progress_bar.update(1)
                denom = max(1, train_batches)
                progress_bar.set_postfix(epoch="{}/{}".format(epoch + 1, epochs), phase="train",
                    msg="{:.4f}".format(train_sums["photo_bce"] / denom),
                    crop="{:.4f}".format(train_sums["crop_bce"] / denom),
                    bit_acc="{:.1%}".format(train_correct / max(1, train_bits)),
                    q_ramp="{:.2f}".format(train_sums["quality_ramp"] / denom))

            model.eval()
            if discriminator is not None:
                discriminator.eval()
            scenario_meters = {name: _new_bit_meter() for name in
                               ("clean_png", "photo", "crop", "crop_photo", "jpeg75")}
            # Report clean stego quality separately from the full geometry path.
            # The former is the direct cover -> embedded-image distortion; the
            # latter also includes the reference perspective/border pipeline.
            val_image_sums = {
                "mse": 0.0, "mae": 0.0, "p95_abs_delta": 0.0, "ssim": 0.0,
                "geometry_mse": 0.0, "geometry_mae": 0.0,
                "geometry_p95_abs_delta": 0.0, "geometry_ssim": 0.0,
                "lpips": 0.0, "geometry_lpips": 0.0,
                "pixels": 0, "images": 0,
            }
            val_yuv_sum = 0.0
            val_batches = 0
            val_messages = None
            if n_val:
                val_generator = torch.Generator().manual_seed(seed + 918273)
                val_messages = torch.randint(0, 2, (len(val_set), message_bits),
                                             generator=val_generator).float()
            with torch.no_grad():
                val_offset = 0
                for batch_index, images in enumerate(val_loader):
                    images = images.to(device, non_blocking=device.type == "cuda")
                    messages = val_messages[val_offset:val_offset + images.shape[0]].to(device)
                    val_offset += images.shape[0]
                    # Keep validation transforms fixed across epochs for meaningful comparisons.
                    with _seeded_rng(seed + 300001 + batch_index):
                        encoded, _, _ = _reference_geometry_embed(
                            model, images, messages, global_step,
                          geometry_ramp_steps=geometry_ramp_steps, border=border)
                    # This is the primary visual-quality metric: the direct
                    # embedded image, before geometry or decoder attacks.
                    clean_encoded = model.encode(images, messages)
                    clean_png = quantize_8bit_ste(clean_encoded)
                    png = quantize_8bit_ste(encoded)
                    with _seeded_rng(seed + 400001 + batch_index):
                        photo = stegastamp_transform(encoded, global_step,
                            jpeg_ramp=jpeg_ramp_steps, use_jpeg=use_jpeg)
                    with _seeded_rng(seed + 500001 + batch_index):
                        crop = crop_resize_attack(encoded, crop_scale_min, crop_scale_max)
                    with _seeded_rng(seed + 600001 + batch_index):
                        crop_photo = stegastamp_transform(crop, global_step,
                            jpeg_ramp=jpeg_ramp_steps, use_jpeg=use_jpeg)
                    with _seeded_rng(seed + 700001 + batch_index):
                        jpeg75 = jpeg_compress_decompress(png, quality=75, chroma_subsampling=True)
                    views = {"clean_png": png, "photo": photo, "crop": crop,
                             "crop_photo": crop_photo, "jpeg75": jpeg75}
                    for name, view in views.items():
                        _update_bit_meter(scenario_meters[name], model.decode(view), messages)
                    val_yuv_sum += float(_stegastamp_yuv_loss(encoded, images, yuv_scales, edge_gain=0.0).item())
                    clean_delta = (clean_png - images).square()
                    geometry_delta = (png - images).square()
                    clean_abs_delta = (clean_png - images).abs()
                    geometry_abs_delta = (png - images).abs()
                    val_image_sums["mse"] += float(clean_delta.sum().item())
                    val_image_sums["mae"] += float(clean_abs_delta.sum().item())
                    val_image_sums["p95_abs_delta"] += float(
                            torch.quantile(clean_abs_delta.flatten(1), 0.95, dim=1).mean().item()
                        ) * images.shape[0]
                    val_image_sums["geometry_mse"] += float(geometry_delta.sum().item())
                    val_image_sums["geometry_mae"] += float(geometry_abs_delta.sum().item())
                    val_image_sums["geometry_p95_abs_delta"] += float(
                            torch.quantile(geometry_abs_delta.flatten(1), 0.95, dim=1).mean().item()
                        ) * images.shape[0]
                    if perceptual is not None:
                        val_image_sums["lpips"] += float(
                            _perceptual_loss(perceptual, clean_png, images).item()
                        ) * images.shape[0]
                        val_image_sums["geometry_lpips"] += float(
                            _perceptual_loss(perceptual, png, images).item()
                        ) * images.shape[0]
                    val_image_sums["pixels"] += clean_delta.numel()
                    val_image_sums["ssim"] += float(_ssim_batch(clean_png, images).item()) * images.shape[0]
                    val_image_sums["geometry_ssim"] += float(_ssim_batch(png, images).item()) * images.shape[0]
                    val_image_sums["images"] += images.shape[0]
                    val_batches += 1
                    progress_bar.update(1)
                    progress_bar.set_postfix(epoch="{}/{}".format(epoch + 1, epochs), phase="validate",
                        photo_acc="{:.1%}".format(_finish_bit_meter(scenario_meters["photo"])["bit_accuracy"]),
                        crop_acc="{:.1%}".format(_finish_bit_meter(scenario_meters["crop_photo"])["bit_accuracy"]))

            train_metrics = {key: value / max(1, train_batches) for key, value in train_sums.items()}
            train_metrics["bit_accuracy_photo_crop_mean"] = train_correct / max(1, train_bits)
            train_metrics["gradient_norms"] = {key: gradient_sums[key] / max(1, gradient_samples)
                                                for key in ("total", "encoder", "decoder", "other")}
            train_metrics["gradient_norms"]["finite"] = gradient_sums["finite"]
            train_metrics["gradient_norms"]["sample_count"] = gradient_samples
            validation_metrics = {}
            is_best = False
            if val_batches:
                scenario_metrics = {name: _finish_bit_meter(meter) for name, meter in scenario_meters.items()}
                photo_metric = scenario_metrics["photo"]
                crop_photo_metric = scenario_metrics["crop_photo"]
                val_message_loss = (0.5 * (photo_metric["bce"] + crop_photo_metric["bce"])
                                    if dual_branch else photo_metric["bce"])
                val_image_loss = val_yuv_sum / val_batches
                val_mse = val_image_sums["mse"] / max(1, val_image_sums["pixels"])
                val_mae = val_image_sums["mae"] / max(1, val_image_sums["pixels"])
                val_p95_abs_delta = val_image_sums["p95_abs_delta"] / max(1, val_image_sums["images"])
                val_psnr = None if val_mse == 0 else 10.0 * math.log10(1.0 / val_mse)
                val_ssim = val_image_sums["ssim"] / max(1, val_image_sums["images"])
                geometry_mse = val_image_sums["geometry_mse"] / max(1, val_image_sums["pixels"])
                geometry_mae = val_image_sums["geometry_mae"] / max(1, val_image_sums["pixels"])
                geometry_p95_abs_delta = val_image_sums["geometry_p95_abs_delta"] / max(1, val_image_sums["images"])
                geometry_psnr = (None if geometry_mse == 0 else
                                 10.0 * math.log10(1.0 / geometry_mse))
                geometry_ssim = val_image_sums["geometry_ssim"] / max(1, val_image_sums["images"])
                val_lpips = (None if perceptual is None else
                    val_image_sums["lpips"] / max(1, val_image_sums["images"]))
                geometry_lpips = (None if perceptual is None else
                    val_image_sums["geometry_lpips"] / max(1, val_image_sums["images"]))
                validation_metrics = {"scenarios": scenario_metrics,
                    "image_quality": {
                        # Backward-compatible aliases: these are now the
                        # direct clean stego-image metrics.
                         "mse": val_mse, "mae": val_mae, "p95_abs_delta": val_p95_abs_delta,
                         "psnr_db": val_psnr, "ssim": val_ssim, "lpips": val_lpips,
                         "clean_png": {"mse": val_mse, "mae": val_mae,
                                       "p95_abs_delta": val_p95_abs_delta,
                                       "psnr_db": val_psnr, "ssim": val_ssim,
                                       "lpips": val_lpips},
                         "geometry_png": {"mse": geometry_mse, "mae": geometry_mae,
                                          "p95_abs_delta": geometry_p95_abs_delta,
                                          "psnr_db": geometry_psnr, "ssim": geometry_ssim,
                                          "lpips": geometry_lpips},
                    },
                    "yuv_loss": val_image_loss}
                criterion_metrics = [photo_metric, crop_photo_metric] if dual_branch else [photo_metric]
                ready = all(m["bce"] <= adaptive_msg_threshold and m["ber"] <= adaptive_ber_threshold
                            for m in criterion_metrics)
                if adaptive_image_loss and adaptive_start_step is None:
                    adaptive_ready_epochs = adaptive_ready_epochs + 1 if ready else 0
                    if adaptive_ready_epochs >= adaptive_patience and global_step >= no_im_loss_steps:
                        adaptive_start_step = global_step
                        print("adaptive image-loss ramp unlocked at step {} after {} consecutive validation passes".format(
                            adaptive_start_step, adaptive_ready_epochs))
                adaptive_factor = (1.0 if not adaptive_image_loss else
                    (0.0 if adaptive_start_step is None else _linear_ramp(global_step - adaptive_start_step, adaptive_ramp_steps)))
                val_loss = val_message_loss + adaptive_factor * l2_loss_scale * _linear_ramp(global_step, l2_loss_ramp_steps) * val_image_loss
                is_best = val_loss < best
                print("epoch {}/{} | train photo/crop BCE {:.4f}/{:.4f}, BER {:.2%}, grad {:.3g} | train img yuv/lpips/gan {:.4f}/{:.4f}/{:.4f} -> weighted {:.4f}/{:.4f}/{:.4f} | val photo {:.2%}/{:.4f}, crop+photo {:.2%}/{:.4f}, clean PNG {:.2%}, JPEG75 {:.2%} | clean P {:.2f} dB S {:.5f} L {} MAE {:.5f} p95 {:.5f} | geom P {:.2f} dB S {:.5f} L {} MAE {:.5f} p95 {:.5f}, q-ramp {:.2f}".format(
                    epoch + 1, epochs, train_metrics["photo_bce"], train_metrics["crop_bce"],
                    1.0 - train_metrics["bit_accuracy_photo_crop_mean"],
                    train_metrics["gradient_norms"]["total"],
                    train_metrics["yuv"], train_metrics["lpips"], train_metrics["gan"],
                    train_metrics["weighted_yuv"], train_metrics["weighted_lpips"], train_metrics["weighted_gan"],
                    photo_metric["bit_accuracy"], photo_metric["bce"],
                    crop_photo_metric["bit_accuracy"], crop_photo_metric["bce"],
                    scenario_metrics["clean_png"]["bit_accuracy"], scenario_metrics["jpeg75"]["bit_accuracy"],
                    (val_psnr if val_psnr is not None else float("inf")),
                    val_ssim, "n/a" if val_lpips is None else "{:.5f}".format(val_lpips), val_mae,
                    val_p95_abs_delta,
                    (geometry_psnr if geometry_psnr is not None else float("inf")), geometry_ssim,
                    "n/a" if geometry_lpips is None else "{:.5f}".format(geometry_lpips), geometry_mae,
                    geometry_p95_abs_delta, adaptive_factor))
            else:
                print("epoch {}/{} | train photo/crop BCE {:.4f}/{:.4f}, bit accuracy {:.2%} | validation disabled".format(
                    epoch + 1, epochs, train_metrics["photo_bce"], train_metrics["crop_bce"],
                    train_metrics["bit_accuracy_photo_crop_mean"]))

            epoch_inference = _export_epoch_inference(
                model, dataset, val_set if n_val else train_set, epoch + 1, global_step,
                epoch_export_root, message_bits, seed, device, epoch_export_samples)
            record = {"epoch": epoch + 1, "global_step": global_step,
                      "train": train_metrics, "validation": validation_metrics,
                      "epoch_inference": epoch_inference,
                      "adaptive_image_loss": {"enabled": bool(adaptive_image_loss),
                          "ready_epochs": adaptive_ready_epochs, "start_step": adaptive_start_step,
                          "ramp_factor": (1.0 if not adaptive_image_loss else
                              (0.0 if adaptive_start_step is None else _linear_ramp(global_step - adaptive_start_step, adaptive_ramp_steps)))}}
            with metrics_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            curve_records.append(record)
            _write_curve_files(curves_file, curve_records)

            if is_best:
                best = val_loss
            train_state = {
                "secret_optimizer": secret_optimizer.state_dict(),
                "full_optimizer": full_optimizer.state_dict(),
                "discriminator": discriminator.state_dict() if discriminator is not None else None,
                "discriminator_optimizer": discriminator_optimizer.state_dict() if discriminator_optimizer is not None else None,
                "train_config": current_train_config,
                "adaptive_ready_epochs": adaptive_ready_epochs,
                "adaptive_start_step": adaptive_start_step,
            }
            save_checkpoint(latest_output, model, optimizer=None, epoch=epoch + 1,
                            best_val_loss=best, global_step=global_step, training_state=train_state)
            if is_best or not output.exists() or not val_loader:
                save_checkpoint(output, model, optimizer=None, epoch=epoch + 1,
                                best_val_loss=best, global_step=global_step, training_state=train_state)
            if max_steps is not None and global_step >= max_steps:
                break
    finally:
        progress_bar.close()
    print("epoch diagnostics: {}".format(metrics_file.resolve()))
    return model

def _capture_rng_state():
    state = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state):
    if state.get("python_rng_state") is not None:
        random.setstate(state["python_rng_state"])
    if state.get("numpy_rng_state") is not None:
        np.random.set_state(state["numpy_rng_state"])
    if state.get("torch_rng_state") is not None:
        torch.set_rng_state(state["torch_rng_state"])
    if torch.cuda.is_available() and state.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])


def _read_checkpoint(path, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("format") not in ({CHECKPOINT_FORMAT} | LEGACY_CHECKPOINT_FORMATS):
        raise ValueError(
            "Unsupported checkpoint format. Expected a StegaStamp-compatible model checkpoint ({}) or a supported legacy checkpoint.".format(CHECKPOINT_FORMAT)
        )
    required = ("message_bits", "width", "use_stn", "residual_scale", "image_size", "model")
    if any(key not in state for key in required):
        raise ValueError("Checkpoint is missing architecture metadata: {}".format(path))
    return state


def save_checkpoint(path, model, optimizer=None, epoch=0, best_val_loss=float("inf"), global_step=0, training_state=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "format": CHECKPOINT_FORMAT,
        "message_bits": model.message_bits,
        "width": model.width,
        "use_stn": model.use_stn,
        "residual_scale": model.residual_scale,
        "image_size": model.image_size[0],
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "training_state": training_state or {},
    }
    state.update(_capture_rng_state())
    torch.save(state, path)


def load_checkpoint(path, device=None):
    device = _device(device)
    state = _read_checkpoint(path, device)
    model = MarkProbe(state["message_bits"], state["width"],
                      use_stn=state["use_stn"], residual_scale=state["residual_scale"],
                      image_size=state["image_size"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    return model, device


def _image_tensor(image, image_size=None):
    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    if image_size:
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        height, width = image_size
        image = image.resize((width, height), Image.Resampling.BICUBIC)
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def encode_image(model, image, bits, device=None):
    device = _device(device)
    tensor = _image_tensor(image, model.image_size).to(device)
    bits = torch.as_tensor(bits, dtype=torch.float32, device=device).reshape(1, -1)
    if bits.shape[1] != model.message_bits:
        raise ValueError("Expected {} message bits".format(model.message_bits))
    model.eval()
    with torch.no_grad():
        encoded = model.encode(tensor, bits)[0].clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    return Image.fromarray(np.uint8(np.clip(np.round(encoded * 255.0), 0, 255)), "RGB")


def decode_image(model, image, device=None):
    device = _device(device)
    tensor = _image_tensor(image, model.image_size).to(device)
    model.eval()
    with torch.no_grad():
        probs = model.decode(tensor).sigmoid()[0]
    bits = (probs >= 0.5).to(torch.uint8).cpu().tolist()
    return bits, probs.cpu().tolist()

