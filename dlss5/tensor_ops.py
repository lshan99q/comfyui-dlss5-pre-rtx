"""Device-resident feature, reprojection, resize, and composition operations.

Images use HWC float32 tensors. No frame data is copied to the host here.
The NumPy modules remain the portable reference for numerical comparisons.
"""
from __future__ import annotations

import math
import struct

import torch
from torch.nn import functional as F

from .features import AutomaticMask, NetworkGeometry

BLEND_SCALE = 0.73974609375
_UINT32_MASK = 0xFFFFFFFF


class ReferenceNoiseTables(torch.nn.Module):
    """Immutable tables that preserve the reference's float32 transcendentals.

    Uniforms have exactly 24 bits, so three 2**24-entry tables cover every
    possible radius/sine/cosine value. These 192 MiB of constants are built
    once on CPU at initialization and move/offload with the model. Per-frame
    hashing, lookup, multiplication and half publication all run on-device.
    This avoids CPU/GPU libm differences crossing half-rounding boundaries.
    """
    def __init__(self):
        super().__init__()
        import numpy as np

        size = 1 << 24
        radius, cosine, sine = (np.empty(size, dtype=np.float32) for _ in range(3))
        for start in range(0, size, 1 << 20):
            end = min(size, start + (1 << 20))
            uniform = np.arange(start + 1, end + 1, dtype=np.float32) * np.float32(2**-24)
            angle = np.float32(6.2831854820251465) * uniform
            radius[start:end] = np.sqrt(np.float32(-2) * np.log(uniform))
            cosine[start:end], sine[start:end] = np.cos(angle), np.sin(angle)
        for name, values in (("radius", radius), ("cosine", cosine), ("sine", sine)):
            self.register_buffer(name, torch.from_numpy(values), persistent=False)
        # The temporal head is half-published: its sigmoid has only 65536
        # possible inputs. Preserve NumPy's exp/divide rounding here as well.
        logits = np.arange(1 << 16, dtype=np.uint16).view(np.float16).astype(np.float32)
        with np.errstate(over="ignore", invalid="ignore"):
            sigmoid = np.float32(1) / (np.float32(1) + np.exp(-logits))
        self.register_buffer("sigmoid", torch.from_numpy(sigmoid), persistent=False)


def half(value: torch.Tensor) -> torch.Tensor:
    return value.to(torch.float16).to(torch.float32)


def _divide(value: torch.Tensor, denominator) -> torch.Tensor:
    # CUDA's scalar float division can multiply by a rounded reciprocal.
    # Evaluate the division in double before rounding to float32, as needed
    # for reference-compatible motion/UV coordinates at half boundaries.
    if value.device.type == "mps":
        return value / denominator
    divisor = denominator.double() if isinstance(denominator, torch.Tensor) else denominator
    return (value.double() / divisor).float()


def _half_scalar(value: float) -> float:
    # Controls are host scalars, not image data. Round without allocating a
    # device scalar or introducing a device-to-host synchronization.
    return struct.unpack("e", struct.pack("e", float(value)))[0]


def _rgb(image: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(image, torch.Tensor) or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"{name} must be a [H,W,3] tensor")
    if min(image.shape[:2]) <= 0:
        raise ValueError(f"{name} extent must be positive")
    return image.float()


def _dynamic_shift_mix(value: torch.Tensor) -> torch.Tensor:
    value = value & _UINT32_MASK
    return ((value ^ (value >> ((value >> 28) + 4))) * 0x108EF2D9) & _UINT32_MASK


def _uniform24_bits(value: torch.Tensor) -> torch.Tensor:
    mixed = _dynamic_shift_mix(value)
    return (mixed >> 30) ^ (mixed >> 8)


def _uniform24(value: torch.Tensor) -> torch.Tensor:
    return (_uniform24_bits(value) + 1).float() * 5.960464477539063e-8


def deterministic_noise(height: int, width: int, frame_index: int = 0, *, device=None,
                         tables: ReferenceNoiseTables | None = None) -> torch.Tensor:
    yy = torch.arange(height, device=device, dtype=torch.int64)[:, None]
    xx = torch.arange(width, device=device, dtype=torch.int64)[None, :]
    seed = (yy * 0xD8163841) ^ (xx * 0x8DA6B343)
    seed = seed ^ ((int(frame_index) * 0x9E3779B9) & _UINT32_MASK) ^ 0x243F6A88
    multiplied = _dynamic_shift_mix(seed)
    mixed = multiplied ^ (multiplied >> 22)
    if tables is not None:
        radius_a = tables.radius[_uniform24_bits(mixed * 0xCAA5B80D + 0x21DD796B)]
        angle_b = _uniform24_bits(mixed * 0x83232C31 + 0x3463E0AC)
        radius_b = tables.radius[_uniform24_bits(mixed * 0x2C9277B5 + 0xAC564B05)]
        angle_a = _uniform24_bits(mixed * 0xFA6DC5F9 + 0x4712A88E)
        return half(torch.stack((radius_b * tables.cosine[angle_a], radius_b * tables.sine[angle_a],
                                 radius_a * tables.cosine[angle_b]), dim=-1))
    ra = _uniform24(mixed * 0xCAA5B80D + 0x21DD796B)
    ab = _uniform24(mixed * 0x83232C31 + 0x3463E0AC)
    rb = _uniform24(mixed * 0x2C9277B5 + 0xAC564B05)
    aa = _uniform24(mixed * 0xFA6DC5F9 + 0x4712A88E)
    radius_a, radius_b = (-2 * ra.log()).sqrt(), (-2 * rb.log()).sqrt()
    angle_a, angle_b = 6.2831854820251465 * aa, 6.2831854820251465 * ab
    return half(torch.stack((radius_b * angle_a.cos(), radius_b * angle_a.sin(),
                             radius_a * angle_b.cos()), dim=-1))


def scaled_color(color: torch.Tensor) -> torch.Tensor:
    return half(half(half(color) - 0.5) * 0.125)


def _extended_indices(count: int, extent: int, device) -> torch.Tensor:
    index = torch.arange(count, device=device)
    return torch.where(index < extent, index, (2 * extent - 2 - index).clamp_min(0))


def make_features(
    color: torch.Tensor, *, frame_index: int = 0,
    geometry: NetworkGeometry | None = None, normalized_style: float = 0.0,
    local_tone_strength: float = 1.0, local_structure_strength: float = 1.0,
    automatic_mask: AutomaticMask | None = None, control_mask: torch.Tensor | None = None,
    history: torch.Tensor | None = None,
    noise_tables: ReferenceNoiseTables | None = None,
) -> torch.Tensor:
    color = _rgb(color, "color")
    height, width = color.shape[:2]
    geometry = geometry or NetworkGeometry.identity(width, height)
    if (geometry.output_height, geometry.output_width) != (height, width):
        raise ValueError("geometry must match color")
    if control_mask is not None and control_mask.shape != color.shape:
        raise ValueError("control mask must match color")
    if history is not None and history.shape != color.shape:
        raise ValueError("history must match color")
    if control_mask is not None:
        structure = skin = automatic = 0.0
    elif automatic_mask is not None:
        skin, automatic = automatic_mask.skin_structure_strength, automatic_mask.automatic_mask_structure_strength
        enabled = max(skin, automatic) >= 0
        structure = 1.0 if enabled else local_structure_strength
        skin = (skin if skin >= 0 else local_structure_strength) if enabled else -1.0
        automatic = (automatic if automatic >= 0 else local_structure_strength) if enabled else -1.0
    else:
        structure, skin, automatic = local_structure_strength, -1.0, -1.0
    rows = _extended_indices(geometry.network_height, height, color.device)
    columns = _extended_indices(geometry.network_width, width, color.device)
    extended = color[rows[:, None], columns[None, :]]
    features = color.new_zeros((geometry.network_height, geometry.network_width, 16))
    features[..., :3] = deterministic_noise(geometry.network_height, geometry.network_width,
                                           frame_index, device=color.device, tables=noise_tables)
    features[..., 3] = 1.0
    features[..., 4:7] = scaled_color(extended)
    features[..., 7:10] = features[..., 4:7] if history is None else scaled_color(history)[rows[:, None], columns[None, :]]
    features[..., 10] = _half_scalar(normalized_style)
    if control_mask is not None:
        mask = control_mask[rows[:, None], columns[None, :]].float()
        features[..., 11] = half(mask[..., 1] * local_tone_strength)
        features[..., 12] = half(mask[..., 2] * local_structure_strength)
    else:
        features[..., 11] = _half_scalar(local_tone_strength)
        features[..., 12] = _half_scalar(structure)
    features[..., 13] = _half_scalar(skin)
    features[..., 14] = _half_scalar(automatic)
    return features


def normalize_pixel_motion(pixel_motion: torch.Tensor, *, scale_x: float, scale_y: float,
                           effective_width: int, effective_height: int,
                           jitter_dx: float = 0.0, jitter_dy: float = 0.0) -> torch.Tensor:
    if pixel_motion.ndim != 3 or pixel_motion.shape[-1] != 2:
        raise ValueError("pixel motion must be [H,W,2]")
    if effective_width <= 0 or effective_height <= 0:
        raise ValueError("effective motion extent must be positive")
    if not all(math.isfinite(v) for v in (scale_x, scale_y, jitter_dx, jitter_dy)):
        raise ValueError("motion scale and jitter must be finite")
    motion = pixel_motion.float()
    return torch.stack((_divide(motion[..., 0] * scale_x + jitter_dx, effective_width),
                        _divide(motion[..., 1] * scale_y + jitter_dy, effective_height)), dim=-1)


def _catmull_coordinates(normalized: torch.Tensor, dimension: int):
    pixel = normalized * dimension - 0.5
    base_index = pixel.floor()
    t = (pixel - base_index).clamp(0, 1)
    square, cube = t * t, t * t * t
    w0 = -0.5 * t + square - 0.5 * cube
    w1 = 1 - 2.5 * square + 1.5 * cube
    w2 = 0.5 * t + 2 * square - 1.5 * cube
    w3 = -0.5 * square + 0.5 * cube
    g = w1 + w2
    base = base_index + 0.5
    return ((base - 1).clamp(0.5, dimension - 0.5),
            (base + _divide(w2, g)).clamp(0.5, dimension - 0.5),
            (base + 2).clamp(0.5, dimension - 0.5), w0, w3, g)


def sample_history(history: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    history = _rgb(history, "history")
    if u.shape != v.shape or u.ndim != 2:
        raise ValueError("history UVs must be matching 2-D tensors")
    result = history.new_empty((*u.shape, 3))
    # Limit five-tap gather intermediates for large video frames.
    rows = max(1, (1 << 18) // u.shape[1])
    for start in range(0, u.shape[0], rows):
        end = min(u.shape[0], start + rows)
        result[start:end] = _sample_history_strip(history, u[start:end], v[start:end])
    return result


def _sample_history_strip(history: torch.Tensor, u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    height, width = history.shape[:2]
    x0, xm, x3, xw0, xw3, xg = _catmull_coordinates(u.float(), width)
    y0, ym, y3, yw0, yw3, yg = _catmull_coordinates(v.float(), height)
    xs = torch.stack((x0, xm, xm, xm, x3))
    ys = torch.stack((ym, y0, ym, y3, ym))
    # Keep pixel-centre coordinates and the recovered arithmetic order.
    # grid_sample renormalizes coordinates and fuses interpolation arithmetic,
    # which can change half-published history and hence later network outputs.
    px, py = xs - 0.5, ys - 0.5
    ix = px.floor().long().clamp(0, width - 1)
    iy = py.floor().long().clamp(0, height - 1)
    ix1, iy1 = (ix + 1).clamp_max(width - 1), (iy + 1).clamp_max(height - 1)
    tx, ty = (px - ix).clamp(0, 1)[..., None], (py - iy).clamp(0, 1)[..., None]
    top = history[iy, ix] * (1 - tx) + history[iy, ix1] * tx
    bottom = history[iy1, ix] * (1 - tx) + history[iy1, ix1] * tx
    taps = top * (1 - ty) + bottom * ty
    weights = torch.stack((xw0 * yg, xg * yw0, xg * yg, xg * yw3, xw3 * yg))
    # Match the reference's left-to-right summation order.
    total = taps[0] * weights[0, ..., None]
    denominator = weights[0]
    for index in range(1, 5):
        total = total + taps[index] * weights[index, ..., None]
        denominator = denominator + weights[index]
    return _divide(total, denominator[..., None])


def _guided_motion(motion: torch.Tensor, depth: torch.Tensor, inverted: bool) -> torch.Tensor:
    height, width = motion.shape[:2]
    if depth.ndim == 2:
        depth = depth[..., None]
    if depth.ndim != 3 or depth.shape[:2] != (height, width) or depth.shape[-1] < 1:
        raise ValueError("depth must match the current frame")
    padded = F.pad(depth[..., 0][None, None].float(), (1, 1, 1, 1), mode="replicate")[0, 0]
    candidates = torch.stack((padded[1:-1, 1:-1], padded[:-2, :-2], padded[:-2, 2:],
                              padded[2:, :-2], padded[2:, 2:]))
    pick = candidates.argmax(0) if inverted else candidates.argmin(0)
    # argmin/argmax select the first equal candidate, matching the reference.
    dx = torch.where((pick == 1) | (pick == 3), -1, torch.where(pick == 0, 0, 1))
    dy = torch.where((pick == 1) | (pick == 2), -1, torch.where(pick == 0, 0, 1))
    x = (torch.arange(width, device=motion.device)[None, :] + dx).clamp(0, width - 1)
    y = (torch.arange(height, device=motion.device)[:, None] + dy).clamp(0, height - 1)
    return motion[y, x]


def make_temporal_features(color: torch.Tensor, history: torch.Tensor, motion: torch.Tensor, *,
                           frame_index: int, depth: torch.Tensor | None = None,
                           depth_guide: str = "observed", depth_inverted: bool = False,
                           geometry: NetworkGeometry | None = None, **controls) -> torch.Tensor:
    color = _rgb(color, "color")
    height, width = color.shape[:2]
    if history.shape != color.shape or motion.shape != (height, width, 2):
        raise ValueError("history and motion must match current color")
    if depth_guide == "closest":
        if depth is None:
            raise ValueError("closest-depth guide requires depth_image")
        motion = _guided_motion(motion, depth, depth_inverted)
    elif depth_guide != "observed":
        raise ValueError("depth_guide must be 'observed' or 'closest'")
    u = _divide(torch.arange(width, device=color.device, dtype=torch.float32)[None, :] + 0.5, width) + motion[..., 0]
    v = _divide(torch.arange(height, device=color.device, dtype=torch.float32)[:, None] + 0.5, height) + motion[..., 1]
    return make_features(color, history=sample_history(history, u, v), geometry=geometry,
                         frame_index=frame_index, **controls)


def blend_effect(color: torch.Tensor, predicted: torch.Tensor, control_mask: torch.Tensor | None = None,
                 intensity: float = 1.0) -> torch.Tensor:
    if not math.isfinite(intensity):
        raise ValueError("intensity must be finite")
    if control_mask is None:
        blend = min(1.0, max(0.0, intensity))
    else:
        if control_mask.shape != color.shape:
            raise ValueError("control mask must match color")
        blend = (control_mask[..., :1].float() * intensity).clamp(0, 1)
    return (color + blend * (predicted - color)).clamp(0, 1)


def compose_head(head: torch.Tensor, color: torch.Tensor, *, control_mask=None,
                 intensity: float = 1.0) -> torch.Tensor:
    return blend_effect(color, (color + half(head[..., :3]) * 0.25).clamp(0, 1), control_mask, intensity)


def compose_temporal(head: torch.Tensor, color: torch.Tensor, features: torch.Tensor, *,
                     blend_scale: float = BLEND_SCALE, control_mask=None,
                     intensity: float = 1.0, reference_tables: ReferenceNoiseTables | None = None) -> torch.Tensor:
    logit = head[..., 3:4].to(torch.float16)
    if reference_tables is None:
        probability = logit.float().sigmoid()
    else:
        probability = reference_tables.sigmoid[logit.view(torch.int16).long() & 0xFFFF]
    alpha = (probability * _half_scalar(blend_scale)).clamp(0, 1)
    predicted = (color + half(head[..., :3]) * 0.25).clamp(0, 1)
    history = features[..., 7:10] * 8 + 0.5
    temporal = predicted + alpha * (history - predicted)
    if control_mask is None and intensity == 1:
        return temporal
    return blend_effect(color, temporal, control_mask, intensity)


def compose_detail(source: torch.Tensor, output: torch.Tensor, *, detail_strength: float = 1.0,
                   colour_strength: float = 1.0, radius: float = 4.0) -> torch.Tensor:
    if not all(math.isfinite(v) for v in (detail_strength, colour_strength, radius)) or radius <= 0:
        raise ValueError("strengths must be finite and radius must be positive")
    if source.shape != output.shape:
        raise ValueError("source and output must share a shape")
    if detail_strength == 1 and colour_strength == 1:
        return output
    change = (output - source).permute(2, 0, 1)[None]
    extent = math.ceil(3 * radius)
    offsets = torch.arange(-extent, extent + 1, device=source.device, dtype=torch.float32)
    kernel = (-offsets.square() / (2 * radius * radius)).exp()
    kernel = kernel / kernel.sum()
    channels = source.shape[-1]
    horizontal = kernel.reshape(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.reshape(1, 1, -1, 1).expand(channels, 1, -1, 1)
    low = F.conv2d(F.pad(change, (extent, extent, 0, 0), mode="replicate"), horizontal, groups=channels)
    low = F.conv2d(F.pad(low, (0, 0, extent, extent), mode="replicate"), vertical, groups=channels)
    low = low[0].permute(1, 2, 0)
    return (source + colour_strength * low + detail_strength * (output - source - low)).clamp(0, 1)


def _resample_axis(image: torch.Tensor, target: int, axis: int) -> torch.Tensor:
    source = image.shape[axis]
    if source == target:
        return image
    scale = source / target
    filter_scale = max(1.0, scale)
    support = 3 * filter_scale
    # Pillow evaluates float-plane filter coefficients and sums in double.
    # MPS has no float64 support, so it uses the float32 approximation.
    accumulation_dtype = torch.float32 if image.device.type == "mps" else torch.float64
    centers = (torch.arange(target, device=image.device, dtype=accumulation_dtype) + 0.5) * scale
    left = (centers - support + 0.5).floor().long()
    taps = math.ceil(2 * support) + 1
    indices = left[:, None] + torch.arange(taps, device=image.device)[None, :]
    distance = (indices.to(accumulation_dtype) + 0.5 - centers[:, None]) / filter_scale
    def sinc(value):
        angle = value * math.pi
        return torch.where(value == 0, 1.0, angle.sin() / angle)
    # Basic precompiled tensor operations also work on CUDA installs without
    # NVRTC's builtins (torch.sinc may JIT a CUDA kernel on those installs).
    weights = sinc(distance) * sinc(distance / 3)
    valid = (indices >= 0) & (indices < source) & (distance.abs() < 3)
    weights = torch.where(valid, weights, 0)
    weights = weights / weights.sum(-1, keepdim=True)
    indices = indices.clamp(0, source - 1)
    moved = image.movedim(axis, 0)
    flat = moved.reshape(source, -1)
    result = image.new_empty((target, flat.shape[1]))
    # Bound the gather temporary to ~16 MiB, independently of the frame size.
    chunk = max(1, (1 << 22) // (taps * flat.shape[1]))
    for start in range(0, target, chunk):
        end = min(target, start + chunk)
        values = flat[indices[start:end]]
        products = values * weights[start:end, :, None]
        summed = products[:, 0]
        for tap in range(1, taps):
            summed = summed + products[:, tap]
        result[start:end] = summed
    return result.reshape(target, *moved.shape[1:]).movedim(0, axis)


def resample(image: torch.Tensor, width: int, height: int) -> torch.Tensor:
    if width <= 0 or height <= 0:
        raise ValueError("resample target must be positive")
    image = image.float()
    source_height, source_width = image.shape[:2]
    if (height, width) == (source_height, source_width):
        return image
    if (source_width % width == 0 and source_height % height == 0
            and source_width // width == source_height // height and source_width // width > 1):
        factor = source_width // width
        return image.reshape(height, factor, width, factor, image.shape[-1]).mean((1, 3))
    # Pillow's float-plane Lanczos order: horizontal, then vertical. Normalize
    # truncated kernels at the edges instead of replicating edge pixels.
    return _resample_axis(_resample_axis(image, width, 1), height, 0)
