"""Recovered temporal preprocessing/postprocessing for DLSS 5 neural rendering.

This module is plain NumPy/PyTorch-adjacent glue around the local model runtime.
It does not call NGX, NVIDIA DLLs, MLX, OpenCV, or any native bridge.

Motion is current-to-previous. The portable boundary is a signed normalized-UV
offset added to the current pixel centre. Pixel motion can be converted with
``normalize_pixel_motion`` using the recovered host scale/jitter contract.
"""
from __future__ import annotations

import numpy as np

from .features import (
    AutomaticMask,
    NetworkGeometry,
    deterministic_noise,
    half,
    make_features,
    scaled_color,
)

BLEND_SCALE = 0.73974609375


def normalize_pixel_motion(
    pixel_motion: np.ndarray,
    *,
    scale_x: float,
    scale_y: float,
    effective_width: int,
    effective_height: int,
    jitter_dx: float = 0.0,
    jitter_dy: float = 0.0,
) -> np.ndarray:
    """Convert signed pixel motion to normalized current->previous UV offsets.

    Recovered host contract::

        uv_x = (pixel_x * scale_x + jitter_dx) / effective_width
        uv_y = (pixel_y * scale_y + jitter_dy) / effective_height

    ``jitter_dx/y`` is previous jitter minus current jitter, in pixels.
    """
    pixel_motion = np.asarray(pixel_motion, dtype=np.float32)
    if pixel_motion.ndim != 3 or pixel_motion.shape[2] != 2:
        raise ValueError("pixel motion must be [H,W,2]")
    if effective_width <= 0 or effective_height <= 0:
        raise ValueError("effective motion extent must be positive")
    for value in (scale_x, scale_y, jitter_dx, jitter_dy):
        if not np.isfinite(value):
            raise ValueError("motion scale and jitter must be finite")

    out = np.empty_like(pixel_motion)
    out[..., 0] = (
        pixel_motion[..., 0] * np.float32(scale_x) + np.float32(jitter_dx)
    ) / np.float32(effective_width)
    out[..., 1] = (
        pixel_motion[..., 1] * np.float32(scale_y) + np.float32(jitter_dy)
    ) / np.float32(effective_height)
    return out


def _catmull_coordinates(normalized: np.ndarray, dimension: int):
    pixel = normalized * np.float32(dimension) - np.float32(0.5)
    base_index = np.floor(pixel)
    t = np.clip(pixel - base_index, 0, 1).astype(np.float32)
    square = t * t
    cube = square * t
    w0 = -0.5 * t + square - 0.5 * cube
    w1 = 1 - 2.5 * square + 1.5 * cube
    w2 = 0.5 * t + 2 * square - 1.5 * cube
    w3 = -0.5 * square + 0.5 * cube
    g = w1 + w2
    base = base_index + np.float32(0.5)
    lower = np.float32(0.5)
    upper = np.float32(dimension) - np.float32(0.5)
    return (
        np.clip(base - 1, lower, upper),
        np.clip(base + w2 / g, lower, upper),
        np.clip(base + 2, lower, upper),
        w0.astype(np.float32),
        w3.astype(np.float32),
        g.astype(np.float32),
    )


def _sample_linear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Clamp-to-edge bilinear sampling at pixel-centre coordinates."""
    height, width = image.shape[:2]
    px = x - np.float32(0.5)
    py = y - np.float32(0.5)
    x0 = np.clip(np.floor(px), 0, width - 1).astype(np.int64)
    y0 = np.clip(np.floor(py), 0, height - 1).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    tx = np.clip(px - x0, 0, 1).astype(np.float32)[..., None]
    ty = np.clip(py - y0, 0, 1).astype(np.float32)[..., None]
    top = image[y0, x0] * (1 - tx) + image[y0, x1] * tx
    bottom = image[y1, x0] * (1 - tx) + image[y1, x1] * tx
    return top * (1 - ty) + bottom * ty


def sample_history(history: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Recovered five-tap Catmull-Rom history reconstruction."""
    history = np.asarray(history, dtype=np.float32)
    if history.ndim != 3 or history.shape[2] != 3:
        raise ValueError("history must be [H,W,3]")
    height, width = history.shape[:2]

    x0, xm, x3, xw0, xw3, xg = _catmull_coordinates(
        np.asarray(u, dtype=np.float32), width
    )
    y0, ym, y3, yw0, yw3, yg = _catmull_coordinates(
        np.asarray(v, dtype=np.float32), height
    )

    weights = [xw0 * yg, xg * yw0, xg * yg, xg * yw3, xw3 * yg]
    taps = [
        _sample_linear(history, x0, ym),
        _sample_linear(history, xm, y0),
        _sample_linear(history, xm, ym),
        _sample_linear(history, xm, y3),
        _sample_linear(history, x3, ym),
    ]
    numerator = sum(weight[..., None] * tap for weight, tap in zip(weights, taps))
    denominator = sum(weights)[..., None]
    return (numerator / denominator).astype(np.float32)


def _closest_depth_offsets(depth: np.ndarray, inverted: bool):
    """Dormant recovered closest-depth guide: current pixel + four diagonals."""
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 2:
        depth = depth[..., None]
    if depth.ndim != 3 or depth.shape[2] < 1:
        raise ValueError("depth must be [H,W] or [H,W,C>=1]")

    height, width = depth.shape[:2]
    yy, xx = np.indices((height, width))
    best_x = xx.copy()
    best_y = yy.copy()
    best = depth[..., 0].copy()
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cx = np.clip(xx + dx, 0, width - 1)
        cy = np.clip(yy + dy, 0, height - 1)
        candidate = depth[cy, cx, 0]
        closer = candidate > best if inverted else candidate < best
        best_x = np.where(closer, cx, best_x)
        best_y = np.where(closer, cy, best_y)
        best = np.where(closer, candidate, best)
    return best_x, best_y


def make_temporal_features(
    color: np.ndarray,
    history: np.ndarray,
    motion: np.ndarray,
    *,
    frame_index: int,
    depth: np.ndarray | None = None,
    depth_guide: str = "observed",
    depth_inverted: bool = False,
    normalized_style: float = 0.0,
    local_tone_strength: float = 1.0,
    local_structure_strength: float = 1.0,
    automatic_mask: AutomaticMask | None = None,
    control_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Build logical-size 16-channel temporal features.

    ``depth_guide='observed'`` matches the surfaced DLL: depth is ignored and
    motion is sampled at the current pixel. ``'closest'`` exposes the dormant
    structurally recovered depth-guided branch.
    """
    color = np.asarray(color, dtype=np.float32)
    history = np.asarray(history, dtype=np.float32)
    motion = np.asarray(motion, dtype=np.float32)
    height, width = color.shape[:2]
    if color.shape != (height, width, 3):
        raise ValueError("color must be [H,W,3]")
    if history.shape != color.shape:
        raise ValueError("history must match current color")
    if motion.shape != (height, width, 2):
        raise ValueError("motion must be [H,W,2]")

    features = make_features(
        color,
        frame_index=frame_index,
        normalized_style=normalized_style,
        local_tone_strength=local_tone_strength,
        local_structure_strength=local_structure_strength,
        automatic_mask=automatic_mask,
        control_mask=control_mask,
    )

    yy, xx = np.indices((height, width))
    if depth_guide == "closest":
        if depth is None:
            raise ValueError("closest-depth guide requires a depth image")
        sx, sy = _closest_depth_offsets(depth, depth_inverted)
        sampled_motion = motion[sy, sx]
    elif depth_guide == "observed":
        sampled_motion = motion
    else:
        raise ValueError("depth_guide must be 'observed' or 'closest'")

    u = (xx.astype(np.float32) + np.float32(0.5)) / np.float32(width)
    v = (yy.astype(np.float32) + np.float32(0.5)) / np.float32(height)
    u = u + sampled_motion[..., 0]
    v = v + sampled_motion[..., 1]
    features[..., 7:10] = scaled_color(sample_history(history, u, v))
    return features


def extend_features(
    features: np.ndarray,
    geometry: NetworkGeometry,
    frame_index: int,
) -> np.ndarray:
    """Mirror logical temporal features to the vendor-aligned network extent."""
    if geometry.is_identity:
        return features
    rows = geometry.source_rows()
    columns = geometry.source_columns()
    extended = features[rows[:, None], columns[None, :], :].copy()
    noise = deterministic_noise(
        geometry.network_height, geometry.network_width, frame_index
    )
    outside = (
        (rows[:, None] != np.arange(geometry.network_height)[:, None])
        | (columns[None, :] != np.arange(geometry.network_width)[None, :])
    )
    extended[..., 0:3] = np.where(outside[..., None], noise, extended[..., 0:3])
    return extended


def compose_temporal(
    head: np.ndarray,
    color: np.ndarray,
    features: np.ndarray,
    *,
    blend_scale: float = BLEND_SCALE,
    control_mask: np.ndarray | None = None,
    intensity: float = 1.0,
) -> np.ndarray:
    """Compose the learned temporal blend before detail/colour postprocessing."""
    head = np.asarray(head, dtype=np.float32)
    color = np.asarray(color, dtype=np.float32)
    features = np.asarray(features, dtype=np.float32)
    if head.shape[:2] != color.shape[:2]:
        raise ValueError("head and color must share height/width")
    if features.shape[:2] != color.shape[:2] or features.shape[2] != 16:
        raise ValueError("features must be [H,W,16] matching color")

    logit = half(head[..., 3:4])
    alpha = np.clip(
        1 / (1 + np.exp(-logit)) * half(blend_scale),
        0,
        1,
    )
    predicted = np.clip(
        color + half(head[..., :3]) * np.float32(0.25),
        0,
        1,
    )
    history = features[..., 7:10] * np.float32(8) + np.float32(0.5)
    temporal = predicted + alpha * (history - predicted)

    if control_mask is None and intensity == 1:
        return temporal.astype(np.float32)
    if control_mask is None:
        blend = np.float32(intensity)
    else:
        control_mask = np.asarray(control_mask, dtype=np.float32)
        if control_mask.shape != color.shape:
            raise ValueError("control mask must match color")
        blend = control_mask[..., :1] * np.float32(intensity)
    blend = np.clip(blend, 0, 1)
    return np.clip(color + blend * (temporal - color), 0, 1).astype(np.float32)
