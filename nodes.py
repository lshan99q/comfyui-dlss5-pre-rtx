from __future__ import annotations

import gc
import os
import weakref
from dataclasses import dataclass
from functools import wraps
from inspect import signature
from itertools import chain
from pathlib import Path
from threading import RLock
from typing import Any

import torch

import folder_paths
from comfy import model_management


CATEGORY = "DLSS 5/PyTorch (experimental)"
MODEL_FOLDER_NAME = "dlss5"
MODEL_EXTENSIONS = {".safetensors"}
NO_MODEL_SENTINEL = "[no logical DLSS 5 .safetensors found]"

_MODEL_DIR = Path(folder_paths.models_dir) / MODEL_FOLDER_NAME
_MODEL_DIR.mkdir(parents=True, exist_ok=True)

try:
    folder_paths.add_model_folder_path(MODEL_FOLDER_NAME, str(_MODEL_DIR))
except Exception:
    if MODEL_FOLDER_NAME not in folder_paths.folder_names_and_paths:
        folder_paths.folder_names_and_paths[MODEL_FOLDER_NAME] = (
            [str(_MODEL_DIR)],
            MODEL_EXTENSIONS,
        )


@dataclass(eq=False)
class DLSS5ModelHandle:
    pipeline: Any
    model_path: str
    precision: str
    device: str


_PIPELINE_CACHE: dict[tuple[str, int, int, str, str], DLSS5ModelHandle] = {}
_LIVE_HANDLES: weakref.WeakSet[DLSS5ModelHandle] = weakref.WeakSet()
_MODEL_LOCK = RLock()


def _import_pipeline_class():
    try:
        from .dlss5.pipeline import NeuralRenderingPipeline
    except (ImportError, ValueError):
        from dlss5.pipeline import NeuralRenderingPipeline
    return NeuralRenderingPipeline


def _import_render_runtime():
    try:
        from .dlss5.features import AutomaticMask, NetworkGeometry, PROFILES
        from .dlss5.tensor_ops import (
            BLEND_SCALE,
            blend_effect,
            compose_detail,
            compose_head,
            compose_temporal,
            make_features,
            make_temporal_features,
            normalize_pixel_motion,
        )
    except (ImportError, ValueError):
        from dlss5.features import AutomaticMask, NetworkGeometry, PROFILES
        from dlss5.tensor_ops import (
            BLEND_SCALE,
            blend_effect,
            compose_detail,
            compose_head,
            compose_temporal,
            make_features,
            make_temporal_features,
            normalize_pixel_motion,
        )
    return {
        "AutomaticMask": AutomaticMask,
        "NetworkGeometry": NetworkGeometry,
        "PROFILES": PROFILES,
        "make_features": make_features,
        "compose_head": compose_head,
        "blend_effect": blend_effect,
        "compose_detail": compose_detail,
        "BLEND_SCALE": BLEND_SCALE,
        "compose_temporal": compose_temporal,
        "make_temporal_features": make_temporal_features,
        "normalize_pixel_motion": normalize_pixel_motion,
    }


def _model_names() -> list[str]:
    try:
        names = folder_paths.get_filename_list(MODEL_FOLDER_NAME)
    except Exception:
        names = []
    names = [name for name in names if Path(name).suffix.lower() in MODEL_EXTENSIONS]
    return sorted(names) or [NO_MODEL_SENTINEL]


def _resolve_model_path(model_name: str) -> str:
    if model_name == NO_MODEL_SENTINEL:
        raise FileNotFoundError(
            f"No DLSS 5 logical safetensors were found. Put a fully-logical "
            f".safetensors file in: {_MODEL_DIR}"
        )

    full_path = None
    try:
        full_path = folder_paths.get_full_path(MODEL_FOLDER_NAME, model_name)
    except Exception:
        pass
    if not full_path:
        candidate = (_MODEL_DIR / model_name).resolve()
        if candidate.is_file():
            full_path = str(candidate)
    if not full_path or not Path(full_path).is_file():
        raise FileNotFoundError(f"DLSS 5 model not found: {model_name}")
    return str(Path(full_path).resolve())


def _clear_pipeline_cache() -> None:
    with _MODEL_LOCK:
        # ComfyUI can retain loader outputs after our own cache is cleared.
        for handle in list(_LIVE_HANDLES):
            handle.pipeline.model.to("cpu")
        _PIPELINE_CACHE.clear()
        gc.collect()
        model_management.soft_empty_cache()


def _with_model_on_device(function):
    parameters = signature(function)

    @wraps(function)
    def render(self, dlss5_model, image, *args, **kwargs):
        if not isinstance(dlss5_model, DLSS5ModelHandle):
            raise TypeError("dlss5_model must come from the DLSS 5 PyTorch Model Loader")
        with _MODEL_LOCK, torch.inference_mode():
            pipeline = dlss5_model.pipeline
            _image_batch(image)
            model_management.throw_exception_if_processing_interrupted()
            # Like ComfyUI's standalone upscaler, reserve working memory before
            # moving weights and always offload, including on OOM/cancellation.
            # This is a heuristic; attention memory depends on the frame extent.
            arguments = parameters.bind(self, dlss5_model, image, *args, **kwargs).arguments
            scale = arguments["processing_scale"]
            pixels = max(320, int(image.shape[1] * scale)) * max(320, int(image.shape[2] * scale))
            # Include non-persistent lookup/bias buffers, which state_dict-based
            # size estimates omit, and reserve first-use reference constants.
            model_bytes = sum(t.numel() * t.element_size() for t in
                              chain(pipeline.model.parameters(), pipeline.model.buffers()))
            if not hasattr(pipeline.model, "noise_tables"):
                model_bytes += (3 * (1 << 24) + (1 << 16)) * 4
            memory = model_bytes + max(1024 ** 3, pixels * 1024)
            try:
                model_management.free_memory(memory, pipeline.device)
                pipeline.model.to(pipeline.device)
                pipeline.model.interrupt_check = model_management.throw_exception_if_processing_interrupted
                return function(self, dlss5_model, image, *args, **kwargs)
            finally:
                pipeline.model.interrupt_check = None
                pipeline.model.to("cpu")
    return render


def _load_pipeline(model_path: str, precision: str, device: str) -> DLSS5ModelHandle:
    stat = os.stat(model_path)
    key = (model_path, stat.st_size, stat.st_mtime_ns, precision, device)
    cached = _PIPELINE_CACHE.get(key)
    if cached is not None:
        return cached

    _clear_pipeline_cache()
    NeuralRenderingPipeline = _import_pipeline_class()
    pipeline = NeuralRenderingPipeline.from_safetensors(
        model_path,
        device=model_management.get_torch_device() if device == "auto" else device,
        precision=precision,
        offload_device="cpu",
    )
    handle = DLSS5ModelHandle(
        pipeline=pipeline,
        model_path=model_path,
        precision=precision,
        device=str(pipeline.device),
    )
    _PIPELINE_CACHE[key] = handle
    _LIVE_HANDLES.add(handle)
    return handle


def _image_batch(image: torch.Tensor) -> torch.Tensor:
    if not isinstance(image, torch.Tensor):
        raise TypeError("IMAGE input must be a torch.Tensor")
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError(f"Expected ComfyUI IMAGE as [B,H,W,C>=3], got {tuple(image.shape)}")
    if min(image.shape[:3]) <= 0:
        raise ValueError("IMAGE batch and spatial extent must be non-empty")
    return image[..., :3].detach()


def _motion_batch(motion: torch.Tensor) -> torch.Tensor:
    if not isinstance(motion, torch.Tensor):
        raise TypeError("motion_vectors must be a torch.Tensor")
    if motion.ndim != 4 or motion.shape[-1] < 2:
        raise ValueError(
            f"Expected motion_vectors as [B,H,W,C>=2], got {tuple(motion.shape)}"
        )
    return motion[..., :2].detach()


def _depth_batch(depth: torch.Tensor) -> torch.Tensor:
    if not isinstance(depth, torch.Tensor):
        raise TypeError("depth_image must be a torch.Tensor")
    if depth.ndim != 4 or depth.shape[-1] < 1:
        raise ValueError(f"Expected depth_image as [B,H,W,C>=1], got {tuple(depth.shape)}")
    return depth[..., :1].detach()


def _matching_frame(
    array: torch.Tensor | None,
    index: int,
    batch: int,
    *,
    name: str,
) -> torch.Tensor | None:
    if array is None:
        return None
    count = array.shape[0]
    if count == 1:
        return array[0]
    if count != batch:
        raise ValueError(f"{name} batch must be 1 or match input batch ({batch}); got {count}")
    return array[index]


def _matching_motion_frame(
    motion: torch.Tensor,
    index: int,
    batch: int,
) -> torch.Tensor:
    count = motion.shape[0]
    if count == 1:
        return motion[0]
    if count == batch:
        return motion[index]
    if count == batch - 1 and index > 0:
        return motion[index - 1]
    raise ValueError(
        f"motion_vectors batch must be 1, input batch ({batch}), or transitions ({batch - 1}); got {count}"
    )


def _resolved_controls(
    profile: str,
    use_custom_controls: bool,
    style_index: int,
    local_tone_strength: float,
    local_structure_strength: float,
    runtime: dict[str, Any],
) -> dict[str, float]:
    profiles = runtime["PROFILES"]
    if profile not in profiles:
        raise ValueError(f"unknown profile: {profile}")
    controls = dict(profiles[profile])
    if use_custom_controls:
        controls.update(
            normalized_style=float(style_index) / 128.0,
            local_tone_strength=float(local_tone_strength),
            local_structure_strength=float(local_structure_strength),
        )
    return controls


def _resolved_automatic_mask(
    use_auto_mask: bool,
    skin_structure_strength: float,
    automatic_mask_structure_strength: float,
    runtime: dict[str, Any],
):
    if not use_auto_mask:
        return None
    return runtime["AutomaticMask"](
        skin_structure_strength=float(skin_structure_strength),
        automatic_mask_structure_strength=float(automatic_mask_structure_strength),
    )


def _decode_motion(
    motion: torch.Tensor,
    *,
    motion_format: str,
    motion_encoding: str,
    motion_value_scale: float,
    motion_scale_x: float,
    motion_scale_y: float,
    jitter_delta_x: float,
    jitter_delta_y: float,
    width: int,
    height: int,
    runtime: dict[str, Any],
) -> torch.Tensor:
    motion = motion.float()
    if motion_encoding == "0.5-centered RG":
        motion = (motion - 0.5) * 2.0
    elif motion_encoding != "signed RG":
        raise ValueError("unsupported motion_encoding")
    motion = motion * motion_value_scale

    if motion_format == "pixel":
        return runtime["normalize_pixel_motion"](
            motion,
            scale_x=float(motion_scale_x),
            scale_y=float(motion_scale_y),
            effective_width=width,
            effective_height=height,
            jitter_dx=float(jitter_delta_x),
            jitter_dy=float(jitter_delta_y),
        )
    if motion_format == "normalized UV":
        out = torch.empty_like(motion)
        out[..., 0] = (
            motion[..., 0] * motion_scale_x
            + jitter_delta_x / width
        )
        out[..., 1] = (
            motion[..., 1] * motion_scale_y
            + jitter_delta_y / height
        )
        return out
    raise ValueError("motion_format must be 'pixel' or 'normalized UV'")


def _base_render_required(*, video: bool) -> dict[str, Any]:
    required: dict[str, Any] = {
        "dlss5_model": ("DLSS5_MODEL",),
        "image": ("IMAGE",),
    }
    if video:
        required["motion_vectors"] = ("IMAGE",)
    required.update(
        {
            "profile": (
                ["standard", "natural", "cinematic", "neutral"],
                {"default": "standard"},
            ),
            "processing_scale": (
                "FLOAT",
                {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.05},
            ),
            "intensity": (
                "FLOAT",
                {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
            ),
            "detail_strength": (
                "FLOAT",
                {"default": 1.0, "min": 0.0, "max": 8.0, "step": 0.01},
            ),
            "colour_strength": (
                "FLOAT",
                {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
            ),
            "detail_radius": (
                "FLOAT",
                {"default": 4.0, "min": 0.1, "max": 32.0, "step": 0.1},
            ),
            "frame_index": (
                "INT",
                {"default": 0, "min": 0, "max": 2147483647, "step": 1},
            ),
            "use_custom_controls": ("BOOLEAN", {"default": False}),
            "style_index": (
                "INT",
                {"default": 0, "min": 0, "max": 255, "step": 1},
            ),
            "local_tone_strength": (
                "FLOAT",
                {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
            ),
            "local_structure_strength": (
                "FLOAT",
                {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01},
            ),
            "use_auto_mask": ("BOOLEAN", {"default": False}),
            "skin_structure_strength": (
                "FLOAT",
                {"default": -1.0, "min": -1.0, "max": 4.0, "step": 0.01},
            ),
            "automatic_mask_structure_strength": (
                "FLOAT",
                {"default": -1.0, "min": -1.0, "max": 4.0, "step": 0.01},
            ),
        }
    )
    return required


class DLSS5PyTorchModelLoader:
    @classmethod
    def IS_CHANGED(cls, model_name: str, precision: str, device: str):
        stat = os.stat(_resolve_model_path(model_name))
        return (stat.st_size, stat.st_mtime_ns, precision, device)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_model_names(),),
                "precision": (["fast", "reference"], {"default": "fast"}),
                "device": (["auto", "cuda", "cpu", "mps"], {"default": "auto"}),
            }
        }

    RETURN_TYPES = ("DLSS5_MODEL",)
    RETURN_NAMES = ("dlss5_model",)
    FUNCTION = "load_model"
    CATEGORY = CATEGORY

    def load_model(self, model_name: str, precision: str, device: str):
        model_path = _resolve_model_path(model_name)
        if precision not in {"fast", "reference"}:
            raise ValueError("precision must be 'fast' or 'reference'")
        if device not in {"auto", "cuda", "cpu", "mps"}:
            raise ValueError("unsupported device")
        return (_load_pipeline(model_path, precision, device),)


class DLSS5PyTorchEnhance:
    """First-frame/still renderer with every recovered non-temporal control."""

    @classmethod
    def INPUT_TYPES(cls):
        required = _base_render_required(video=False)
        required["batch_noise_mode"] = (
            ["independent (same frame index)", "sequence (advance frame index)"],
            {"default": "independent (same frame index)"},
        )
        return {
            "required": required,
            "optional": {"control_image": ("IMAGE",)},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "enhance"
    CATEGORY = CATEGORY

    @_with_model_on_device
    def enhance(
        self,
        dlss5_model: DLSS5ModelHandle,
        image: torch.Tensor,
        profile: str,
        processing_scale: float,
        intensity: float,
        detail_strength: float,
        colour_strength: float,
        detail_radius: float,
        frame_index: int,
        use_custom_controls: bool,
        style_index: int,
        local_tone_strength: float,
        local_structure_strength: float,
        use_auto_mask: bool,
        skin_structure_strength: float,
        automatic_mask_structure_strength: float,
        batch_noise_mode: str,
        control_image: torch.Tensor | None = None,
    ):
        if not isinstance(dlss5_model, DLSS5ModelHandle):
            raise TypeError("dlss5_model must come from the DLSS 5 PyTorch Model Loader")
        if batch_noise_mode not in {
            "independent (same frame index)",
            "sequence (advance frame index)",
        }:
            raise ValueError("unsupported batch_noise_mode")

        source = _image_batch(image)
        control = _image_batch(control_image) if control_image is not None else None
        batch = source.shape[0]
        if batch == 0:
            raise ValueError("IMAGE batch must contain at least one frame")
        if control is not None and control.shape[1:3] != source.shape[1:3]:
            raise ValueError("control_image height/width must match input image")
        if control is not None and processing_scale != 1.0:
            raise ValueError("DLSS 5 control masks require processing_scale=1.0")

        runtime = _import_render_runtime()
        controls = _resolved_controls(
            profile,
            use_custom_controls,
            style_index,
            local_tone_strength,
            local_structure_strength,
            runtime,
        )
        automatic_mask = _resolved_automatic_mask(
            use_auto_mask,
            skin_structure_strength,
            automatic_mask_structure_strength,
            runtime,
        )

        progress = None
        try:
            from comfy.utils import ProgressBar
            progress = ProgressBar(batch)
        except Exception:
            pass

        outputs = torch.empty((*source.shape[:3], 3), device=model_management.intermediate_device(),
                              dtype=model_management.intermediate_dtype())
        for index in range(batch):
            model_management.throw_exception_if_processing_interrupted()
            current_frame_index = frame_index
            if batch_noise_mode == "sequence (advance frame index)":
                current_frame_index += index
            result = dlss5_model.pipeline.enhance_tensor(
                source[index],
                profile=profile,
                processing_scale=processing_scale,
                detail_strength=detail_strength,
                colour_strength=colour_strength,
                detail_radius=detail_radius,
                intensity=intensity,
                frame_index=current_frame_index,
                control_mask=_matching_frame(
                    control, index, batch, name="control_image"
                ),
                automatic_mask=automatic_mask,
                **controls,
            )
            outputs[index].copy_(result.clamp(0, 1))
            if progress is not None:
                progress.update(1)

        return (outputs,)


class DLSS5PyTorchVideoEnhance:
    """Temporal renderer: ordered IMAGE batch + current-to-previous motion vectors."""

    @classmethod
    def INPUT_TYPES(cls):
        required = _base_render_required(video=True)
        required.update(
            {
                "motion_format": (
                    ["pixel", "normalized UV"],
                    {"default": "pixel"},
                ),
                "motion_encoding": (
                    ["signed RG", "0.5-centered RG"],
                    {"default": "signed RG"},
                ),
                "motion_value_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": -4096.0, "max": 4096.0, "step": 0.01},
                ),
                "motion_scale_x": (
                    "FLOAT",
                    {"default": 1.0, "min": -16.0, "max": 16.0, "step": 0.01},
                ),
                "motion_scale_y": (
                    "FLOAT",
                    {"default": 1.0, "min": -16.0, "max": 16.0, "step": 0.01},
                ),
                "jitter_delta_x": (
                    "FLOAT",
                    {"default": 0.0, "min": -16.0, "max": 16.0, "step": 0.01},
                ),
                "jitter_delta_y": (
                    "FLOAT",
                    {"default": 0.0, "min": -16.0, "max": 16.0, "step": 0.01},
                ),
                "blend_scale": (
                    "FLOAT",
                    {"default": 0.73974609375, "min": 0.0, "max": 1.0, "step": 0.0001},
                ),
                "depth_guide": (
                    ["observed (matches DLL)", "closest-depth (experimental)"],
                    {"default": "observed (matches DLL)"},
                ),
                "depth_inverted": ("BOOLEAN", {"default": False}),
                "scene_cut_threshold": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
            }
        )
        return {
            "required": required,
            "optional": {
                "control_image": ("IMAGE",),
                "depth_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "enhance_video"
    CATEGORY = CATEGORY

    @staticmethod
    def _scene_cut(current: torch.Tensor, previous: torch.Tensor, threshold: float) -> bool:
        if threshold <= 0:
            return False
        luma_current = (
            current[..., 0] * 0.2126
            + current[..., 1] * 0.7152
            + current[..., 2] * 0.0722
        )
        luma_previous = (
            previous[..., 0] * 0.2126
            + previous[..., 1] * 0.7152
            + previous[..., 2] * 0.0722
        )
        # Only the optional cut decision needs a scalar synchronization. The
        # frames and temporal history remain on the rendering device.
        return (luma_current - luma_previous).abs().mean().item() > threshold

    @_with_model_on_device
    def enhance_video(
        self,
        dlss5_model: DLSS5ModelHandle,
        image: torch.Tensor,
        motion_vectors: torch.Tensor,
        profile: str,
        processing_scale: float,
        intensity: float,
        detail_strength: float,
        colour_strength: float,
        detail_radius: float,
        frame_index: int,
        use_custom_controls: bool,
        style_index: int,
        local_tone_strength: float,
        local_structure_strength: float,
        use_auto_mask: bool,
        skin_structure_strength: float,
        automatic_mask_structure_strength: float,
        motion_format: str,
        motion_encoding: str,
        motion_value_scale: float,
        motion_scale_x: float,
        motion_scale_y: float,
        jitter_delta_x: float,
        jitter_delta_y: float,
        blend_scale: float,
        depth_guide: str,
        depth_inverted: bool,
        scene_cut_threshold: float,
        control_image: torch.Tensor | None = None,
        depth_image: torch.Tensor | None = None,
    ):
        if not isinstance(dlss5_model, DLSS5ModelHandle):
            raise TypeError("dlss5_model must come from the DLSS 5 PyTorch Model Loader")
        if processing_scale != 1.0:
            raise ValueError("temporal DLSS 5 currently requires processing_scale=1.0")

        source = _image_batch(image)
        motion = _motion_batch(motion_vectors)
        control = _image_batch(control_image) if control_image is not None else None
        depth = _depth_batch(depth_image) if depth_image is not None else None
        batch, height, width, _ = source.shape
        if batch == 0:
            raise ValueError("IMAGE batch must contain at least one frame")
        if motion.shape[1:3] != (height, width):
            raise ValueError("motion_vectors height/width must match input image")
        if control is not None and control.shape[1:3] != (height, width):
            raise ValueError("control_image height/width must match input image")
        if depth is not None and depth.shape[1:3] != (height, width):
            raise ValueError("depth_image height/width must match input image")
        if depth_guide == "closest-depth (experimental)" and depth is None:
            raise ValueError("closest-depth mode requires depth_image")

        runtime = _import_render_runtime()
        controls = _resolved_controls(
            profile,
            use_custom_controls,
            style_index,
            local_tone_strength,
            local_structure_strength,
            runtime,
        )
        automatic_mask = _resolved_automatic_mask(
            use_auto_mask,
            skin_structure_strength,
            automatic_mask_structure_strength,
            runtime,
        )
        depth_mode = (
            "closest" if depth_guide == "closest-depth (experimental)" else "observed"
        )

        progress = None
        try:
            from comfy.utils import ProgressBar
            progress = ProgressBar(batch)
        except Exception:
            pass

        outputs = torch.empty((batch, height, width, 3), device=model_management.intermediate_device(),
                              dtype=model_management.intermediate_dtype())
        history: torch.Tensor | None = None
        previous: torch.Tensor | None = None
        sequence_index = 0
        device = dlss5_model.pipeline.device

        for index in range(batch):
            model_management.throw_exception_if_processing_interrupted()
            frame = source[index].to(device=device, dtype=torch.float32, non_blocking=True)
            if previous is not None and self._scene_cut(
                frame, previous, float(scene_cut_threshold)
            ):
                history = None
                previous = None
                sequence_index = 0

            current_frame_index = int(frame_index) + sequence_index
            control_frame = _matching_frame(
                control, index, batch, name="control_image"
            )
            depth_frame = _matching_frame(depth, index, batch, name="depth_image")
            if control_frame is not None:
                control_frame = control_frame.to(device=device, dtype=torch.float32, non_blocking=True)
            if depth_frame is not None:
                depth_frame = depth_frame.to(device=device, dtype=torch.float32, non_blocking=True)
            geometry = runtime["NetworkGeometry"].vendor_aligned(width, height)

            if history is None:
                features = runtime["make_features"](
                    frame,
                    frame_index=current_frame_index,
                    geometry=geometry,
                    noise_tables=dlss5_model.pipeline.noise_tables(),
                    automatic_mask=automatic_mask,
                    control_mask=control_frame,
                    **controls,
                )
                head = geometry.crop(dlss5_model.pipeline.run_features_tensor(features))
                history = runtime["compose_head"](
                    head,
                    frame,
                )
            else:
                raw_motion = _matching_motion_frame(motion, index, batch).to(device=device, dtype=torch.float32, non_blocking=True)
                normalized_motion = _decode_motion(
                    raw_motion,
                    motion_format=motion_format,
                    motion_encoding=motion_encoding,
                    motion_value_scale=float(motion_value_scale),
                    motion_scale_x=float(motion_scale_x),
                    motion_scale_y=float(motion_scale_y),
                    jitter_delta_x=float(jitter_delta_x),
                    jitter_delta_y=float(jitter_delta_y),
                    width=width,
                    height=height,
                    runtime=runtime,
                )
                network_features = runtime["make_temporal_features"](
                    frame,
                    history,
                    normalized_motion,
                    frame_index=current_frame_index,
                    geometry=geometry,
                    noise_tables=dlss5_model.pipeline.noise_tables(),
                    depth=depth_frame,
                    depth_guide=depth_mode,
                    depth_inverted=bool(depth_inverted),
                    automatic_mask=automatic_mask,
                    control_mask=control_frame,
                    **controls,
                )
                head = geometry.crop(
                    dlss5_model.pipeline.run_features_tensor(network_features)
                )
                history = runtime["compose_temporal"](
                    head,
                    frame,
                    geometry.crop(network_features),
                    blend_scale=float(blend_scale),
                    reference_tables=dlss5_model.pipeline.noise_tables(),
                )

            displayed = runtime["compose_detail"](
                frame,
                history,
                detail_strength=float(detail_strength),
                colour_strength=float(colour_strength),
                radius=float(detail_radius),
            )
            displayed = runtime["blend_effect"](frame, displayed, control_mask=control_frame,
                                                 intensity=float(intensity))
            outputs[index].copy_(displayed.clamp(0, 1))
            previous = frame
            sequence_index += 1
            if progress is not None:
                progress.update(1)

        return (outputs,)


class DLSS5PyTorchClearCache:
    @classmethod
    def IS_CHANGED(cls, clear: bool):
        return float("nan") if clear else False

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"clear": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ()
    FUNCTION = "clear_cache"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    def clear_cache(self, clear: bool):
        if clear:
            _clear_pipeline_cache()
        return ()


NODE_CLASS_MAPPINGS = {
    "DLSS5PyTorchModelLoader": DLSS5PyTorchModelLoader,
    "DLSS5PyTorchEnhance": DLSS5PyTorchEnhance,
    "DLSS5PyTorchVideoEnhance": DLSS5PyTorchVideoEnhance,
    "DLSS5PyTorchClearCache": DLSS5PyTorchClearCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DLSS5PyTorchModelLoader": "DLSS 5 PyTorch Model Loader",
    "DLSS5PyTorchEnhance": "DLSS 5 PyTorch Neural Rendering",
    "DLSS5PyTorchVideoEnhance": "DLSS 5 PyTorch Video Neural Rendering",
    "DLSS5PyTorchClearCache": "DLSS 5 PyTorch Clear Cache",
}
