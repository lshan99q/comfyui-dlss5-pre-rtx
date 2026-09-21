"""Dependency-light smoke test for the self-contained ComfyUI nodes.

Run from the repository root with: python tests/smoke.py
"""
from __future__ import annotations

import ast
import importlib.util
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


class FakeFolderPaths(types.ModuleType):
    def __init__(self, model_root: Path):
        super().__init__("folder_paths")
        self.models_dir = str(model_root)
        self.folder_names_and_paths = {}

    def add_model_folder_path(self, name, path, is_default=False):
        del is_default
        self.folder_names_and_paths.setdefault(name, ([path], set()))

    def get_filename_list(self, name):
        paths, _ = self.folder_names_and_paths[name]
        out = []
        for base in paths:
            base = Path(base)
            if base.exists():
                out.extend(
                    str(p.relative_to(base)).replace("\\", "/")
                    for p in base.rglob("*")
                    if p.is_file()
                )
        return sorted(out)

    def get_full_path(self, name, filename):
        paths, _ = self.folder_names_and_paths[name]
        for base in paths:
            candidate = Path(base) / filename
            if candidate.is_file():
                return str(candidate)
        return None


class FakePipeline:
    loads = []

    def __init__(self, model_path, device, precision):
        self.model_path = model_path
        self.device = torch.device("cpu")
        self.precision = precision
        self.calls = []
        self.feature_calls = []
        self.model = torch.nn.Linear(1, 1)

    @classmethod
    def from_safetensors(cls, model_path, *, device="auto", precision="reference", offload_device=None):
        cls.loads.append((str(model_path), device, precision))
        return cls(str(model_path), device, precision)

    def enhance_tensor(self, image, **kwargs):
        self.calls.append(kwargs)
        return (image.float() + 0.1).clamp(0, 1)

    def run_features_tensor(self, features):
        self.feature_calls.append(features)
        return features.new_zeros((*features.shape[:2], 4))

    def noise_tables(self):
        return None


def install_comfy_stubs(model_root: Path):
    fp = FakeFolderPaths(model_root)
    sys.modules["folder_paths"] = fp

    comfy = types.ModuleType("comfy")
    comfy_utils = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, total):
            self.total = total
            self.value = 0

        def update(self, amount):
            self.value += amount

    comfy_utils.ProgressBar = ProgressBar
    management = types.ModuleType("comfy.model_management")
    management.get_torch_device = lambda: torch.device("cpu")
    management.intermediate_device = lambda: torch.device("cpu")
    management.intermediate_dtype = lambda: torch.float32
    management.module_size = lambda model: 4
    management.free_memory = lambda memory, device: None
    management.soft_empty_cache = lambda: None
    management.throw_exception_if_processing_interrupted = lambda: None
    comfy.model_management = management
    sys.modules["comfy.model_management"] = management
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = comfy_utils
    return fp


def load_nodes(model_root: Path):
    install_comfy_stubs(model_root)
    spec = importlib.util.spec_from_file_location("dlss5_nodes_smoke", REPO / "nodes.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module._import_pipeline_class = lambda: FakePipeline
    return module


def make_model(nodes, name="model.safetensors"):
    path = Path(nodes._MODEL_DIR) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")
    return path


@dataclass(frozen=True)
class FakeAutomaticMask:
    skin_structure_strength: float
    automatic_mask_structure_strength: float


class FakeGeometry:
    def __init__(self, width, height):
        self.network_width = width
        self.network_height = height

    @classmethod
    def vendor_aligned(cls, width, height):
        return cls(width, height)

    def crop(self, array):
        return array


def make_fake_runtime(captured_motion: list[np.ndarray]):
    profiles = {
        "standard": {
            "normalized_style": 0.0,
            "local_tone_strength": 1.0,
            "local_structure_strength": 1.0,
        },
        "natural": {
            "normalized_style": 1.0 / 128.0,
            "local_tone_strength": 1.0,
            "local_structure_strength": 1.0,
        },
        "cinematic": {
            "normalized_style": 2.0 / 128.0,
            "local_tone_strength": 1.0,
            "local_structure_strength": 1.0,
        },
        "neutral": {
            "normalized_style": 0.0,
            "local_tone_strength": 0.0,
            "local_structure_strength": 0.0,
        },
    }

    def make_features(color, **kwargs):
        del kwargs
        h, w = color.shape[:2]
        return color.new_zeros((h, w, 16))

    def compose_head(head, color, **kwargs):
        del head, kwargs
        return (color + 0.1).clamp(0, 1)

    def blend_effect(color, output, control_mask, intensity):
        blend = intensity if control_mask is None else control_mask[..., :1] * intensity
        return (color + blend * (output - color)).clamp(0, 1)

    def compose_detail(source, output, **kwargs):
        del source, kwargs
        return output.float()

    def normalize_pixel_motion(motion, **kwargs):
        width = kwargs["effective_width"]
        height = kwargs["effective_height"]
        out = torch.empty_like(motion)
        out[..., 0] = (
            motion[..., 0] * kwargs["scale_x"] + kwargs["jitter_dx"]
        ) / width
        out[..., 1] = (
            motion[..., 1] * kwargs["scale_y"] + kwargs["jitter_dy"]
        ) / height
        return out

    def make_temporal_features(color, history, motion, **kwargs):
        del history, kwargs
        captured_motion.append(motion.cpu().numpy().copy())
        h, w = color.shape[:2]
        return color.new_zeros((h, w, 16))

    def extend_features(features, geometry, frame_index):
        del geometry, frame_index
        return features

    def compose_temporal(head, color, features, **kwargs):
        del head, features, kwargs
        return (color + 0.2).clamp(0, 1)

    return {
        "blend_effect": blend_effect,
        "AutomaticMask": FakeAutomaticMask,
        "NetworkGeometry": FakeGeometry,
        "PROFILES": profiles,
        "make_features": make_features,
        "compose_head": compose_head,
        "compose_detail": compose_detail,
        "BLEND_SCALE": 0.73974609375,
        "compose_temporal": compose_temporal,
        "extend_features": extend_features,
        "make_temporal_features": make_temporal_features,
        "normalize_pixel_motion": normalize_pixel_motion,
    }


def assert_self_contained():
    required = [
        REPO / "dlss5" / "__init__.py",
        REPO / "dlss5" / "model.py",
        REPO / "dlss5" / "pipeline.py",
        REPO / "dlss5" / "features.py",
        REPO / "dlss5" / "composition.py",
        REPO / "dlss5" / "temporal.py",
    ]
    assert all(path.is_file() for path in required), "self-contained runtime is incomplete"

    for path in REPO.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                assert all(alias.name.split(".")[0] != "mlxdlss" for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                assert (node.module or "").split(".")[0] != "mlxdlss"

    dependency_text = (
        (REPO / "requirements.txt").read_text(encoding="utf-8")
        + (REPO / "pyproject.toml").read_text(encoding="utf-8")
    ).lower()
    assert "mlxdlss" not in dependency_text
    assert "git+https://github.com/iamwavecut/mlx-dlss" not in dependency_text

    sys.path.insert(0, str(REPO))
    try:
        from dlss5.pipeline import NeuralRenderingPipeline
        from dlss5.temporal import normalize_pixel_motion

        assert NeuralRenderingPipeline is not None
        motion = np.ones((2, 4, 2), dtype=np.float32)
        normalized = normalize_pixel_motion(
            motion,
            scale_x=1,
            scale_y=-1,
            effective_width=4,
            effective_height=2,
        )
        assert np.allclose(normalized[..., 0], 0.25)
        assert np.allclose(normalized[..., 1], -0.5)
    finally:
        sys.path.pop(0)


def main():
    assert_self_contained()

    with tempfile.TemporaryDirectory() as td:
        nodes = load_nodes(Path(td) / "models")
        FakePipeline.loads.clear()
        captured_motion: list[np.ndarray] = []
        fake_runtime = make_fake_runtime(captured_motion)
        nodes._import_render_runtime = lambda: fake_runtime

        make_model(nodes, "a.safetensors")
        (Path(nodes._MODEL_DIR) / "ignore.txt").write_text("x")
        assert nodes._model_names() == ["a.safetensors"]

        assert "DLSS5PyTorchEnhance" in nodes.NODE_CLASS_MAPPINGS
        assert "DLSS5PyTorchVideoEnhance" in nodes.NODE_CLASS_MAPPINGS

        loader = nodes.DLSS5PyTorchModelLoader()
        first = loader.load_model("a.safetensors", "fast", "auto")[0]
        second = loader.load_model("a.safetensors", "fast", "auto")[0]
        assert first is second
        assert len(FakePipeline.loads) == 1

        image = torch.zeros((2, 8, 9, 3), dtype=torch.float32)
        still = nodes.DLSS5PyTorchEnhance().enhance(
            first,
            image,
            "standard",
            1.0,
            1.0,
            1.0,
            1.0,
            4.0,
            7,
            True,
            64,
            1.25,
            0.8,
            True,
            -1.0,
            0.5,
            "sequence (advance frame index)",
        )[0]
        assert still.shape == image.shape
        assert torch.allclose(still, torch.full_like(still, 0.1))
        calls = first.pipeline.calls[-2:]
        assert [c["frame_index"] for c in calls] == [7, 8]
        assert abs(calls[0]["normalized_style"] - 0.5) < 1e-9
        assert abs(calls[0]["local_tone_strength"] - 1.25) < 1e-9
        assert calls[0]["automatic_mask"].automatic_mask_structure_strength == 0.5

        control = torch.ones((1, 8, 9, 3), dtype=torch.float32)
        nodes.DLSS5PyTorchEnhance().enhance(
            first,
            image,
            "standard",
            1.0,
            1.0,
            1.0,
            1.0,
            4.0,
            0,
            False,
            0,
            1.0,
            1.0,
            False,
            -1.0,
            -1.0,
            "independent (same frame index)",
            control,
        )
        assert all(c["control_mask"].shape == (8, 9, 3) for c in first.pipeline.calls[-2:])

        video = torch.zeros((3, 8, 9, 3), dtype=torch.float32)
        # Two transition fields: frame 1->0 and frame 2->1, each +2 px X.
        motion = torch.zeros((2, 8, 9, 3), dtype=torch.float32)
        motion[..., 0] = 2.0
        video_out = nodes.DLSS5PyTorchVideoEnhance().enhance_video(
            first,
            video,
            motion,
            "standard",
            1.0,
            1.0,
            1.0,
            1.0,
            4.0,
            0,
            False,
            0,
            1.0,
            1.0,
            False,
            -1.0,
            -1.0,
            "pixel",
            "signed RG",
            1.0,
            1.0,
            1.0,
            0.0,
            0.0,
            0.73974609375,
            "observed (matches DLL)",
            False,
            0.0,
        )[0]
        assert video_out.shape == video.shape
        assert torch.allclose(video_out[0], torch.full_like(video_out[0], 0.1))
        assert torch.allclose(video_out[1:], torch.full_like(video_out[1:], 0.2))
        assert len(captured_motion) == 2
        assert np.allclose(captured_motion[0][..., 0], 2.0 / 9.0)
        assert np.allclose(captured_motion[0][..., 1], 0.0)

        bad_motion = torch.zeros((2, 7, 9, 3), dtype=torch.float32)
        try:
            nodes.DLSS5PyTorchVideoEnhance().enhance_video(
                first,
                video,
                bad_motion,
                "standard",
                1.0,
                1.0,
                1.0,
                1.0,
                4.0,
                0,
                False,
                0,
                1.0,
                1.0,
                False,
                -1.0,
                -1.0,
                "pixel",
                "signed RG",
                1.0,
                1.0,
                1.0,
                0.0,
                0.0,
                0.73974609375,
                "observed (matches DLL)",
                False,
                0.0,
            )
        except ValueError as exc:
            assert "height/width" in str(exc)
        else:
            raise AssertionError("mismatched motion geometry was accepted")

    print("smoke tests: PASS (still + temporal video, self-contained pure-PyTorch runtime)")


if __name__ == "__main__":
    main()
