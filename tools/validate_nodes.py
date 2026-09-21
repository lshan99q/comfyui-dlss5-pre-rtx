"""Exercise real ComfyUI node loading/lifecycle and compare a short video.

No server is started, no workflow is submitted, and no model files are changed.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]


def load_module(name, path, package=False):
    spec = importlib.util.spec_from_file_location(name, path,
        submodule_search_locations=[str(path.parent)] if package else None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def defaults(node, **inputs):
    for name, definition in node.INPUT_TYPES()["required"].items():
        if name in inputs:
            continue
        choices = definition[0]
        options = definition[1] if len(definition) > 1 else {}
        inputs[name] = options.get("default", choices[0] if isinstance(choices, list) else None)
    return inputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", type=Path, default=REPO.parent.parent)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.comfy_root))
    sys.path.insert(0, str(REPO))
    nodes = load_module("dlss5_actual_nodes", REPO / "nodes.py")
    baseline = load_module("dlss5_baseline", args.reference / "dlss5/__init__.py", package=True)
    from dlss5_baseline import features as old_features, temporal as old_temporal, composition as old_composition

    handle = nodes.DLSS5PyTorchModelLoader().load_model(args.weights.name, "fast", "auto")[0]
    assert handle.pipeline.device.type == "cuda"
    assert all(t.device.type == "cpu" for t in handle.pipeline.model.buffers())
    image = torch.from_numpy(np.random.default_rng(73).random((3, 320, 320, 3), dtype=np.float32))
    motion = torch.zeros(2, 320, 320, 3)
    motion[..., 0], motion[..., 1] = 0.35, -0.2
    params = defaults(nodes.DLSS5PyTorchVideoEnhance, image=image, motion_vectors=motion, dlss5_model=handle)
    actual = nodes.DLSS5PyTorchVideoEnhance().enhance_video(**params)[0].cpu().numpy()
    assert all(t.device.type == "cpu" for t in handle.pipeline.model.buffers())
    reference = baseline.NeuralRenderingPipeline.from_safetensors(args.weights, device="cuda", precision="fast")
    history = None
    expected = []
    with torch.inference_mode():
        for index, frame in enumerate(image.numpy()):
            geometry = old_features.NetworkGeometry.vendor_aligned(320, 320)
            if history is None:
                features = old_features.make_features(frame, frame_index=index, geometry=geometry)
                head = geometry.crop(reference.run_features(features))
                history = old_composition.compose_head(head, frame)
            else:
                uv = old_temporal.normalize_pixel_motion(motion[index - 1, ..., :2].numpy(),
                    scale_x=1, scale_y=1, effective_width=320, effective_height=320)
                features = old_temporal.make_temporal_features(frame, history, uv, frame_index=index)
                network = old_temporal.extend_features(features, geometry, index)
                head = geometry.crop(reference.run_features(network))
                history = old_temporal.compose_temporal(head, frame, features)
            expected.append(history.clip(0, 1).copy())
    for index, (expected_frame, actual_frame) in enumerate(zip(expected, actual)):
        diff = np.abs(expected_frame - actual_frame)
        print(f"video frame {index}: max={diff.max():.9g} mean={diff.mean():.9g}", flush=True)
        np.testing.assert_allclose(actual_frame, expected_frame, atol=2e-6, rtol=0)
    reference.model.cpu()
    still_params = defaults(nodes.DLSS5PyTorchEnhance, image=image[:1], dlss5_model=handle)
    result = nodes.DLSS5PyTorchEnhance().enhance(**still_params)[0]
    torch.testing.assert_close(result, torch.from_numpy(actual[:1]), atol=2e-6, rtol=0)
    assert all(t.device.type == "cpu" for t in handle.pipeline.model.buffers())
    nodes.DLSS5PyTorchClearCache().clear_cache(True)
    assert not nodes._PIPELINE_CACHE
    print("actual ComfyUI loader, still/video rendering, CPU offload, and clear-cache: PASS")


if __name__ == "__main__":
    main()
