"""Audit regressions: python tests/regression.py (no checkpoint or GPU needed)."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dlss5.pipeline import NeuralRenderingPipeline, load_weights, validate_weights
from dlss5.model import NeuralRenderingModel, cosine_attention, vendor_approximate_softmax
from smoke import load_nodes, make_model
from tools.validate_nodes import defaults


class WeightValidationTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / "dlss5/weight_spec.json"
        self.spec = json.loads(path.read_text())["tensors"]
        self.weights = {
            name: torch.empty(entry["shape"], device="meta")
            for name, entry in self.spec.items()
        }

    def test_complete_shapes_accepted(self):
        validate_weights(self.weights)

    def test_missing_non_sentinel_tensor_rejected(self):
        del self.weights["block1.layer0.weight1"]
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_weights(self.weights)

    def test_shape_and_dtype_rejected(self):
        name = next(iter(self.weights))
        self.weights[name] = torch.empty(1, device="meta")
        with self.assertRaisesRegex(ValueError, "expected shape"):
            validate_weights(self.weights)
        self.weights[name] = torch.empty(self.spec[name]["shape"], device="meta", dtype=torch.int32)
        with self.assertRaisesRegex(ValueError, "floating-point"):
            validate_weights(self.weights)

    def test_file_loader_rejects_old_six_tensor_loophole(self):
        names = ["block0.layer0.input_adapter_weight", "block0.layer0.qkv_weight",
                 "block30.layer4.weight", "block39.layer0.conv_weight",
                 "block70.layer0.out_gain", "block70.layer0.out_conv_weight"]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "invalid.safetensors"
            save_file({name: torch.zeros(1) for name in names}, str(path),
                      metadata={"format": "dlssnr-logical-v18", "fully_logical": "true"})
            with self.assertRaisesRegex(ValueError, "missing"):
                load_weights(path)

    def test_direct_pipeline_validates_before_constructing_model(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            NeuralRenderingPipeline({}, device="cpu")


class NodeLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.nodes = load_nodes(Path(self.temp.name))
        self.path = make_model(self.nodes)
        self.handle = self.nodes.DLSS5PyTorchModelLoader().load_model(
            "model.safetensors", "fast", "auto")[0]
        self.moves = []
        self.handle.pipeline.model.to = lambda device: self.moves.append(str(device))
        # Movement is recorded, so lifecycle tests can exercise a CUDA target
        # without requiring or allocating on a physical GPU.
        self.handle.pipeline.device = torch.device("cuda")

    def test_offloads_after_success_and_failure(self):
        for fail in (False, True):
            self.moves.clear()

            @self.nodes._with_model_on_device
            def render(instance, dlss5_model, image, processing_scale):
                if fail:
                    raise RuntimeError("simulated OOM")
                return image

            image = torch.zeros(1, 8, 9, 3)
            if fail:
                with self.assertRaisesRegex(RuntimeError, "simulated OOM"):
                    render(None, self.handle, image, 1.0)
            else:
                self.assertIs(render(None, self.handle, image, 1.0), image)
            self.assertEqual(self.moves, ["cuda", "cpu"])
            self.assertIsNone(self.handle.pipeline.model.interrupt_check)

    def test_clear_offloads_handle_retained_outside_private_cache(self):
        self.nodes._PIPELINE_CACHE.clear()
        self.nodes.DLSS5PyTorchClearCache().clear_cache(True)
        self.assertEqual(self.moves, ["cpu"])
        self.assertTrue(math.isnan(self.nodes.DLSS5PyTorchClearCache.IS_CHANGED(True)))

    def test_loader_invalidates_replaced_file(self):
        loader = self.nodes.DLSS5PyTorchModelLoader
        old = loader.IS_CHANGED("model.safetensors", "fast", "auto")
        stat = self.path.stat()
        os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        self.assertNotEqual(old, loader.IS_CHANGED("model.safetensors", "fast", "auto"))

    def test_intensity_range_matches_blend_contract(self):
        for cls in (self.nodes.DLSS5PyTorchEnhance, self.nodes.DLSS5PyTorchVideoEnhance):
            self.assertEqual(cls.INPUT_TYPES()["required"]["intensity"][1]["max"], 1.0)

    def test_model_checks_cancellation_before_each_block_family(self):
        model = NeuralRenderingModel({})
        def interrupt():
            raise RuntimeError("cancelled")
        model.interrupt_check = interrupt
        for call in (lambda: model._window(None, 0, head_count=1),
                     lambda: model._split_window(None, 23), lambda: model._global(None, 31)):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                call()


class RenderingCorrectnessTests(unittest.TestCase):
    def devices(self):
        return ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)

    def pipeline(self, device):
        pipeline = object.__new__(NeuralRenderingPipeline)
        pipeline.device = torch.device(device)
        pipeline.noise_tables = lambda: None
        def head(features):
            result = features.new_zeros((*features.shape[:2], 4))
            result[..., :3] = 0.5
            return result
        pipeline.run_features_tensor = head
        pipeline.run_features = lambda features: head(torch.from_numpy(features)).numpy()
        return pipeline

    def test_global_attention_probabilities_survive_long_rows(self):
        for device in self.devices():
            for dtype in (torch.float16, torch.float32):
                for count in (1024, 2160, 16384):
                    for logit in (0.0, 3.0, 6.0):
                        with self.subTest(device=device, dtype=dtype, count=count, logit=logit):
                            p = vendor_approximate_softmax(torch.full((1, count), logit, device=device, dtype=dtype))
                            self.assertEqual(p.dtype, dtype)
                            self.assertTrue(torch.isfinite(p).all())
                            self.assertTrue((p > 0).all())
                            self.assertAlmostEqual(p.float().sum().item(), 1.0, delta=5e-4)
                p = vendor_approximate_softmax(torch.zeros(1, 64, device=device, dtype=dtype))
                torch.testing.assert_close(p, torch.full_like(p, 1 / 64), atol=0, rtol=0)

    def test_global_attention_retains_constant_values(self):
        for device in self.devices():
            value = torch.ones(1, 1024, 32, device=device)
            identity = torch.eye(32, device=device)
            output = cosine_attention(value, qkv_weight=torch.cat((identity * 0, identity * 0, identity), dim=1),
                                      attention_scale=torch.ones(1, device=device),
                                      attention_bias=torch.zeros(1, 1024, 1024, device=device),
                                      projection_weight=identity, head_count=1)
            torch.testing.assert_close(output, value, atol=0, rtol=0)

    def test_still_final_blend_after_resize_and_detail(self):
        source = torch.rand(13, 17, 3, generator=torch.Generator().manual_seed(17))
        mask = torch.zeros_like(source)
        mask[4:9, 5:12, 0] = 0.5
        mask[..., 1:] = 1
        for device in self.devices():
            pipeline = self.pipeline(device)
            for scale, control in ((2.0, None), (1.5, None), (1.0, mask)):
                options = dict(processing_scale=scale, detail_strength=2, colour_strength=0.5)
                full_control = None if control is None else control.clone()
                if full_control is not None:
                    full_control[..., 0] = 1
                full = pipeline.enhance_tensor(source, control_mask=full_control, **options).cpu()
                for intensity in (0.0, 0.5, 1.0):
                    blend = intensity if control is None else control[..., :1] * intensity
                    expected = source + blend * (full - source)
                    result = pipeline.enhance_tensor(source, control_mask=control, intensity=intensity, **options).cpu()
                    torch.testing.assert_close(result, expected, atol=2e-7, rtol=0)
                    reference = pipeline.enhance(source.numpy(), control_mask=None if control is None else control.numpy(),
                                                 intensity=intensity, **options).image
                    np.testing.assert_allclose(reference, expected.numpy(), atol=2e-6, rtol=0)

    def test_video_final_blend_does_not_feed_back_into_history(self):
        with tempfile.TemporaryDirectory() as directory:
            nodes = load_nodes(Path(directory))
            image = torch.full((3, 13, 17, 3), 0.5)
            mask = torch.zeros_like(image[:1])
            mask[:, 4:9, 5:12, 0] = 0.5
            mask[..., 1:] = 1
            full_mask = mask.clone()
            full_mask[..., 0] = 1
            for device in self.devices():
                handle = nodes.DLSS5ModelHandle(self.pipeline(device), "synthetic", "reference", device)
                node = nodes.DLSS5PyTorchVideoEnhance()
                options = defaults(type(node), dlss5_model=handle, image=image,
                                   motion_vectors=torch.zeros_like(image[1:]),
                                   detail_strength=2.0, colour_strength=0.5)
                # Exercise real video math with a synthetic head; lifecycle has separate tests.
                render = type(node).enhance_video.__wrapped__
                full = render(node, **options, control_image=full_mask)[0]
                for intensity in (0.0, 0.5, 1.0):
                    options["intensity"] = intensity
                    output = render(node, **options, control_image=mask)[0]
                    expected = image + mask[..., :1] * intensity * (full - image)
                    torch.testing.assert_close(output, expected, atol=2e-7, rtol=0)


if __name__ == "__main__":
    unittest.main()
