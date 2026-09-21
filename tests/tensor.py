"""Tensor/reference parity: python tests/tensor.py [--device cuda]."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dlss5 import composition, features, temporal, tensor_ops as ops


class TensorParityTests(unittest.TestCase):
    device = "cpu"

    @classmethod
    def setUpClass(cls):
        cls.tables = ops.ReferenceNoiseTables().to(cls.device)

    @classmethod
    def tearDownClass(cls):
        del cls.tables

    def setUp(self):
        self.rng = np.random.default_rng(41)

    def tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=torch.float32)

    def same(self, expected, actual, *, atol=0, rtol=0):
        self.assertEqual(actual.device.type, self.device)
        np.testing.assert_allclose(actual.cpu().numpy(), expected, atol=atol, rtol=rtol)

    def test_reference_noise_is_exact_across_frames_and_geometry(self):
        for frame in (0, 17, 2147483650):
            self.same(features.deterministic_noise(320, 384, frame),
                      ops.deterministic_noise(320, 384, frame, device=self.device, tables=self.tables))

    def test_feature_channels_masks_and_mirrored_padding(self):
        color = self.rng.random((19, 27, 3), dtype=np.float32)
        mask = self.rng.random(color.shape, dtype=np.float32)
        geometry = features.NetworkGeometry.vendor_aligned(27, 19)
        for automatic in (None, features.AutomaticMask(-1, -1), features.AutomaticMask(0.7, -1)):
            for control in (None, mask):
                options = dict(geometry=geometry, frame_index=5, automatic_mask=automatic,
                               normalized_style=0.17, local_tone_strength=1.13, local_structure_strength=0.81)
                self.same(features.make_features(color, control_mask=control, **options),
                          ops.make_features(self.tensor(color),
                                            control_mask=None if control is None else self.tensor(control),
                                            noise_tables=self.tables, **options))

    def test_lanczos_resize_including_edges_and_degenerate_dimensions(self):
        for shape in ((19, 27, 3), (1, 7, 3), (5, 1, 3)):
            color = self.rng.random(shape, dtype=np.float32)
            for width, height in ((41, 37), (14, 11), (108, 76), (1, 1)):
                self.same(composition.resample(color, width, height),
                          ops.resample(self.tensor(color), width, height), atol=4e-6)

    def test_integer_box_downsample(self):
        color = self.rng.random((32, 48, 3), dtype=np.float32)
        self.same(composition.resample(color, 12, 8), ops.resample(self.tensor(color), 12, 8), atol=2e-7)

    def test_detail_and_masked_head_composition(self):
        color = self.rng.random((19, 27, 3), dtype=np.float32)
        output = self.rng.random(color.shape, dtype=np.float32)
        head = self.rng.normal(size=(19, 27, 4)).astype(np.float32)
        self.same(composition.compose_head(head, color, control_mask=output, intensity=.7),
                  ops.compose_head(self.tensor(head), self.tensor(color), control_mask=self.tensor(output), intensity=.7))
        for radius in (.1, 4, 32):
            self.same(composition.compose_detail(color, output, detail_strength=1.5, colour_strength=.7, radius=radius),
                      ops.compose_detail(self.tensor(color), self.tensor(output), detail_strength=1.5,
                                         colour_strength=.7, radius=radius), atol=1e-6)

    def test_history_sampling_fractional_motion_and_border_clamping(self):
        history = self.rng.random((23, 31, 3), dtype=np.float32)
        u = self.rng.uniform(-.2, 1.2, (23, 31)).astype(np.float32)
        v = self.rng.uniform(-.2, 1.2, (23, 31)).astype(np.float32)
        self.same(temporal.sample_history(history, u, v),
                  ops.sample_history(self.tensor(history), self.tensor(u), self.tensor(v)), atol=3e-6)

    def test_temporal_features_and_depth_modes(self):
        color = self.rng.random((17, 29, 3), dtype=np.float32)
        history = self.rng.random(color.shape, dtype=np.float32)
        motion = self.rng.uniform(-.15, .15, (17, 29, 2)).astype(np.float32)
        depth = self.rng.random((17, 29, 1), dtype=np.float32)
        for mode in ("observed", "closest"):
            for inverted in (False, True):
                expected = temporal.make_temporal_features(color, history, motion, frame_index=1,
                                                           depth=depth, depth_guide=mode, depth_inverted=inverted)
                actual = ops.make_temporal_features(self.tensor(color), self.tensor(history), self.tensor(motion),
                                                    frame_index=1, depth=self.tensor(depth), depth_guide=mode,
                                                    depth_inverted=inverted, noise_tables=self.tables)
                self.same(expected, actual)

    def test_temporal_composition(self):
        color = self.rng.random((17, 29, 3), dtype=np.float32)
        head = self.rng.normal(size=(17, 29, 4)).astype(np.float32)
        inputs = features.make_features(color)
        self.same(temporal.compose_temporal(head, color, inputs),
                  ops.compose_temporal(self.tensor(head), self.tensor(color), self.tensor(inputs),
                                        reference_tables=self.tables))

    def test_temporal_strip_boundary_preserves_features(self):
        color = self.rng.random((480, 640, 3), dtype=np.float32)
        history = self.rng.random(color.shape, dtype=np.float32)
        motion = self.rng.uniform(-.01, .01, (480, 640, 2)).astype(np.float32)
        self.same(temporal.make_temporal_features(color, history, motion, frame_index=7),
                  ops.make_temporal_features(self.tensor(color), self.tensor(history), self.tensor(motion),
                                             frame_index=7, noise_tables=self.tables))

    def test_motion_normalization_with_jitter(self):
        motion = self.rng.normal(size=(17, 29, 2)).astype(np.float32)
        options = dict(scale_x=-2.3, scale_y=.7, effective_width=1920, effective_height=1080,
                       jitter_dx=.13, jitter_dy=-.41)
        self.same(temporal.normalize_pixel_motion(motion, **options),
                  ops.normalize_pixel_motion(self.tensor(motion), **options))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args, rest = parser.parse_known_args()
    TensorParityTests.device = args.device
    unittest.main(argv=[sys.argv[0], *rest])
