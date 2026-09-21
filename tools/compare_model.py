"""Compare individual optimized operations and teacher-forced blocks on CUDA."""
import argparse
import importlib.util
from pathlib import Path
import sys

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dlss5 import model, tensor_ops, features
from dlss5.pipeline import load_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("baseline_model", args.reference / "dlss5/model.py")
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    weights = load_weights(args.weights)
    old = baseline.NeuralRenderingModel(weights).half().cuda().eval()
    new = model.NeuralRenderingModel(weights).half().cuda().eval()
    torch.manual_seed(1)

    def compare(label, expected, actual):
        diff = (actual.float() - expected.float()).abs()
        print(label, "max", diff.max().item(), "mean", diff.mean().item(),
              "mismatch", (diff != 0).float().mean().item(), flush=True)

    with torch.inference_mode():
        x = torch.randn(3, 4, 64, 32, device="cuda", dtype=torch.float16)
        compare("normalization", baseline.vendor_cosine_normalize(x), model.vendor_cosine_normalize(x))
        x = torch.linspace(-500, 500, 100000, device="cuda").half()
        compare("e4m3", baseline.e4m3_round_trip(x), model.e4m3_round_trip(x))
        compare("noise", torch.from_numpy(features.deterministic_noise(320, 320)).cuda(),
                tensor_ops.deterministic_noise(320, 320, device="cuda"))
        for index, heads in ((0, 1), (5, 2), (9, 4), (15, 8)):
            x = torch.randn(1, 24, 32, heads * 32, device="cuda", dtype=torch.float16) * 0.1
            compare(f"window {index}", old._window(x, index, head_count=heads),
                    new._window(x, index, head_count=heads))
            prefix = f"block{index}.layer0"
            if f"{prefix}.ffn_expand_weight" in weights:
                options = dict(expansion_weight=old.weight(f"{prefix}.ffn_expand_weight"),
                               branch_projection_weight=old.weight(f"{prefix}.ffn_branch_projection_weight"),
                               output_projection_weight=old.weight(f"{prefix}.ffn_output_projection_weight"))
                compare(f"branched {index}", baseline.branched_feed_forward(x, **options),
                        model.branched_feed_forward(x, **options))
        x = torch.randn(1, 16, 16, 512, device="cuda", dtype=torch.float16) * 0.1
        compare("split 23", old._split_window(x, 23), new._split_window(x, 23))
        x = torch.randn(1, 8, 8, 1024, device="cuda", dtype=torch.float16) * 0.1
        compare("global 31", old._global(x, 31), new._global(x, 31))
        if args.full:
            for name in ("_window", "_split_window", "_global"):
                original = getattr(old, name)
                optimized = getattr(new, name)
                def wrapped(*a, _old=original, _new=optimized, _name=name, **kw):
                    expected = _old(*a, **kw)
                    actual = _new(*a, **kw)
                    if not torch.equal(expected, actual):
                        compare(f"teacher-forced {_name} {a[1]} {tuple(a[0].shape)}", expected, actual)
                    return expected
                setattr(old, name, wrapped)
            image = np.random.default_rng(1234).random((320, 320, 3), dtype=np.float32)
            inputs = torch.from_numpy(features.make_features(image)).cuda().half()[None]
            expected = old(inputs)
            compare("full identical features", expected, new(inputs))


if __name__ == "__main__":
    main()
