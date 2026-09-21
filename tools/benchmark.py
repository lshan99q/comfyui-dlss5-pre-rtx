"""Reproducible real-checkpoint timing and output comparison (CUDA required).

Use --runtime-root to benchmark a preserved checkout, --backend numpy for the
original host pipeline, and --backend tensor for the device-resident pipeline.
Model loading is excluded; input/output transfers are included in both paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--runtime-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--backend", choices=("numpy", "tensor"), default="tensor")
    parser.add_argument("--precision", choices=("fast", "reference"), default="fast")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run where the GPU driver is accessible")
    sys.path.insert(0, str(args.runtime_root))
    from dlss5.pipeline import NeuralRenderingPipeline

    pipeline = NeuralRenderingPipeline.from_safetensors(args.weights, device="cuda", precision=args.precision)
    source = np.random.default_rng(1234).random((args.height, args.width, 3), dtype=np.float32)
    source_tensor = torch.from_numpy(source)
    options = dict(processing_scale=args.scale, detail_strength=1.2, colour_strength=0.8)

    def render():
        if args.backend == "numpy":
            return pipeline.enhance(source, **options).image
        return pipeline.enhance_tensor(source_tensor, **options).cpu().numpy()

    with torch.inference_mode():
        render()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        elapsed = []
        for index in range(args.iterations):
            started = time.perf_counter()
            result = render()
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - started)
            print(f"iteration {index + 1}: {elapsed[-1]:.4f}s", flush=True)
        if args.profile:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                render()
                torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))
    report = dict(backend=args.backend, precision=args.precision, width=args.width,
                  height=args.height, scale=args.scale, gpu=torch.cuda.get_device_name(),
                  torch=torch.__version__, seconds=elapsed, median_seconds=statistics.median(elapsed),
                  peak_allocated_mb=torch.cuda.max_memory_allocated() / 2**20)
    if args.compare:
        expected = np.load(args.compare)
        diff = np.abs(result - expected)
        report.update(max_absolute_error=float(diff.max()), mean_absolute_error=float(diff.mean()),
                      rmse=float(np.sqrt(np.mean(diff * diff))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_suffix(".npy"), result)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
