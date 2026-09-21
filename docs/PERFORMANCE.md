# Performance on Tesla V100

All numbers measured on a single **Tesla V100-SXM2-16GB**, `precision = fast` (float16),
`processing_scale = 1.0`, `torch 2.6.0+cu124`, driver 580.173.02.

Reproduce with the bundled benchmark:

```bash
python tools/benchmark.py \
    --weights /path/to/ComfyUI/models/dlss5/dlssnr-weights-logical.safetensors \
    --precision fast --width 1920 --height 1080 --iterations 3 \
    --output bench.json
```

Model loading is excluded from the per-frame timings; input/output transfers are included.

## Per-frame cost

| Resolution | Median | Peak VRAM |
| --- | --- | --- |
| 320 × 320 | 0.295 s | 730 MB |
| 1920 × 1080 | 0.700 s | 1732 MB |

Scaling from 0.1 MP to 2.1 MP is close to linear in pixels (0.295 s → 0.700 s, a 2.4× cost for
a 20× pixel increase is *not* linear — the fixed per-invocation overhead dominates at small
sizes, and the network's internal working resolution bounds the large end).

## Memory

Peak allocation grows roughly with output pixels: about **0.83 GB per megapixel** of output,
plus the model weights. For a 16 GB V100 this leaves comfortable headroom at 1080p and is worth
watching above ~4K.

Model load additionally initializes roughly 192 MiB of lookup constants (transcendental tables
used to make the noise deterministic across CPU/GPU rounding differences) plus a 256 KiB
temporal sigmoid table. This is a one-time cost per model handle, not per frame.

## Multi-pass cost

Three chained `DLSS5PyTorchEnhance` nodes on a 1376 × 2304 still, including model load,
completed in about 9 s wall clock through the ComfyUI HTTP API.

Per-pass drift from the original, measured on that image:

| Pass | vs original | vs previous pass |
| --- | ---: | ---: |
| 1 | 13.44 | 13.44 |
| 2 | 24.43 | 11.21 |
| 3 | 32.63 | 8.49 |

The per-pass delta decays (13.4 → 11.2 → 8.5), so chaining converges rather than diverging.
What it converges to is a **tone shift**, not added detail:

```
channel means, pass 3 vs original:  R -1.64   G -2.57   B -3.45   (warm shift)
highlights (>0.70)  -8.95/255       (highlights pulled down)
shadows    (<0.35) +12.80/255       (shadows lifted)
```

That is a filmic tone curve applied three times. Use `intensity`, `local_tone_strength` and
`local_structure_strength` for stronger single-pass output instead.

## Notes

- The first inference after a model load includes the ~192 MiB table build; steady-state timing
  is what the table above reports.
- There is no `torch.compile` and no attention kernel dispatch, so these numbers are plain
  eager-mode PyTorch. A Volta-specific tuning pass was not attempted.
- VRAM figures are `torch.cuda.max_memory_allocated()` for the render alone.
