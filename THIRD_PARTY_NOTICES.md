# Third-Party Notices

This distribution is a repackaging. The inference implementation is not original to this
repository; the sections below record where it comes from.

## ComfyUI-DLSS5-PyTorch

The ComfyUI node, the PyTorch pipeline and the supporting scripts under `dlss5/`, `nodes.py`,
`tools/`, `tests/` and `examples/` come from:

- Project: https://github.com/levzzz5154/ComfyUI-DLSS5-PyTorch
- Upstream notice: `Copyright (c) 2026 levzzz5154`
- License: MIT — `LICENSE` carries this notice verbatim, with a second line added for the
  pre-RTX packaging contributed by `lshan`

Nothing in this repository should be read as claiming authorship of that upstream code.

The upstream project README is preserved at `docs/UPSTREAM-README.md` for reference.

### Changes in this distribution

- Pre-RTX packaging: the compatibility analysis for Volta / SM 7.0 and its verification, in
  `docs/PRE-RTX-COMPATIBILITY.md`
- `install_pre_rtx.sh` — automated deployment and DLL-to-checkpoint conversion
- `verify_pre_rtx.py` — environment and checkpoint validation
- `docs/PERFORMANCE.md` — measured V100 timings
- A `.gitignore` that additionally excludes NVIDIA binaries and extracted weights

No changes were made to the model definition or the inference arithmetic.

## MLX-DLSS

This project contains Python source derived from the MLX-DLSS project:

- Project: https://github.com/iamwavecut/MLX-DLSS
- Reference revision: `771ccb3b3bc84477c99452b63b16cb724319a338`
- Upstream notice: `MLX-DLSS — Copyright 2026 MLX-DLSS contributors`
- License: Apache License 2.0

The derived/adapted implementation is located primarily under `dlss5/`.
Changes made here include packaging the inference code and `weight_spec.json` directly inside a ComfyUI custom node, removing the external `mlxdlss` package dependency, and adapting the public entry point and model lifecycle for ComfyUI. Complete tensor name/shape validation uses the bundled upstream specification.

The device-resident implementation in `dlss5/tensor_ops.py` adapts the recovered feature, temporal, and composition formulas to PyTorch. Model changes batch independent feed-forward operations, vectorize the recovered normalization tree, and cache attention-bias layouts while preserving publication points and arithmetic order.

A copy of the Apache License 2.0 is provided at `licenses/MLX-DLSS-APACHE-2.0.txt`.

NVIDIA DLSS, NVIDIA, and related marks are trademarks of NVIDIA Corporation. No NVIDIA proprietary binaries or model weights are redistributed by this repository.
