# ComfyUI DLSS 5 for pre-RTX GPUs

Run NVIDIA's **DLSS 5 Neural Rendering** (NGX feature 18) inside ComfyUI on GPUs that every
official plugin refuses: **Tesla V100**, and other pre-Turing / non-RTX NVIDIA cards.

Verified end to end on a **Tesla V100-SXM2-16GB (Volta, SM 7.0)** — a card with no RTX
hardware, no NGX support, and no path through any driver-level DLSS 5 integration.

```
LoadImage ──► DLSS 5 PyTorch Model Loader ──► DLSS 5 PyTorch Neural Rendering ──► SaveImage
```

---

## The problem

Every DLSS 5 ComfyUI node pack you will find drives the **native NVIDIA NGX runtime**
(`nvngx_dlssnr.dll`) through D3D12. That path has two hard gates:

| Gate | Consequence |
| --- | --- |
| NGX needs Turing or newer (RTX 20+) | Volta, Pascal and Maxwell are rejected outright |
| Most packs are Windows-only (D3D12 + ReShade + RenoDX) | Linux needs Wine / vkd3d-proton workarounds |

On a V100 there is no configuration, no Wine prefix and no driver version that makes the NGX
path work: `CreateFeature(18)` returns `0xBAD00001` because the runtime has no support for the
architecture. NVIDIA's own DLSS 5 targets RTX 50 series; community NGX builds reach down to
RTX 20. Volta is below all of them.

## How this works

The DLSS 5 neural-rendering network has been reverse-engineered and reimplemented as **plain
PyTorch**. Nothing in the inference path touches NGX, D3D12, ReShade or Wine — it is ordinary
tensor code running on whatever `torch` device you point it at.

```
nvngx_dlssnr.dll                 (NVIDIA, you supply it)
      │  mlxdlss-weights extract / decode      ← offline, reads the DLL as data
      ▼
dlssnr-weights-logical.safetensors   (649 tensors)
      │  torch
      ▼
ComfyUI IMAGE / VIDEO
```

Because the runtime is portable PyTorch, the hardware gate that blocks the NGX route simply
does not exist here. The only thing that ever needed an NVIDIA GPU was **weight extraction**,
and that step reads the DLL as a byte blob — it does not execute it.

> [!IMPORTANT]
> The `nvngx_dlssnr.dll` used to obtain the weights is **not** included here, and the extracted
> weights are not redistributed either. Both are NVIDIA property. You supply your own copy;
> see [Obtaining the runtime](#obtaining-the-runtime).

## Verified hardware

| GPU | Architecture | Compute capability | Status |
| --- | --- | --- | --- |
| Tesla V100-SXM2-16GB | Volta | 7.0 | ✅ verified end to end |
| other pre-Turing NVIDIA | Pascal / Maxwell | 6.x / 5.x | ⚠️ expected to work, untested |
| RTX 20/30/40/50 | Turing+ | 7.5+ | ⚠️ should work, untested here |
| CPU | — | — | ⚠️ supported by the node, extremely slow |
| Apple Silicon | — | — | ⚠️ supported by the node (`mps`), untested |

The device dropdown offers `auto / cuda / cpu / mps`, so this is not really a "pre-RTX" trick —
it is a *no-NGX* path. Pre-RTX cards are simply the case where it is the only option.

## Compatibility notes for Volta

Three things had to be checked before this could work on SM 7.0.

**`torch.float8_e4m3fn` on CUDA.** The network rounds activations to E4M3 at its publication
points. PyTorch's float8 kernels are documented for Ada and newer, so this looked like a
blocker. Measured on `torch 2.6.0+cu124`:

```python
>>> x = torch.randn(4096, device="cuda", dtype=torch.float16)
>>> x.clamp(-448, 448).to(torch.float8_e4m3fn).to(x.dtype)   # works on V100
```

The cast succeeds on SM 7.0. Note this is only used for **rounding** — there is no FP8 matrix
multiply anywhere in the path, so tensor-core FP8 support is irrelevant.

**Precision.** `precision = fast` runs the model in `float16`, which Volta supports natively
(tensor cores included). `precision = reference` runs in `float32`.

**Operator set.** The implementation contains no `torch.compile`, no FlashAttention and no
Triton kernels — attention is written as explicit matmuls plus a recovered softmax. There is
nothing that requires SM 8.x.

Full detail in [`docs/PRE-RTX-COMPATIBILITY.md`](docs/PRE-RTX-COMPATIBILITY.md).

## Performance

Measured on Tesla V100-SXM2-16GB, `precision = fast`, `scale = 1.0`:

| Resolution | Per frame | Peak VRAM |
| --- | --- | --- |
| 320 × 320 | 0.295 s | 730 MB |
| 1920 × 1080 | 0.700 s | 1732 MB |

A 1376 × 2304 still image through three chained passes completed in about 9 s including model
load. Model load includes a one-time ~192 MiB lookup-table initialization.

## Install

### Automated

```bash
git clone https://github.com/<you>/comfyui-dlss5-pre-rtx.git
cd comfyui-dlss5-pre-rtx
./install_pre_rtx.sh --comfyui /path/to/ComfyUI --dll /path/to/nvngx_dlssnr.dll
```

The script copies the node into `custom_nodes/`, installs the extraction tool, converts the
DLL into a logical checkpoint and drops it in `models/dlss5/`. Add `--dry-run` to see the plan
without touching anything.

### Manual

```bash
# 1. the node
cd ComfyUI/custom_nodes
git clone https://github.com/<you>/comfyui-dlss5-pre-rtx.git

# 2. the extractor (deps numpy/torch/safetensors/pillow are already in a ComfyUI venv)
git clone https://github.com/iamwavecut/MLX-DLSS.git
cd MLX-DLSS && git checkout 7debaaf28c8f3b789e0d95cc06abd9796da00170
python -m pip install --no-deps ./python

# 3. extract + decode the weights
mlxdlss-weights extract nvngx_dlssnr.dll weights/dlssnr-packed.safetensors
mlxdlss-weights decode  weights/dlssnr-packed.safetensors weights/dlssnr-logical.safetensors
#    expect: 153 source tensors -> 649 decoded, 0 unsupported, 0 opaque

# 4. deploy and restart ComfyUI
cp weights/dlssnr-logical.safetensors ComfyUI/models/dlss5/
```

Nodes appear under **`DLSS 5/PyTorch (experimental)`**:

- `DLSS5PyTorchModelLoader` — pick the `.safetensors`
- `DLSS5PyTorchEnhance` — stills / image batches
- `DLSS5PyTorchVideoEnhance` — temporal video batches
- `DLSS5PyTorchClearCache`

### Verify

```bash
python verify_pre_rtx.py --comfyui /path/to/ComfyUI
```

Checks torch/device/arch, the FP8 cast, the weight checkpoint contract and the node files, and
prints a pass/fail per item.

## Obtaining the runtime

`nvngx_dlssnr.dll` is proprietary NVIDIA software. It is not shipped here and neither are the
decoded weights.

**It is not in any public NVIDIA SDK.** Both first-party sources were checked after DLSS 5's
2026-09-03 launch and neither carries it — Streamline SDK v2.14.1 ships `nvngx_dlss.dll`,
`nvngx_dlssd.dll`, `nvngx_dlssg.dll` and `nvngx_deepdvc.dll` but no neural-rendering DLL, and the
DLSS SDK v310.9.1 demo zip contains only `nvngx_dlss.dll`.

The file ships inside games that carry DLSS 5. The known source is **NBA 2K27** (first spotted in
its early-access build on 2026-08-27: `nvngx_dlssnr.dll`, version `310.8.0.0`, 165,840,496 bytes).

The build the decoder accepts reports file version **310.8.0.0**. The extractor names one
specific SHA-256 as canonical, but **other 310.8.0.0 builds decode to the identical logical
structure** — 153 source tensors expanding to 649, with zero unsupported and zero opaque
entries. `verify_pre_rtx.py` therefore checks the structure, not just the hash, because the
hash check alone rejects functionally identical builds.

Full detail — how to locate it in a game install, how to verify a copy, what will not work, and
the licensing position — is in
**[`docs/OBTAINING-THE-RUNTIME.md`](docs/OBTAINING-THE-RUNTIME.md)**.

Use a copy you are legally entitled to use. This repository does not link to mirrors.

## Known limitations

- **This is a reimplementation, not NVIDIA's binary.** Quality is DLSS-5-*like*, not
  DLSS-5-*equal*; parts of the weight layout are reconstructed.
- **Not designed to be chained.** Three passes compound into a dynamic-range compression and a
  warm colour shift (highlights −8.95/255, shadows +12.80/255, blue −3.45/255) with diminishing
  returns. Tune `intensity` / `local_structure_strength` instead of stacking passes.
- Weights are validated by name, shape and dtype, but numerical equivalence against NVIDIA's
  original output is not established.
- Model load initializes roughly 192 MiB of lookup constants — a one-time cost per model handle.
- The recovered temporal path expects motion vectors; without them the video node reprojects
  with zero motion, which is wrong for real camera movement.

## Credits and license

This repository packages a pre-RTX-ready distribution of someone else's work. The inference
implementation is **not** original to this repository:

- **[levzzz5154/ComfyUI-DLSS5-PyTorch](https://github.com/levzzz5154/ComfyUI-DLSS5-PyTorch)** —
  the ComfyUI node and PyTorch pipeline, MIT, Copyright (c) 2026 levzzz5154. `LICENSE` is kept
  verbatim.
- **[iamwavecut/MLX-DLSS](https://github.com/iamwavecut/MLX-DLSS)** — the weight extractor and
  the recovered network definition that `dlss5/` derives from, Apache-2.0. See
  `THIRD_PARTY_NOTICES.md` and `licenses/`.
- **[jlrouzies-fr/DLSS5-Feeder](https://github.com/jlrouzies-fr/DLSS5-Feeder)** — documented the
  runtime sourcing used to locate a `310.8.0.0` build.

What this repository adds is the pre-RTX packaging: the compatibility analysis and its
verification, the install and verification scripts, and the measured performance on Volta.

NVIDIA, DLSS and NGX are trademarks of NVIDIA Corporation. No NVIDIA proprietary binaries or
model weights are redistributed by this repository.
