# ComfyUI-DLSS5-PyTorch

A **self-contained ComfyUI implementation with a GPU-resident PyTorch rendering pipeline** of the recovered DLSS 5 Neural Rendering network.

The reverse-engineering work this implementation is based on comes from [iamwavecut/MLX-DLSS](https://github.com/iamwavecut/MLX-DLSS). The required model/runtime code is included directly in this repository as ordinary Python source under `dlss5/`.

![DLSS 5 PyTorch workflow in ComfyUI showing the model loader, input image, rendering controls, and output preview](examples/workflow-preview.png)

[Download the sample workflow and input image](#sample-image-workflow).

## Installation

Installation has two parts: install the custom nodes, then extract, decode, and install the DLSS 5 safetensors checkpoint. Complete both before running a workflow.

### Install the custom nodes

Clone this repository into your ComfyUI installation and install the dependencies with the Python interpreter used by ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/levzzz5154/ComfyUI-DLSS5-PyTorch.git
cd ComfyUI-DLSS5-PyTorch
python -m pip install -r requirements.txt
```

Next, complete the [weight extraction and decoding steps below](#weights-extract-and-decode-with-mlx-dlss). After installing the checkpoint, restart ComfyUI and open the [sample workflow](#sample-image-workflow).

The requirements are only normal Python libraries used by the implementation:

```text
numpy
Pillow
safetensors
```

PyTorch itself is supplied by ComfyUI.

### Weights: extract and decode with MLX-DLSS

Weights are not included or downloaded by this extension. Use [MLX-DLSS](https://github.com/iamwavecut/MLX-DLSS) once to extract a DLL you supply and decode its packed tensors. ComfyUI then loads the resulting **fully-logical `.safetensors`** directly; the extraction tools and DLL are not needed for rendering.

#### 1. Obtain the neural-rendering DLL

You need **`nvngx_dlssnr.dll`**, not the Super Resolution DLL (`nvngx_dlss.dll`) or frame-generation library (`nvngx_dlssg.dll` / `libnvidia-ngx-dlssg.so`). Upstream identifies the Streamline SDK distribution's `bin/x64/nvngx_dlssnr.dll` and games shipping DLSS 5 as sources; see [upstream weight requirements](https://github.com/iamwavecut/MLX-DLSS#weights) and [NVIDIA Streamline releases](https://github.com/NVIDIA-RTX/Streamline/releases). Supply your own copy; no proprietary DLL or extracted weights are redistributed here.

The extractor's [known-build table](https://github.com/iamwavecut/MLX-DLSS/blob/7debaaf28c8f3b789e0d95cc06abd9796da00170/python/mlxdlss/tools/cli.py) identifies:

- File version: `310.8.0.0`
- SHA-256: `ceb6432f6fbdf44d886014bcd47241932bf8b67439feef9bbdd0961436662650`

Check the hash with the command below. A matching version number alone is insufficient. The hash command prints `unknown checkpoint` for an unrecognized file but still exits successfully; read its output. An unknown build may decode, but successful decoding and shape validation do not establish numerical compatibility with the supported build.

#### 2. Install the extraction tools separately

Use Git and Python 3.10 or newer. An isolated environment keeps the extractor's dependencies separate from ComfyUI. The following pins the inspected extractor revision, which emits `dlssnr-logical-v18`, a format this extension accepts. Clone MLX-DLSS outside `ComfyUI/custom_nodes`:

```bash
git clone https://github.com/iamwavecut/MLX-DLSS.git
cd MLX-DLSS
git checkout 7debaaf28c8f3b789e0d95cc06abd9796da00170
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install ./python
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install ./python
```

The base Python package is sufficient. Extraction reads the DLL as data: it does not execute it, require an NVIDIA GPU, or require building the Swift/Metal runtime. Core ML, web, and video extras are unnecessary. The package's base dependencies include PyTorch, NumPy, Pillow, and safetensors; see [upstream package configuration](https://github.com/iamwavecut/MLX-DLSS/blob/7debaaf28c8f3b789e0d95cc06abd9796da00170/python/pyproject.toml).

#### 3. Extract the packed weights, then decode them

Run from the MLX-DLSS checkout. Replace the quoted DLL path with your own. On Linux/macOS, with the environment above active:

```bash
mlxdlss-weights sha256 "/path/to/nvngx_dlssnr.dll"
mkdir -p weights
mlxdlss-weights extract "/path/to/nvngx_dlssnr.dll" weights/dlssnr-weights-packed.safetensors
mlxdlss-weights decode weights/dlssnr-weights-packed.safetensors weights/dlssnr-weights-logical.safetensors
```

Windows PowerShell, without needing to activate the environment:

```powershell
.\.venv\Scripts\mlxdlss-weights.exe sha256 "C:\path\to\nvngx_dlssnr.dll"
New-Item -ItemType Directory -Force weights
.\.venv\Scripts\mlxdlss-weights.exe extract "C:\path\to\nvngx_dlssnr.dll" weights/dlssnr-weights-packed.safetensors
.\.venv\Scripts\mlxdlss-weights.exe decode weights/dlssnr-weights-packed.safetensors weights/dlssnr-weights-logical.safetensors
```

`extract` produces an **opaque packed intermediate**, which this ComfyUI loader cannot use. `decode` produces the logical floating-point tensors. For the supported layout, the decoder reports 153 decoded source tensors, 649 output tensors, `unsupportedSourceTensorCount: 0`, and `opaqueOutputTensorCount: 0`. The output metadata must contain `fully_logical=true` and `format=dlssnr-logical-v18`. See the [decoder source](https://github.com/iamwavecut/MLX-DLSS/blob/7debaaf28c8f3b789e0d95cc06abd9796da00170/python/mlxdlss/tools/unpack_dlssnr_weights.py).

Alternatively, upstream's combined command runs extraction, decoding, and MLX packaging:

```bash
mlxdlss-weights all "/path/to/nvngx_dlssnr.dll" weights
```

On Windows, use `.\.venv\Scripts\mlxdlss-weights.exe` as above. The combined command writes `weights/dlssnr-weights-logical.safetensors` plus packed and `.dlssmodel` artifacts. ComfyUI needs only the logical safetensors file; omit `--coreml` and ignore the MLX package. The separate `extract` / `decode` commands avoid generating that extra package.

#### 4. Install and validate the logical checkpoint

Copy **`weights/dlssnr-weights-logical.safetensors`** into your ComfyUI installation:

```text
ComfyUI/
└── models/
    └── dlss5/
        └── dlssnr-weights-logical.safetensors
```

Create `models/dlss5` if needed. For Windows portable ComfyUI, this is under `ComfyUI_windows_portable/ComfyUI/models/dlss5/`. The filename is arbitrary; changing a packed file's name does not decode it.

Restart ComfyUI, add **DLSS 5 PyTorch Model Loader**, and select the file. Connect its `dlss5_model` output to the still or video rendering node. The loader accepts formats `dlssnr-logical-v8` through `dlssnr-logical-v18`, requires `fully_logical=true`, and checks all 649 required tensor names, shapes, and floating-point dtypes. These format versions are decoder revisions, not NVIDIA DLL versions.

For a standalone validation before launching ComfyUI, run this from **this repository's root** using ComfyUI's Python interpreter (replace `python` with its full path if necessary):

```bash
python -c "from dlss5.pipeline import load_weights; w = load_weights('../../models/dlss5/dlssnr-weights-logical.safetensors'); print('Validated', len(w), 'tensors')"
```

This loads the weights on CPU and validates the same contract as the node; it does not run inference.

#### Extraction and loading troubleshooting

| Symptom | What to check |
| --- | --- |
| `mlxdlss-weights` is not found | Activate the extraction environment, or invoke `.venv/bin/mlxdlss-weights` on Linux/macOS or `.\.venv\Scripts\mlxdlss-weights.exe` on Windows. |
| `unknown checkpoint` from `sha256` | Compare the full hash above and verify that the input is the neural-rendering DLL. The same version string can accompany a different file. |
| Missing `WEIGHTS_HT`, unknown tensor families, or nonzero unsupported/opaque counts | Check the DLL build and extractor revision. Do not edit metadata to force an unsupported file through the loader. `mlxdlss-weights inspect weights/dlssnr-weights-packed.safetensors` lists the extracted contents for diagnosis. |
| Unsupported format or missing `fully_logical=true` | Select the output of `decode`, not the packed intermediate, a frame-generation checkpoint, `.dlssmodel`, or `.mlpackage`. Use the pinned extractor if a newer revision emits a format this extension does not accept. |
| Missing tensors or shape mismatch | Re-extract and decode with the documented DLL and tool revision; the checkpoint must match the bundled specification. |
| No models in the loader dropdown | Check the actual ComfyUI installation's `models/dlss5/` directory, the `.safetensors` extension, and restart ComfyUI after copying the file. |

## Sample image workflow

Download the [sample workflow](examples/dlss5-image-sample.json) and its [input screenshot](examples/Screenshot_20260830_105338.png).

1. Open or drag the workflow JSON into ComfyUI.
2. Upload the screenshot through **Load Image**, or copy it into `ComfyUI/input/` under its original filename, `Screenshot_20260830_105338.png`.
3. In **DLSS 5 PyTorch Model Loader**, select your extracted logical checkpoint. The saved workflow uses `dlssnr-decoded.safetensors`; select `dlssnr-weights-logical.safetensors` instead if you followed the extraction guide above.
4. Run the workflow. It uses the standard profile, processing scale 1, full intensity, `fast` precision, and automatic device selection, then displays the result in **Preview Image**.

The screenshot is the input, not an example of the processed output. Model weights must be supplied separately.

## Nodes

### DLSS 5 PyTorch Model Loader

Loads the logical safetensors directly into the local PyTorch implementation.

- `fast`: float16 model execution on GPU while preserving recovered E4M3 publication points.
- `reference`: float32/reference execution.
- `auto`: follows ComfyUI's selected device, including its CPU mode.

The loader validates all required tensor names and shapes against the bundled reference specification. Weights stay on CPU between renders. Before rendering, the node requests memory through ComfyUI, moves the model to the selected device, and offloads it to CPU on completion or failure. Cancellation is checked between frames and transformer blocks. Output placement and dtype follow ComfyUI's intermediate tensor settings.

During a render, each input frame is transferred to the selected device once. Features, model heads, control/motion/depth processing, and video history remain there. Only the finished image is copied to ComfyUI's intermediate device. Enabling scene-cut detection adds one scalar synchronization per transition for the reset decision.

To preserve the original deterministic noise despite CPU/GPU transcendental rounding differences, the first tensor render initializes approximately 192 MiB of lookup constants (plus a 256 KiB temporal sigmoid table). They contain no model weights or image data. All per-frame lookups and noise arithmetic run on the rendering device; the tables offload with the model. Initialization is a one-time cost per model handle.

### DLSS 5 PyTorch Neural Rendering

The still/first-frame node. It intentionally contains **all recovered non-temporal controls** and none of the motion/history-specific controls.

Inputs include:

- current `IMAGE`
- profile: standard / natural / cinematic / neutral
- processing scale
- intensity (0–1 final effect blend; 1 applies the full effect)
- detail strength / colour strength / detail radius
- deterministic frame index
- custom style index
- local tone strength
- local structure strength
- automatic-mask enable
- skin structure strength
- automatic-mask structure strength
- optional RGB `control_image`
- image batches

The RGB control image follows the recovered contract:

- R: final effect blend
- G: local tone multiplier
- B: local structure multiplier

Intensity and the red control channel blend the finished effect against the original image after resizing and detail filtering. Zero intensity or zero red preserves the source pixel. Video history retains the full neural result before this display blend.

An explicit control image takes precedence over automatic masking. `sequence (advance frame index)` only changes deterministic noise across a batch; it does **not** turn the still node into a temporal sequence.

### DLSS 5 PyTorch Video Neural Rendering

The temporal node. It takes an ordered ComfyUI `IMAGE` batch and carries the previous neural-rendered display history internally from one frame to the next.

It has the same art-direction/effect controls as the still node, plus:

- `motion_vectors` — RG are X/Y current-to-previous motion
- `motion_format` — pixel or normalized UV
- `motion_encoding` — raw signed RG, or 0.5-centered RG
- `motion_value_scale` — decoding scale for the supplied motion values
- `motion_scale_x/y` — recovered host motion scale/sign controls
- `jitter_delta_x/y` — previous jitter minus current jitter, in pixels
- `blend_scale` — recovered temporal blend cap (default `0.73974609375`)
- optional `depth_image`
- `depth_guide`
- `depth_inverted`
- optional scene-cut history reset threshold
- optional RGB control image

The first frame is evaluated as the normal first-frame path. Every later frame reprojects the retained previous output using the supplied current-to-previous motion and writes the reconstructed history into network feature channels 7–9.

`motion_vectors` may have:

- one frame, broadcast to every transition;
- the same batch size as the video (frame 0 is ignored); or
- `video_batch_size - 1`, one field per actual transition.

For `motion_encoding = signed RG`, values are used directly. For `0.5-centered RG`, `0.5` means zero and `[0,1]` maps to `[-1,1]` before `motion_value_scale` is applied.

For pixel motion, the recovered conversion is:

```text
normalizedX = (motionX * motionScaleX + jitterDeltaX) / width
normalizedY = (motionY * motionScaleY + jitterDeltaY) / height
```

Positive normalized offsets sample history to the right/down (`historyUV = currentUV + motion`).

#### Depth behavior

The surfaced DLL reads depth-related parameters, but the observed shipping path binds a null depth texture, so its normal behavior samples motion at the current pixel. Therefore:

- `observed (matches DLL)` ignores depth and matches that surfaced path;
- `closest-depth (experimental)` enables the separately recovered dormant branch that chooses the closest of the current pixel and four diagonals before sampling motion.

Depth is not a direct transformer feature channel.

Temporal mode currently requires `processing_scale = 1.0`.

### DLSS 5 PyTorch Clear Cache

Clears this extension's model lookup cache, offloads live model handles to CPU, and asks ComfyUI to release unused device allocations. It runs every time it is queued with `clear=true`. ComfyUI may retain CPU weights in cached loader outputs; this node does not invalidate the entire workflow cache. It has no dependency link to the render nodes, so use a separate queue execution when ordering matters.

## What “self-contained” means here

This repository does **not** depend on the `mlxdlss` Python package and does not download MLX-DLSS at install or runtime.

Self-contained refers to the bundled implementation, not to having zero library dependencies. The ComfyUI rendering path uses PyTorch for the transformer, preprocessing, motion reprojection, composition, and resizing. NumPy initializes immutable reference lookup tables once and supports the retained reference API; Pillow is used only by that reference API. Checkpoint loading uses safetensors. The nodes require ComfyUI, and model weights must be supplied separately.

It also does **not** use or bundle:

- `nvngx_dlssnr.dll`
- NVIDIA NGX
- D3D12 runtime bridges
- VapourSynth/VapourKit
- custom `.so` / `.dll` native extensions
- precompiled CUDA binaries
- opaque executable blobs

The inference implementation is visible source:

```text
ComfyUI-DLSS5-PyTorch/
├── __init__.py
├── nodes.py
└── dlss5/
    ├── __init__.py
    ├── model.py
    ├── pipeline.py
    ├── features.py
    ├── temporal.py
    ├── composition.py
    ├── tensor_ops.py
    └── weight_spec.json
```

The **model weight data** is not bundled. You need the libraries listed below and a compatible fully-logical DLSS 5 `.safetensors` file; proprietary model weights are not redistributed here.

## Architecture/runtime

`dlss5/model.py` contains the recovered 71-block transformer graph in PyTorch, including E4M3 publication emulation, the custom polynomial gate, cosine attention, shifted 8×8 windows, global bottleneck attention, hierarchical pooling/upsampling, and the four-channel output head.

`dlss5/tensor_ops.py` implements device-resident deterministic noise, 16-channel feature construction, five-tap history reconstruction, closest-depth motion guidance, learned temporal composition, separable detail filtering, and Lanczos resizing. `dlss5/pipeline.py` exposes `enhance_tensor` and `run_features_tensor`; the ComfyUI nodes use these tensor APIs.

The NumPy implementations in `features.py`, `temporal.py`, and `composition.py`, plus the NumPy pipeline API, remain available for reference comparisons. They are not used for per-frame image processing by the nodes.

Transformer optimizations batch the independent feed-forward heads/branches into GEMMs, vectorize cosine normalization while preserving its half-rounding reduction tree, use native PyTorch float8 conversion for E4M3 publication on CUDA, and cache recovered attention-bias layouts. The recovered bit-affine exponential is preserved. The 64-token window path retains E4M3 probabilities; longer global-attention rows use float32 totals and float16 probabilities to avoid overflow and E4M3 underflow. This numerical safeguard is not validated against NVIDIA captures at those extents.

There is no hidden runtime behind the ComfyUI nodes.

## What is *not* a runtime input

Albedo, normals, roughness, metallic/specular buffers and similar renderer G-buffer attributes are **not inputs to this recovered checkpoint's deployed transformer**. They are associated with NVIDIA's 3D-guided training/supervision story, not extra sockets that should be invented in this ComfyUI implementation.

The actual recovered transformer feature layout is:

```text
0-2   deterministic Gaussian noise
3     constant 1
4-6   current RGB
7-9   reprojected history RGB (current RGB on first frame)
10    style
11    local tone
12    local structure
13    skin structure
14    automatic-mask structure
15    zero
```

Motion and the optional depth guide are preprocessing inputs used to construct channels 7–9; they are not concatenated directly into the transformer.

## Current limitations

The CUDA rendering path is GPU-resident and optimized, but it is not NVIDIA's fused production runtime. It uses ordinary PyTorch operations, without a custom CUDA extension or a mandatory compilation step. It does not promise real-time full-HD rendering. CPU remains supported; MPS uses float32 resampling/coordinate arithmetic because Metal does not support float64, so exact CUDA/reference parity is not claimed there.

The video node currently expects motion fields to be supplied by the workflow; it does not generate optical flow itself.

## Measured performance

RTX 4070 SUPER (12 GiB), PyTorch 2.10.0+cu130, `fast` precision, processing scale 1, deterministic synthetic RGB input, detail strength 1.2 and colour strength 0.8. Medians of three warmed runs include frame input/output transfers. Model loading, first-use lookup initialization, and ComfyUI's between-node weight transfers are excluded.

| Image size | Previous pipeline | Tensor pipeline | Speedup | Peak allocated VRAM |
| --- | ---: | ---: | ---: | ---: |
| 320 × 320 | 520 ms | 74 ms | 7.0× | 730 MiB |
| 640 × 360 | 603 ms | 106 ms | 5.7× | 1,071 MiB |
| 1920 × 1080 | 2,253 ms | 884 ms | 2.5× | 1,733 MiB |

Maximum output difference from the prior pipeline was `1.2e-7` for these runs. A three-frame 320 × 320 video with fractional pixel motion matched exactly through the actual ComfyUI nodes. These measurements describe the tested checkpoint and inputs, not a guarantee for every device or workflow.

Reproduce a warmed tensor benchmark with your own checkpoint:

```bash
python tools/benchmark.py --weights /path/to/logical.safetensors --backend tensor --width 1920 --height 1080 --output /tmp/dlss5-benchmark
```

The benchmark accepts `--runtime-root` to compare another checkout, `--backend numpy` for its original API, and `--compare /path/to/output.npy` to report numerical differences. It records JSON metrics and the output array.

## Why not use the native DLL?

There are already ComfyUI projects wrapping the native NVIDIA runtime. This project has a different goal: make the recovered model graph directly inspectable and modifiable in PyTorch so it can serve as a baseline for CUDA/FP8/INT8 work.

## Development check

```bash
python tests/smoke.py
python tests/regression.py
python tests/tensor.py
python tests/tensor.py --device cuda
```

The smoke test checks both still and temporal node plumbing and explicitly verifies that the repository has no `mlxdlss` package dependency/import.
The regression tests cover checkpoint validation, offloading after failures, cache invalidation, intensity limits, and cancellation. They require no model weights or GPU.
The tensor suite compares noise, masks, padding, resampling, detail composition, depth guidance, temporal history, and motion normalization against the NumPy/Pillow reference. CUDA tests require GPU access. `tools/validate_nodes.py` exercises the installed ComfyUI nodes and checks a short video against a preserved reference checkout; it requires a real checkpoint.

## Credits

The architecture recovery, tensor layouts, feature reconstruction, temporal reconstruction, and reference implementation this project is derived from were produced by the **MLX-DLSS contributors**:

https://github.com/iamwavecut/MLX-DLSS

Vendored/adapted portions retain the upstream Apache-2.0 licensing requirements. See `THIRD_PARTY_NOTICES.md` and `licenses/MLX-DLSS-APACHE-2.0.txt`.

This project is not affiliated with or endorsed by NVIDIA, Comfy Org, or the MLX-DLSS contributors.

## License

The original ComfyUI wrapper code in this repository is MIT licensed. Code derived from MLX-DLSS remains subject to Apache License 2.0; see the included third-party notices and license copy.
