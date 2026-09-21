# Running DLSS 5 Neural Rendering on pre-RTX hardware

Notes from bringing this node up on a **Tesla V100-SXM2-16GB (Volta, SM 7.0)** — a card that
sits below every supported DLSS 5 configuration.

## Why the NGX path cannot work here

DLSS 5 Neural Rendering is NGX **feature 18**. The feature is created through
`NVSDK_NGX_D3D12_CreateFeature`, and NGX itself only exists from Turing (RTX 20) onward.
Every DLSS 5 ComfyUI node pack in circulation is a client of that API:

| Project | Runtime | Pre-RTX? |
| --- | --- | --- |
| `Blueforcer/ComfyUI-DLSS5-Enhancer` | native D3D12 worker + ReShade + RenoDX | no, Windows only |
| `orex2121/ComfyUI-DLSS5-orex` | in-process D3D12/NGX bridge | no, Windows only |
| `RH-RunningHub/ComfyUI-RH-DLSS5` | D3D12 bridge, or Wine + vkd3d-proton on Linux | no, needs RTX |
| `LQCCS/ComfyUI-DLSS5NR-Wine` | Wine + MinGW-crossbuilt host | no, needs RTX |
| `bmitch87/DLSS5VKLayer` | Vulkan layer forwarding to a Wine NGX helper | no, "NVIDIA GPU and NVIDIA driver" |

On Volta the failure is upstream of any of them: the runtime reports the feature as
unsupported (`0xBAD00001`). There is no Wine prefix, driver version or vkd3d-proton release
that changes this, because the architecture check happens inside NVIDIA's own code.

The Linux Vulkan-layer approach is worth calling out because it is often described as
"DLSS 5 on Linux with any GPU". It is not a workaround for the hardware gate — it still
requires an NVIDIA GPU and the official driver, and it still ships frames into `nvngx_dlssnr.dll`.

## The way through: don't call NGX at all

The network has been reverse-engineered and reimplemented in portable PyTorch
(`levzzz5154/ComfyUI-DLSS5-PyTorch`, derived from `iamwavecut/MLX-DLSS`). Inference is ordinary
tensor code. The hardware question therefore reduces to: *does the operator and dtype set used
by the reimplementation run on this architecture?*

### 1. The FP8 cast — the one real risk

`dlss5/model.py` rounds activations to E4M3 at the network's publication points:

```python
_FLOAT8_CAST_DEVICES = frozenset({"cpu", "cuda"})

def e4m3_round_trip(value):
    if value.device.type in _FLOAT8_CAST_DEVICES and value.dtype in (torch.float16, torch.float32):
        return value.clamp(-448, 448).to(torch.float8_e4m3fn).to(value.dtype)
    ...
```

`torch.float8_e4m3fn` on CUDA is normally associated with SM 8.9+ (Ada) and SM 9.0 (Hopper);
NVIDIA exposes native FP8 conversion instructions only there. On SM 7.0 this looked like a
hard blocker. Measured, it is not:

```
torch: 2.6.0+cu124 | device: Tesla V100-SXM2-16GB | CC: (7, 0)

--- native float8_e4m3fn cast on CUDA (V100) ---
OK native cuda float8 cast -> torch.Size([4096]) torch.float16
```

PyTorch services the conversion on unsupported architectures. Two details make this safe
rather than lucky:

- the operation is a **rounding** step, not a GEMM — there is no FP8 tensor-core path involved,
  so the absence of FP8 tensor cores is irrelevant;
- the module already carries an exact bit-level equivalent (`e4m3_round_trip_bitwise`) used for
  MPS and tracing, so a device that *did* reject the cast has a correct fallback.

### 2. Precision

`precision = "fast"` moves the model to `float16` on any non-CPU device. Volta has native FP16
and FP16 tensor cores, so this is well supported. `precision = "reference"` runs `float32`.

### 3. Operator set

Grepped the whole implementation for architecture-sensitive constructs:

| Construct | Present? |
| --- | --- |
| `torch.compile` | no |
| FlashAttention / `scaled_dot_product_attention` | no |
| Triton kernels | no |
| FP8 matmul | no |
| manual matmul + softmax attention | yes |

Attention is written out explicitly (`query @ key.transpose(-2,-1)`, a recovered approximate
softmax, then `probabilities @ value`), so nothing depends on SM 8.x kernels.

## End-to-end verification on V100

Weights were produced from a `310.8.0.0` DLL:

```
extract: 153 source tensors, format dlssnr-WEIGHTS_HT-packed-u8-v2
decode : 153 -> 649 tensors, unsupportedSourceTensorCount: 0, opaqueOutputTensorCount: 0
         format: dlssnr-logical-v18, fully_logical=true
validate: 649 tensors checked against the bundled weight_spec.json
```

Real inference, `precision = fast`:

```
===========================================
320x320     median 0.2946 s/frame   peak VRAM  730 MB
1920x1080   median 0.7003 s/frame   peak VRAM 1732 MB
===========================================
gpu: Tesla V100-SXM2-16GB
torch: 2.6.0+cu124
```

End-to-end through the ComfyUI API (`LoadImage → Loader → Enhance → SaveImage`), 1920×1080:

```
mean |Δ| vs input = 9.536/255      (not a passthrough)
detail energy     0.00572 → 0.00566
luma              78.5 → 77.9
```

Output was visually correct with no channel swap.

## Caveats

- **Only V100 is verified.** Pascal (SM 6.x) and Maxwell (SM 5.x) are plausible — the code path
  has no SM-version check — but they were not tested, and FP16 throughput there is poor.
- **The reimplementation is approximate.** Parts of the weight layout are reconstructed from
  consumer evidence; the upstream authors describe the result as "DLSS-5-like, not
  DLSS-5-equal". Running it does not make a V100 produce NVIDIA's reference output.
- **Do not chain passes.** Measured over three passes: highlights −8.95/255, shadows
  +12.80/255, blue channel −3.45/255, with per-pass deltas decaying 13.4 → 11.2 → 8.5. It
  converges, but what accumulates is a tone/colour shift, not detail.
