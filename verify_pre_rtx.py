#!/usr/bin/env python3
"""verify_pre_rtx.py — check that this node can actually run on the current machine.

Verifies, in order:
  1. torch is importable and reports a device
  2. the device reports its compute capability and the FP8 E4M3 rounding cast works on it
  3. float16 execution is available
  4. the weight checkpoint exists and satisfies the 649-tensor contract
  5. the node source files are present
  6. (optional) one real inference actually produces a non-identity result

Run with the same interpreter ComfyUI uses:

    python verify_pre_rtx.py --comfyui /path/to/ComfyUI
    python verify_pre_rtx.py --comfyui /path/to/ComfyUI --infer
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

PASS, FAIL, WARN = "PASS", "FAIL", "WARN"
results: list[tuple[str, str, str]] = []


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    colour = {PASS: "\033[32m", FAIL: "\033[31m", WARN: "\033[33m"}[status]
    print(f"  {colour}{status}\033[0m  {name}" + (f"  — {detail}" if detail else ""))


def section(title: str) -> None:
    print(f"\n\033[36m==> {title}\033[0m")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comfyui", required=True, type=Path,
                    help="ComfyUI root (contains custom_nodes/ and models/)")
    ap.add_argument("--node-name", default="ComfyUI-DLSS5-PyTorch",
                    help="folder name under custom_nodes/ (default: %(default)s)")
    ap.add_argument("--weights", type=Path,
                    help="explicit path to the logical safetensors (default: models/dlss5/...)")
    ap.add_argument("--infer", action="store_true",
                    help="also run one real inference and check the output differs from the input")
    args = ap.parse_args()

    root = args.comfyui.expanduser().resolve()
    node_dir = root / "custom_nodes" / args.node_name
    weights = (args.weights.expanduser().resolve() if args.weights
               else root / "models" / "dlss5" / "dlssnr-weights-logical.safetensors")

    print(f"ComfyUI   : {root}")
    print(f"node      : {node_dir}")
    print(f"weights   : {weights}")
    print(f"python    : {sys.executable}")

    # -- 1. torch / device ---------------------------------------------------
    section("1. torch and device")
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        record("import torch", FAIL, str(exc))
        return report()
    record("import torch", PASS, f"torch {torch.__version__}")

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)
        record("CUDA device", PASS, f"{name} (SM {cc[0]}.{cc[1]})")
        device = torch.device("cuda")
        if cc[0] < 8:
            record("pre-Turing GPU", WARN,
                   "SM < 8.0 — this is exactly the case the NGX path cannot serve; "
                   "the no-NGX path below is the one that matters")
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        record("CUDA device", WARN, "not available; falling back to Apple MPS")
        device = torch.device("mps")
    else:
        record("CUDA device", WARN, "not available; falling back to CPU (very slow)")
        device = torch.device("cpu")

    # -- 2. FP8 E4M3 rounding cast ------------------------------------------
    section("2. FP8 E4M3 rounding cast (used at the network's publication points)")
    if device.type == "cuda":
        try:
            x = torch.randn(4096, device=device, dtype=torch.float16)
            y = x.clamp(-448, 448).to(torch.float8_e4m3fn).to(x.dtype)
            moved = int((y != x).sum())
            record("float8_e4m3fn cast", PASS,
                   f"works on {device} (rounded {moved}/{x.numel()} values)")
        except Exception as exc:
            record("float8_e4m3fn cast", FAIL, f"{type(exc).__name__}: {exc}")
            print("      the node would need the bit-level fallback path here")
    else:
        record("float8_e4m3fn cast", WARN, f"skipped (device is {device.type})")

    # -- 3. float16 ----------------------------------------------------------
    section("3. float16 execution")
    try:
        a = torch.randn(512, 512, device=device, dtype=torch.float16)
        b = (a @ a).float().sum().item()
        record("fp16 matmul", PASS, f"ok on {device}")
    except Exception as exc:
        record("fp16 matmul", FAIL, f"{type(exc).__name__}: {exc}")

    # -- 4. checkpoint contract ---------------------------------------------
    section("4. weight checkpoint")
    if not weights.is_file():
        record("checkpoint present", FAIL,
               "not found — run install_pre_rtx.sh, or point --weights at your file")
    else:
        record("checkpoint present", PASS, f"{weights.stat().st_size / 2**20:.1f} MB")
        try:
            from safetensors import safe_open
            with safe_open(str(weights), framework="pt", device="cpu") as f:
                meta = f.metadata() or {}
                n = len(f.keys())
            if meta.get("fully_logical") == "true":
                record("fully_logical", PASS, f"format {meta.get('format')}")
            else:
                record("fully_logical", FAIL,
                       "metadata is not 'fully_logical=true' — did you point at the packed "
                       "intermediate instead of the decode() output?")
            if n == 649:
                record("tensor count", PASS, "649 tensors")
            else:
                record("tensor count", FAIL, f"expected 649, found {n}")
        except Exception as exc:
            record("read checkpoint", FAIL, f"{type(exc).__name__}: {exc}")

    # -- 5. node files -------------------------------------------------------
    section("5. node files")
    if not node_dir.is_dir():
        record("node directory", FAIL, f"missing: {node_dir}")
    else:
        record("node directory", PASS, str(node_dir))
        for rel in ("nodes.py", "__init__.py", "dlss5/pipeline.py",
                    "dlss5/model.py", "dlss5/weight_spec.json"):
            p = node_dir / rel
            record(rel, PASS if p.is_file() else FAIL, "" if p.is_file() else "missing")

    # -- 6. real inference ---------------------------------------------------
    if args.infer:
        section("6. real inference")
        if not weights.is_file() or not node_dir.is_dir():
            record("inference", FAIL, "skipped — need both the node and the checkpoint")
        else:
            sys.path.insert(0, str(node_dir))
            try:
                from dlss5.pipeline import NeuralRenderingPipeline

                pipe = NeuralRenderingPipeline.from_safetensors(
                    weights, device=device.type, precision="fast")
                # enhance_tensor takes a single [H,W,3] frame already on the model device
                src = torch.rand(64, 64, 3, device=pipe.device, dtype=torch.float32)
                with torch.no_grad():
                    out = pipe.enhance_tensor(src)
                if out.shape != src.shape:
                    record("inference", WARN, f"ran but output shape is {tuple(out.shape)}")
                else:
                    delta = float((out - src).abs().mean())
                    if delta > 1e-6:
                        record("inference", PASS, f"ran; mean |Δ| vs input = {delta:.6f}")
                    else:
                        record("inference", FAIL, "output is identical to the input")
            except Exception as exc:
                record("inference", FAIL, f"{type(exc).__name__}: {exc}")

    return report()


def report() -> int:
    print("\n" + "=" * 62)
    failed = [r for r in results if r[1] == FAIL]
    warned = [r for r in results if r[1] == WARN]
    passed = [r for r in results if r[1] == PASS]
    print(f"{len(passed)} passed, {len(warned)} warning(s), {len(failed)} failed")
    if failed:
        print("\n\033[31mFAILED:\033[0m")
        for name, _, detail in failed:
            print(f"  - {name}: {detail}")
        return 1
    print("\033[32mAll required checks passed.\033[0m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
