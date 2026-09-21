#!/usr/bin/env bash
# install_pre_rtx.sh — deploy ComfyUI DLSS 5 (pre-RTX / no-NGX path) and build the weights.
#
# Converts a user-supplied, legally obtained nvngx_dlssnr.dll (310.8.0.0) into the logical
# safetensors checkpoint the node loads. No NVIDIA binary is downloaded or redistributed here.
#
# Usage:
#   ./install_pre_rtx.sh --comfyui /path/to/ComfyUI --dll /path/to/nvngx_dlssnr.dll
#   ./install_pre_rtx.sh --comfyui /path/to/ComfyUI --dll dlssnr.dll --dry-run
set -euo pipefail

COMFYUI_DIR=""
DLL=""
NODE_NAME="ComfyUI-DLSS5-PyTorch"
MLX_REV="7debaaf28c8f3b789e0d95cc06abd9796da00170"
WORK_DIR="${TMPDIR:-/tmp}/dlss5-pre-rtx-build"
DRY_RUN=0
SKIP_NODE=0
MODEL_DIR_NAME="dlss5"
WEIGHTS_NAME="dlssnr-weights-logical.safetensors"

die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }
info() { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarn:\033[0m %s\n' "$*" >&2; }
ok()   { printf '\033[32m  ok\033[0m %s\n' "$*"; }

run() {
  if [ "$DRY_RUN" -eq 1 ]; then printf '  [dry-run] %s\n' "$*"; else "$@"; fi
}

usage() {
  cat <<'EOF'
install_pre_rtx.sh — deploy ComfyUI DLSS 5 (pre-RTX / no-NGX path) and build the weights.

Converts a user-supplied, legally obtained nvngx_dlssnr.dll (310.8.0.0) into the logical
safetensors checkpoint the node loads. No NVIDIA binary is downloaded or redistributed here.

Usage:
  ./install_pre_rtx.sh --comfyui /path/to/ComfyUI --dll /path/to/nvngx_dlssnr.dll
  ./install_pre_rtx.sh --comfyui /path/to/ComfyUI --dll dlssnr.dll --dry-run

Options:
  --comfyui PATH   ComfyUI root (the folder containing custom_nodes/ and models/)
  --dll PATH       your nvngx_dlssnr.dll, version 310.8.0.0
  --node-name N    folder name under custom_nodes/ (default: ComfyUI-DLSS5-PyTorch)
  --work-dir PATH  scratch directory for the extractor (default: $TMPDIR/dlss5-pre-rtx-build)
  --dry-run        print what would happen, change nothing
  --skip-node      do not copy the node into custom_nodes/ (weights only)
  --keep-temp      keep the scratch directory
  -h, --help       this text
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --comfyui)   COMFYUI_DIR="${2:-}"; shift 2 ;;
    --dll)       DLL="${2:-}"; shift 2 ;;
    --node-name) NODE_NAME="${2:-}"; shift 2 ;;
    --work-dir)  WORK_DIR="${2:-}"; shift 2 ;;
    --dry-run)   DRY_RUN=1; shift ;;
    --skip-node) SKIP_NODE=1; shift ;;
    --keep-temp) KEEP_TEMP=1; shift ;;
    -h|--help)   usage; exit 0 ;;
    *)           die "unknown option: $1 (try --help)" ;;
  esac
done

[ -n "$COMFYUI_DIR" ] || die "--comfyui is required"
[ -d "$COMFYUI_DIR" ] || die "no such directory: $COMFYUI_DIR"
COMFYUI_DIR="$(cd "$COMFYUI_DIR" && pwd)"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$REPO_DIR/nodes.py" ] || die "run this from the repository root (nodes.py not found)"
[ -f "$REPO_DIR/$WEIGHTS_NAME" ] || true   # weights are not vendored; not an error

[ "$SKIP_NODE" -eq 1 ] || [ -n "$NODE_NAME" ] || die "--node-name cannot be empty"

# ---------------------------------------------------------------- python lookup
find_python() {
  local c
  for c in "$COMFYUI_DIR/.venv/bin/python" \
           "$COMFYUI_DIR/venv/bin/python" \
           "$COMFYUI_DIR/python_embeded/python.exe" \
           "$(command -v python3 || true)" \
           "$(command -v python || true)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then printf '%s' "$c"; return 0; fi
  done
  return 1
}
PY="$(find_python)" || die "could not find a Python interpreter (looked for .venv, venv, python_embeded)"
info "ComfyUI root : $COMFYUI_DIR"
info "Python       : $PY"
info "Interpreter  : $("$PY" -c 'import sys; print(sys.version.split()[0])' 2>/dev/null || echo '?')"

if ! "$PY" -c 'import torch' 2>/dev/null; then
  die "torch is not importable with $PY — point --comfyui at a working ComfyUI install"
fi
"$PY" - <<'PYEOF'
import torch
if torch.cuda.is_available():
    cc = torch.cuda.get_device_capability(0)
    print(f"  ok torch {torch.__version__} | {torch.cuda.get_device_name(0)} | SM {cc[0]}.{cc[1]}")
else:
    print(f"  ok torch {torch.__version__} | CUDA unavailable (CPU/MPS runs are possible but slow)")
PYEOF

if [ -n "$DLL" ]; then
  [ -f "$DLL" ] || die "no such DLL: $DLL"
  DLL="$(cd "$(dirname "$DLL")" && pwd)/$(basename "$DLL")"
fi

# ---------------------------------------------------------------- 1. node files
if [ "$SKIP_NODE" -eq 1 ]; then
  info "1/3 skipping node copy (--skip-node)"
else
  DEST="$COMFYUI_DIR/custom_nodes/$NODE_NAME"
  info "1/3 installing node -> $DEST"
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '  [dry-run] sync %s -> %s\n' "$REPO_DIR" "$DEST"
  else
    mkdir -p "$DEST"
    if command -v rsync >/dev/null 2>&1; then
      rsync -a --delete \
        --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
        --exclude 'models' --exclude 'weights' --exclude '*.safetensors' \
        --exclude "$WEIGHTS_NAME" \
        "$REPO_DIR"/ "$DEST"/
    else
      ( cd "$REPO_DIR" && tar cf - \
          --exclude='.git' --exclude='__pycache__' --exclude='*.pyc' \
          --exclude='models' --exclude='weights' --exclude='*.safetensors' . ) \
        | ( cd "$DEST" && tar xf - )
    fi
    ok "node files copied"
  fi
fi

# ---------------------------------------------------------------- 2. extractor
info "2/3 preparing the weight extractor (MLX-DLSS @ ${MLX_REV:0:10})"
if "$PY" -c 'import mlxdlss' 2>/dev/null; then
  ok "mlxdlss already installed"
else
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '  [dry-run] git clone MLX-DLSS into %s and pip install --no-deps ./python\n' "$WORK_DIR"
  else
    mkdir -p "$WORK_DIR"
    if [ ! -d "$WORK_DIR/MLX-DLSS/.git" ]; then
      run git clone https://github.com/iamwavecut/MLX-DLSS.git "$WORK_DIR/MLX-DLSS"
    fi
    ( cd "$WORK_DIR/MLX-DLSS" && git checkout -q "$MLX_REV" 2>/dev/null || warn "could not check out pinned revision; continuing on default branch" )
    run "$PY" -m pip install --no-deps --quiet "$WORK_DIR/MLX-DLSS/python"
    "$PY" -c 'import mlxdlss' 2>/dev/null || die "mlxdlss install failed"
    ok "mlxdlss installed"
  fi
fi

# ---------------------------------------------------------------- 3. weights
MODEL_DIR="$COMFYUI_DIR/models/$MODEL_DIR_NAME"
OUT="$MODEL_DIR/$WEIGHTS_NAME"
info "3/3 building weights -> $OUT"

if [ -z "$DLL" ]; then
  if [ -f "$OUT" ]; then
    ok "checkpoint already present; nothing to do (pass --dll to rebuild)"
  else
    die "--dll is required: no existing $WEIGHTS_NAME and no DLL supplied"
  fi
else
  SIZE="$(stat -c%s "$DLL" 2>/dev/null || echo 0)"
  info "    source DLL : $DLL ($((SIZE/1024/1024)) MB)"

  if [ "$DRY_RUN" -eq 1 ]; then
    printf '  [dry-run] mlxdlss-weights extract %s -> %s/packed.safetensors\n' "$DLL" "$WORK_DIR"
    printf '  [dry-run] mlxdlss-weights decode  packed -> %s\n' "$OUT"
  else
    mkdir -p "$WORK_DIR" "$MODEL_DIR"
    PACKED="$WORK_DIR/dlssnr-packed.safetensors"

    info "    reading the DLL as data (it is never executed)"
    "$PY" -m mlxdlss.tools.cli extract "$DLL" "$PACKED" || die "extract failed — is this a 310.8.0.0 neural-rendering DLL?"
    "$PY" -m mlxdlss.tools.cli decode  "$PACKED" "$OUT"        || die "decode failed"

    # structural gate: the canonical build yields 153 -> 649 with nothing opaque
    "$PY" - "$OUT" <<'PYEOF' || exit 1
import sys
from safetensors import safe_open
path = sys.argv[1]
with safe_open(path, framework="pt", device="cpu") as f:
    meta = f.metadata() or {}
    n = len(f.keys())
print(f"  ok decoded {n} logical tensors")
if meta.get("fully_logical") != "true":
    print("  ! metadata 'fully_logical' is not true", file=sys.stderr); sys.exit(1)
if n != 649:
    print(f"  ! expected 649 tensors, got {n}", file=sys.stderr); sys.exit(1)
print(f"  ok format {meta.get('format')} | fully_logical=true | tensor count matches")
PYEOF
    ok "checkpoint written ($(( $(stat -c%s "$OUT") /1024/1024 )) MB)"
    [ "${KEEP_TEMP:-0}" -eq 1 ] || rm -rf "$WORK_DIR"
  fi
fi

cat <<EOF

$(printf '\033[32mDone.\033[0m')

Next:
  1. restart ComfyUI so the custom node is rescanned
  2. add a node from the 'DLSS 5/PyTorch (experimental)' category
  3. point DLSS5PyTorchModelLoader at: models/$MODEL_DIR_NAME/$WEIGHTS_NAME
  4. sanity-check the install with: $PY verify_pre_rtx.py --comfyui "$COMFYUI_DIR"
EOF
