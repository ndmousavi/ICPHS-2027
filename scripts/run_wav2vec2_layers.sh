#!/usr/bin/env bash
# Sequentially extract facebook/wav2vec2-large-xlsr-53 layer embeddings for the
# Persian corpus, one nkululeko "predict" run per layer (configs in
# nkululeko_configs/), on a single free GPU. Each run writes a CSV with the
# original file/speaker/task columns plus feat_0..feat_1023 embedding columns
# (--list mode preserves existing columns, see nkululeko/predict.py).
#
# Prerequisite: resample the corpus to 16kHz first (wav2vec2 needs 16kHz
# input but the corpus is natively 44.1kHz, and neither nkululeko module
# resamples on the fly). Use nkululeko's own resampler via its --folder CLI
# mode -- NOT --config: the --config-driven RESAMPLE.replace is mishandled
# (always overwrites in place, opposite of what the ini says) as of
# nkululeko 1.11.0 / main (2026-09-17). --folder is unaffected:
#   python -m nkululeko.resample --folder data/persian
#   python scripts/build_16k_manifest.py
# The first command writes "<name>_16kHz.wav" next to each original (originals
# untouched); the second rewrites persian_nkululeko_data.csv into
# persian_nkululeko_data_16k.csv pointing at those siblings.
#
# Usage:
#   ./run_wav2vec2_layers.sh [gpu_id]
#
# If gpu_id is omitted, the script picks the GPU with the least memory used
# (via nvidia-smi) at start time. Runs are sequential on purpose: each layer
# loads its own truncated copy of the model, so running layers in parallel on
# one GPU would just contend for memory for no speed benefit.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$SCRIPT_DIR/nkululeko_configs"
LOG_DIR="$SCRIPT_DIR/nkululeko_logs"
RESULT_DIR="$SCRIPT_DIR/nkululeko_results"
DATA_LIST="$SCRIPT_DIR/persian_nkululeko_data_16k.csv"
VENV_PY="$REPO_ROOT/venv/bin/python"
PYTHON="${PYTHON:-$([ -x "$VENV_PY" ] && echo "$VENV_PY" || echo python)}"
mkdir -p "$LOG_DIR" "$RESULT_DIR"

if [ ! -f "$DATA_LIST" ]; then
    echo "ERROR: $DATA_LIST not found. Resample first:" >&2
    echo "  $PYTHON -m nkululeko.resample --folder $REPO_ROOT/data/persian" >&2
    echo "  $PYTHON $SCRIPT_DIR/build_16k_manifest.py" >&2
    exit 1
fi

# torch orders CUDA devices by compute capability by default, NOT by PCI bus
# id -- so CUDA_VISIBLE_DEVICES=<nvidia-smi index> can silently select a
# *different* physical GPU than the one nvidia-smi reported as free. Forcing
# PCI_BUS_ID ordering makes torch's numbering match nvidia-smi's.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

GPU_ID="${1:-${GPU_ID:-}}"
if [ -z "$GPU_ID" ]; then
    GPU_ID=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t',' -k2 -n | head -n1 | cut -d',' -f1 | tr -d ' ')
fi
echo "Using GPU $GPU_ID (PCI_BUS_ID order) via $PYTHON"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

LAYERS=(03 06 09 12 15 18 21 24)

FAILED=()
for L in "${LAYERS[@]}"; do
    name="persian_wav2vec2_xlsr53_L${L}"
    cfg="$CONFIG_DIR/${name}.ini"
    outfile="$RESULT_DIR/${name}.csv"
    echo "=== $(date '+%F %T') running $name on GPU $GPU_ID ==="
    "$PYTHON" -m nkululeko.predict \
        --list "$DATA_LIST" \
        --model wav2vec2-large-xlsr-53 \
        --config "$cfg" \
        --type feats \
        --outfile "$outfile" \
        2>&1 | tee "$LOG_DIR/$name.log"
    status=${PIPESTATUS[0]}
    if [ "$status" -ne 0 ]; then
        echo "!!! $name failed (exit $status), continuing with next layer"
        FAILED+=("$name")
    fi
done

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "Finished with failures: ${FAILED[*]}"
    exit 1
fi
echo "All layers extracted successfully -> $RESULT_DIR/persian_wav2vec2_xlsr53_L*.csv"
