#!/usr/bin/env bash
# Full Track 1 pipeline: 3 LODO folds -> final all-data model -> decoder -> zip.
#   ./run_pipeline.sh [run_name] [extra train_lodo.py args...]
set -euo pipefail
cd "$(dirname "$0")"
NAME=${1:-convnext_tiny_v1}; shift || true
PY=.venv-rocm/bin/python
$PY train_lodo.py --holdout all-folds --name "$NAME" "$@"
$PY train_lodo.py --holdout none --name "$NAME" "$@"
$PY fit_decoder.py --run "$NAME"
$PY make_submission.py --run "$NAME"
