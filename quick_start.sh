#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

print_usage() {
  cat <<'EOF'
SafePath Quick Start

Usage:
  ./quick_start.sh <command>

Commands:
  setup            Create .venv and install requirements
  check            Verify key project artifacts and directories
  validate-mini    Validate the processed mini dataset
  validate-full    Validate the processed full BDD100K dataset
  train-mini-base  Train the baseline model on the mini config
  train-mini-prop  Train the proposed model on the mini config
  train-full-smoke Run a 1-epoch baseline smoke test on full BDD100K
  train-full-base  Train the baseline model on full BDD100K
  train-full-prop  Train the proposed model on full BDD100K
  eval-mini        Run baseline vs proposed evaluation on the mini dataset
  eval-full-val    Run baseline vs proposed evaluation on BDD100K val split
  export           Export both trained models to ONNX
  tensorboard      Print the command to launch TensorBoard
  help             Show this help message

Notes:
  - Run commands from anywhere; the script resolves the project root.
  - Override Python with: PYTHON_BIN=python3.11 ./quick_start.sh check
EOF
}

run_python() {
  (cd "$PROJECT_ROOT" && "$PYTHON_BIN" "$@")
}

require_file() {
  local path="$1"
  if [[ ! -e "$PROJECT_ROOT/$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
}

cmd_setup() {
  cd "$PROJECT_ROOT"
  if [[ ! -d .venv ]]; then
    "$PYTHON_BIN" -m venv .venv
  fi
  . .venv/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements.txt
  echo "Setup complete. Activate with: source .venv/bin/activate"
}

cmd_check() {
  echo "Project root: $PROJECT_ROOT"
  require_file "requirements.txt"
  require_file "configs/train_mini.yaml"
  require_file "scripts/train_baseline.py"
  require_file "scripts/train_proposed.py"
  require_file "scripts/eval_models.py"
  require_file "reports/midterm/AI688-001-Midterm.md"

  echo
  echo "Processed datasets:"
  ls -1 "$PROJECT_ROOT/data/processed" || true

  echo
  echo "Existing checkpoints:"
  ls -1 "$PROJECT_ROOT/models/baseline_mobilenetv3_small/checkpoints" 2>/dev/null || echo "  baseline checkpoints missing"
  ls -1 "$PROJECT_ROOT/models/proposed_deeplabv3plus/checkpoints" 2>/dev/null || echo "  proposed checkpoints missing"

  echo
  echo "Existing exports:"
  ls -1 "$PROJECT_ROOT/exports" 2>/dev/null || echo "  no exports found"
}

cmd_validate_mini() {
  require_file "data/processed/bdd100k_mini/images/train"
  run_python scripts/validate_data.py --data-dir data/processed/bdd100k_mini --split all --strict
}

cmd_validate_full() {
  require_file "data/processed/bdd100k/images/train"
  run_python scripts/validate_data.py --data-dir data/processed/bdd100k --split val --strict
}

cmd_train_mini_base() {
  run_python scripts/train_baseline.py \
    --config configs/train_mini.yaml \
    --data-dir data/processed/bdd100k_mini \
    --output-dir models/baseline_mobilenetv3_small
}

cmd_train_mini_prop() {
  run_python scripts/train_proposed.py \
    --config configs/train_mini.yaml \
    --data-dir data/processed/bdd100k_mini \
    --output-dir models/proposed_deeplabv3plus
}

cmd_train_full_smoke() {
  run_python scripts/train_baseline.py \
    --data-dir data/processed/bdd100k \
    --output-dir models/baseline_full_smoke \
    --epochs 1 \
    --batch-size 4 \
    --lr 0.001 \
    --num-workers 0
}

cmd_train_full_base() {
  run_python scripts/train_baseline.py \
    --data-dir data/processed/bdd100k \
    --output-dir models/baseline_full \
    --epochs 15 \
    --batch-size 8 \
    --lr 0.001 \
    --num-workers 0
}

cmd_train_full_prop() {
  run_python scripts/train_proposed.py \
    --data-dir data/processed/bdd100k \
    --output-dir models/proposed_full \
    --epochs 15 \
    --batch-size 8 \
    --lr 0.001 \
    --num-workers 0
}

cmd_eval_mini() {
  require_file "models/baseline_mobilenetv3_small/checkpoints/best.pt"
  require_file "models/proposed_deeplabv3plus/checkpoints/best.pt"
  run_python scripts/eval_models.py \
    --compare \
    --baseline-checkpoint models/baseline_mobilenetv3_small/checkpoints/best.pt \
    --proposed-checkpoint models/proposed_deeplabv3plus/checkpoints/best.pt \
    --data-dir data/processed/bdd100k_mini \
    --split val \
    --input-size 128 256 \
    --batch-size 4 \
    --num-workers 2 \
    --benchmark-fps \
    --num-warmup 5 \
    --num-iterations 25 \
    --output reports/midterm/results/comparison.md
}

cmd_eval_full_val() {
  require_file "models/baseline_full/checkpoints/best.pt"
  require_file "models/proposed_full/checkpoints/best.pt"
  run_python scripts/eval_models.py \
    --compare \
    --baseline-checkpoint models/baseline_full/checkpoints/best.pt \
    --proposed-checkpoint models/proposed_full/checkpoints/best.pt \
    --data-dir data/processed/bdd100k \
    --split val \
    --num-workers 0 \
    --benchmark-fps \
    --output reports/midterm/results/comparison_full.md
}

cmd_export() {
  require_file "models/baseline_mobilenetv3_small/checkpoints/best.pt"
  require_file "models/proposed_deeplabv3plus/checkpoints/best.pt"

  run_python scripts/export_onnx.py \
    --model baseline \
    --checkpoint models/baseline_mobilenetv3_small/checkpoints/best.pt \
    --output exports/baseline.onnx \
    --input-size 128 256

  run_python scripts/export_onnx.py \
    --model proposed \
    --checkpoint models/proposed_deeplabv3plus/checkpoints/best.pt \
    --output exports/proposed.onnx \
    --input-size 128 256
}

cmd_tensorboard() {
  cat <<'EOF'
Run one of these commands from the project root:

tensorboard --logdir models/baseline_mobilenetv3_small/logs --port 6006
tensorboard --logdir models/proposed_deeplabv3plus/logs --port 6007
EOF
}

main() {
  local command="${1:-help}"
  case "$command" in
    setup) cmd_setup ;;
    check) cmd_check ;;
    validate-mini) cmd_validate_mini ;;
    validate-full) cmd_validate_full ;;
    train-mini-base) cmd_train_mini_base ;;
    train-mini-prop) cmd_train_mini_prop ;;
    train-full-smoke) cmd_train_full_smoke ;;
    train-full-base) cmd_train_full_base ;;
    train-full-prop) cmd_train_full_prop ;;
    eval-mini) cmd_eval_mini ;;
    eval-full-val) cmd_eval_full_val ;;
    export) cmd_export ;;
    tensorboard) cmd_tensorboard ;;
    help|-h|--help) print_usage ;;
    *)
      echo "Unknown command: $command" >&2
      print_usage
      exit 1
      ;;
  esac
}

main "$@"
