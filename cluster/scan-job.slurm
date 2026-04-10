#!/bin/bash
#SBATCH --job-name=crosswalk-scan
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: sbatch scan-job.slurm <job-json> <result-json>"
  exit 1
fi

JOB_JSON="$1"
RESULT_JSON="$2"

ROOT_DIR="${ROOT_DIR:-$PWD}"
cd "$ROOT_DIR"

mkdir -p logs "$(dirname "$RESULT_JSON")"

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
fi

python -m crosswalk_detector.cli run-scan-job \
  --job-file "$JOB_JSON" \
  --output "$RESULT_JSON" \
  --progress
