#!/bin/bash
#SBATCH --job-name=rc-kt-spontaneous
#SBATCH --partition=normal
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=rc-kt-spontaneous-%j.out
#SBATCH --error=rc-kt-spontaneous-%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${RC_KT_VENV:-${SCRIPT_DIR}/../../.venv}/bin/activate"
python "${SCRIPT_DIR}/batch_spontaneous.py" \
  --data-dir "${RC_KT_DATA_DIR:-${SCRIPT_DIR}/data}" \
  --output-dir "${RC_KT_OUTPUT_DIR:-${SCRIPT_DIR}/results}" \
  --seed "${RC_KT_SEED:-0}" --n-jobs 32
