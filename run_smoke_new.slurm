#!/bin/bash
#SBATCH -o slurm_out_%j.log
#SBATCH -e slurm_err_%j.log
#SBATCH --partition=gpu-a10
#SBATCH --qos=normal
#SBATCH -J newthinking_smoke
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

set -euo pipefail

echo "=============================="
echo " NewThinking Smoke Test Job"
echo " Job ID: ${SLURM_JOB_ID:-none}"
echo " Host: $(hostname)"
echo " Start time: $(date)"
echo "=============================="

# ==============================
# 1. Project settings
# ==============================

PROJECT_DIR="/home/xinyiwang/workdir/newthinking"
CONDA_ENV="newthinking"
BRATS_ROOT="data/BraTS2023"
SPLIT_FILE="data_splits/brats_split.json"
CONFIG_FILE="src/configs/experiments/brats_group_e_full.yaml"


echo "[1/8] GPU info"
nvidia-smi || true

echo "[2/8] Activate conda env"

if [ -f "/opt/app/anaconda3/etc/profile.d/conda.sh" ]; then
  source /opt/app/anaconda3/etc/profile.d/conda.sh
  conda activate "${CONDA_ENV}"
elif [ -f "/opt/app/anaconda3/bin/activate" ]; then
  source /opt/app/anaconda3/bin/activate "${CONDA_ENV}"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
else
  echo "[ERROR] Cannot find conda activation script."
  exit 1
fi

echo "[3/8] Enter project"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

echo "[4/8] Python / CUDA check"
python - <<'PY'
import sys
import torch

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
PY

echo "[5/8] Build BraTS split"
python src/scripts/build_splits.py \
  --dataset brats \
  --root "${BRATS_ROOT}" \
  --out "${SPLIT_FILE}" \
  --val-ratio 0.10 \
  --labeled-ratio 0.10 \
  --seed 3407

echo "[6/8] Smoke train"
python src/scripts/train.py \
  --config "${CONFIG_FILE}" \
  --seed 0 \
  --smoke

echo "[7/8] Smoke evaluate"
python -m src.scripts.evaluate \
  --config "${CONFIG_FILE}" \
  --ckpt best \
  --seed 0 \
  --source teacher \
  --max-val-batches 5

echo "[8/8] Smoke test finished"
echo "=============================="
echo " Job finished at $(date)"
echo " Outputs:"
echo "   runs/experiments/brats_group_e_full/seed_0/best.pt"
echo "   runs/experiments/brats_group_e_full/seed_0/last.pt"
echo "   runs/experiments/brats_group_e_full/seed_0/history.csv"
echo "   runs/experiments/brats_group_e_full/seed_0/eval_results_teacher.json"
echo "=============================="