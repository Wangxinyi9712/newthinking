#!/bin/bash
#SBATCH -o slurm_out_ours.log
#SBATCH -e slurm_err_ours.log
#SBATCH --partition=gpu-a10
#SBATCH --qos=normal
#SBATCH -J newthinking_ours
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

set -euo pipefail

echo "=============================="
echo " NewThinking Revised Our Method"
echo " Job ID: ${SLURM_JOB_ID:-none}"
echo " Host: $(hostname)"
echo " Start time: $(date)"
echo "=============================="

PROJECT_DIR="/home/xinyiwang/workdir/newthinking"
CONDA_ENV="newthinking"
BRATS_ROOT="data/BraTS2023"
SPLIT_FILE="data_splits/brats_split.json"
CONFIG_FILE="src/configs/ours.yaml"

echo "[1/7] GPU info"
nvidia-smi || true

echo "[2/7] Activate conda env"

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

echo "[3/7] Enter project"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

echo "[4/7] Python / CUDA check"
python - <<'PY'
import sys
import torch

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
PY

echo "[5/7] Build BraTS split"
python src/scripts/build_splits.py \
  --dataset brats \
  --root "${BRATS_ROOT}" \
  --out "${SPLIT_FILE}" \
  --val-ratio 0.10 \
  --labeled-ratio 0.10 \
  --seed 3407

echo "[6/7] Train revised Ours"
python src/scripts/train.py \
  --config "${CONFIG_FILE}" \
  --seeds 0 1 2

echo "[7/7] Evaluate revised Ours"

for SEED in 0 1 2; do
  python -m src.scripts.evaluate \
    --config "${CONFIG_FILE}" \
    --ckpt best \
    --seed "${SEED}" \
    --source teacher
done

echo "=============================="
echo " Job finished at $(date)"
echo " Outputs:"
echo "   runs/ours/seed_0"
echo "   runs/ours/seed_1"
echo "   runs/ours/seed_2"
echo "=============================="