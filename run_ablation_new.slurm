#!/bin/bash
#SBATCH -o slurm_out_%j.log
#SBATCH -e slurm_err_%j.log
#SBATCH --partition=gpu-a10
#SBATCH --qos=normal
#SBATCH -J newthinking_ablation
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

set -euo pipefail

echo "=============================="
echo " NewThinking Ablation Training Job"
echo " Job ID: ${SLURM_JOB_ID:-none}"
echo " Host: $(hostname)"
echo " Start time: $(date)"
echo "=============================="

PROJECT_DIR="/home/xinyiwang/workdir/newthinking"
CONDA_ENV="newthinking"
BRATS_ROOT="data/BraTS2023"
SPLIT_FILE="data_splits/brats_split.json"


echo "[1/9] GPU info"
nvidia-smi || true

echo "[2/9] Activate conda env"

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

echo "[3/9] Enter project"
cd "${PROJECT_DIR}"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

echo "[4/9] Python / CUDA check"
python - <<'PY'
import sys
import torch

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
PY

echo "[5/9] Build BraTS split"
python src/scripts/build_splits.py \
  --dataset brats \
  --root "${BRATS_ROOT}" \
  --out "${SPLIT_FILE}" \
  --val-ratio 0.10 \
  --labeled-ratio 0.10 \
  --seed 3407

echo "[6/9] Define experiment groups"

CONFIGS=(
  "src/configs/experiments/brats_group_a_supervised.yaml"
  "src/configs/experiments/brats_group_b_mean_teacher.yaml"
  "src/configs/experiments/brats_group_c_frequency.yaml"
  "src/configs/experiments/brats_group_d_prototype.yaml"
  "src/configs/experiments/brats_group_e_full.yaml"
)

SEEDS=(0 1 2)

echo "[7/9] Start A-E ablation experiments"

for CFG in "${CONFIGS[@]}"; do
  if [ ! -f "${CFG}" ]; then
    echo "[ERROR] Config not found: ${CFG}"
    exit 1
  fi

  echo "=============================="
  echo " Config: ${CFG}"
  echo "=============================="

  for SEED in "${SEEDS[@]}"; do
    echo "------------------------------"
    echo " Train: config=${CFG}, seed=${SEED}"
    echo "------------------------------"

    python src/scripts/train.py \
      --config "${CFG}" \
      --seed "${SEED}"

    echo "------------------------------"
    echo " Evaluate: config=${CFG}, seed=${SEED}"
    echo "------------------------------"

    python -m src.scripts.evaluate \
      --config "${CFG}" \
      --ckpt best \
      --seed "${SEED}" \
      --source teacher
  done
done

echo "[8/9] Summarize results"

if [ -f "src/scripts/summarize_results.py" ]; then
  python src/scripts/summarize_results.py \
    --root runs/experiments \
    --out runs/experiments_summary.csv
else
  echo "[WARN] src/scripts/summarize_results.py not found. Skip summary."
fi

echo "[9/9] Finished"
echo "=============================="
echo " Job finished at $(date)"
echo " Outputs:"
echo "   runs/experiments/"
echo "   runs/experiments_summary.csv"
echo "=============================="