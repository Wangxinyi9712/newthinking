#!/bin/bash
#SBATCH -o slurm_out_baselines.log
#SBATCH -e slurm_err_baselines.log
#SBATCH --partition=gpu-a10
#SBATCH --qos=normal
#SBATCH -J newthinking_baselines
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

set -euo pipefail

echo "=============================="
echo " NewThinking Baseline Training Job"
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

echo "[6/9] Start baseline experiments"

CONFIGS=(
  "src/configs/baselines/brats_baseline_f_nnunet.yaml"
  "src/configs/baselines/brats_baseline_g_bcp.yaml"
  "src/configs/baselines/brats_baseline_h_unimatch.yaml"
  "src/configs/baselines/brats_baseline_i_corrmatch.yaml"
  "src/configs/baselines/brats_baseline_j_wtbcp.yaml"
)

SEEDS=(0 1 2)

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

echo "[7/9] Summarize baseline results"

if [ -f "src/scripts/summarize_results.py" ]; then
  python src/scripts/summarize_results.py \
    --root runs/baselines \
    --out runs/baselines_summary.csv
else
  echo "[WARN] src/scripts/summarize_results.py not found. Skip summary."
fi

echo "[8/9] Finished"

echo "=============================="
echo " Job finished at $(date)"
echo " Outputs:"
echo "   runs/baselines/"
echo "   runs/baselines_summary.csv"
echo "=============================="