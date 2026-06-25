#!/bin/bash
#SBATCH -o slurm_out_new.log
#SBATCH -e slurm_err_new.log
#SBATCH --partition=gpu-l20
#SBATCH --qos=normal
#SBATCH -J newthinking
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

set -euo pipefail

echo "=============================="
echo " NewThinking Training Job"
echo " Job ID: ${SLURM_JOB_ID:-none}"
echo " Host: $(hostname)"
echo " Start time: $(date)"
echo "=============================="

mkdir -p logs

echo "[1/6] GPU info"
nvidia-smi || true

echo "[2/6] Activate conda env"
source /opt/app/anaconda3/bin/activate newthinking

echo "[3/6] Enter project"
cd /home/xinyiwang/workdir/newthinking

echo "[4/6] Python / CUDA check"
python - <<'PY'
import sys
import torch

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
PY

echo "[5/6] Build BraTS split"
python src/scripts/build_splits.py \
  --dataset brats \
  --root data/BraTS2023 \
  --out data_splits/brats_split.json \
  --val-ratio 0.10 \
  --labeled-ratio 0.10 \
  --seed 3407

echo "[6/6] Train"
python src/scripts/train.py

echo "[7/7] Evaluate best checkpoint"
python -m src.scripts.evaluate --ckpt best

echo "=============================="
echo " Job finished at $(date)"
echo " Outputs:"
echo "   runs/brats_group_e/seed_0/best.pt"
echo "   runs/brats_group_e/seed_0/last.pt"
echo "   runs/brats_group_e/seed_0/history.csv"
echo "   runs/brats_group_e/seed_0/eval_results.json"
echo "=============================="