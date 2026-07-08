#!/bin/bash
#SBATCH -o slurm_out_cross.log
#SBATCH -e slurm_err_cross.log
#SBATCH --partition=gpu-a10
#SBATCH --qos=normal
#SBATCH -J newthinking_cross
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

set -euo pipefail

echo "=============================="
echo " NewThinking Cross-domain Method Job"
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

source /opt/app/anaconda3/bin/activate newthinking

echo "[3/9] Enter project"
cd /home/xinyiwang/workdir/newthinking

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

echo "[6/9] Start K-O cross-domain experiments"

CONFIGS=(
  "src/configs/experiments/brats_group_k_unimatch_sdf.yaml"
  "src/configs/experiments/brats_group_l_unimatch_pde.yaml"
  "src/configs/experiments/brats_group_m_unimatch_ot.yaml"
  "src/configs/experiments/brats_group_n_pde_sdf_ot_uniteacher.yaml"
  "src/configs/experiments/brats_group_o_pde_sdf_ot_proto_uniteacher.yaml"
)

#python src/scripts/train.py --config ./src/configs/experiments/brats_group_k_unimatch_sdf.yaml --seed 0 --smoke
#python src/scripts/train.py --config ./src/configs/experiments/brats_group_k_unimatch_sdf.yaml --seed 0
python src/scripts/train.py --config ./src/configs/experiments/brats_group_k_unimatch_sdf.yaml --seed 1
python src/scripts/train.py --config ./src/configs/experiments/brats_group_k_unimatch_sdf.yaml --seed 2
python src/scripts/train.py --config ./src/configs/experiments/brats_group_l_unimatch_pde.yaml --seed 0
python src/scripts/train.py --config ./src/configs/experiments/brats_group_l_unimatch_pde.yaml --seed 1
python src/scripts/train.py --config ./src/configs/experiments/brats_group_l_unimatch_pde.yaml --seed 2
#python src/scripts/train.py --config ./src/configs/experiments/brats_group_m_unimatch_ot.yaml --seed 0
#python src/scripts/train.py --config ./src/configs/experiments/brats_group_m_unimatch_ot.yaml --seed 1
#python src/scripts/train.py --config ./src/configs/experiments/brats_group_m_unimatch_ot.yaml --seed 2
#python src/scripts/train.py --config ./src/configs/experiments/brats_group_n_pde_sdf_ot_uniteacher.yaml --seed 0
#python src/scripts/train.py --config ./src/configs/experiments/brats_group_n_pde_sdf_ot_uniteacher.yaml --seed 1
#python src/scripts/train.py --config ./src/configs/experiments/brats_group_n_pde_sdf_ot_uniteacher.yaml --seed 2
#python src/scripts/train.py --config ./src/configs/experiments/brats_group_o_pde_sdf_ot_proto_uniteacher.yaml --seed 0
#python src/scripts/train.py --config ./src/configs/experiments/brats_group_o_pde_sdf_ot_proto_uniteacher.yaml --seed 1
#python src/scripts/train.py --config ./src/configs/experiments/brats_group_o_pde_sdf_ot_proto_uniteacher.yaml --seed 2



#python src/scripts/evaluate.py --config ./src/configs/experiments/brats_group_k_unimatch_sdf.yaml --ckpt best --seed 0 --max-val-batches 5
#python src/scripts/evaluate.py --config ./src/configs/experiments/brats_group_k_unimatch_sdf.yaml --ckpt best --seed 0 --source teacher
python src/scripts/evaluate.py --config ./src/configs/experiments/brats_group_k_unimatch_sdf.yaml --ckpt best --seed 1 --source teacher
python src/scripts/evaluate.py --config ./src/configs/experiments/brats_group_k_unimatch_sdf.yaml --ckpt best --seed 2 --source teacher
python src/scripts/evaluate.py --config ./src/configs/experiments/brats_group_l_unimatch_pde.yaml --ckpt best --seed 0 --source teacher
python src/scripts/evaluate.py --config ./src/configs/experiments/brats_group_l_unimatch_pde.yaml --ckpt best --seed 1 --source teacher
python src/scripts/evaluate.py --config ./src/configs/experiments/brats_group_l_unimatch_pde.yaml --ckpt best --seed 2 --source teacher
#python src/scripts/evaluate.py --config ./src/configs/experiments/brats_group_m_unimatch_ot.yaml --ckpt best --seed 0 --source teacher
#python src/scripts/evaluate.py --config ./src/configs/experiments/brats_group_m_unimatch_ot.yaml --ckpt best --seed 1 --source teacher
#python src/scripts/evaluate.py --config ./src/configs/experiments/brats_group_m_unimatch_ot.yaml --ckpt best --seed 2 --source teacher
#python src/scripts/evaluate.py --config ./src/configs/experiments/brats_group_n_pde_sdf_ot_uniteacher.yaml --ckpt best --seed 0 --source teacher
#python src/scripts/evaluate.py --config ./src/configs/experiments/brats_group_n_pde_sdf_ot_uniteacher.yaml --ckpt best --seed 1 --source teacher
#python src/scripts/evaluate.py --config ./src/configs/experiments/brats_group_n_pde_sdf_ot_uniteacher.yaml --ckpt best --seed 2 --source teacher
#python src/scripts/evaluate.py --config ./src/configs/experiments/brats_group_o_pde_sdf_ot_proto_uniteacher.yaml --ckpt best --seed 0 --source teacher
#python src/scripts/evaluate.py --config ./src/configs/experiments/brats_group_o_pde_sdf_ot_proto_uniteacher.yaml --ckpt best --seed 1 --source teacher
#python src/scripts/evaluate.py --config ./src/configs/experiments/brats_group_o_pde_sdf_ot_proto_uniteacher.yaml --ckpt best --seed 2 --source teacher