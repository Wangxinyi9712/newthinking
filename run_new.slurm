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

nvidia-smi
source /opt/app/anaconda3/bin/activate newthinking
cd /home/xinyiwang/workdir/newthinking

python src/scripts/train.py --config src/configs/brats_group_e.yaml

python src/scripts/evaluate.py \
  --config src/configs/brats_group_e.yaml \
  --ckpt runs/brats_group_e/seed_0/best.pt \
  --source teacher

python src/scripts/evaluate.py \
  --config src/configs/brats_group_e.yaml \
  --ckpt runs/brats_group_e/seed_0/best.pt \
  --source student
python src/scripts/evaluate.py \
  --config src/configs/brats_group_e.yaml \
  --ckpt runs/brats_group_e/seed_0/best.pt \
  --source ensemble


python src/scripts/infer.py \
  --config src/configs/brats_group_e.yaml \
  --ckpt runs/brats_group_e/seed_0/best.pt \
  --out runs/brats_group_e/infer_seed0 \
  --source teacher

python src/scripts/infer.py \
  --config src/configs/brats_group_e.yaml \
  --ckpt runs/brats_group_e/seed_0/best.pt \
  --out runs/brats_group_e/infer_seed0 \
  --source student

python src/scripts/infer.py \
  --config src/configs/brats_group_e.yaml \
  --ckpt runs/brats_group_e/seed_0/best.pt \
  --out runs/brats_group_e/infer_seed0 \
  --source ensemble