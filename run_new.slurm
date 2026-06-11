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
  --ckpt best \
  --split seed_0 \
  --source teacher

python src/scripts/evaluate.py \
  --config src/configs/brats_group_e.yaml \
  --ckpt best \
  --split seed_0 \
  --source student

python src/scripts/evaluate.py \
  --config src/configs/brats_group_e.yaml \
  --ckpt best \
  --split seed_0 \
  --source ensemble

python src/scripts/evaluate.py \
  --config src/configs/brats_group_e.yaml \
  --ckpt last \
  --split seed_0 \
  --source teacher


python src/scripts/infer.py \
  --config src/configs/brats_group_e.yaml \
  --ckpt best \
  --split seed_0 \
  --out runs/brats_group_e/infer_best_seed0_teacher \
  --source teacher

python src/scripts/infer.py \
  --config src/configs/brats_group_e.yaml \
  --ckpt best \
  --split seed_0 \
  --out runs/brats_group_e/infer_best_seed0_student \
  --source student

python src/scripts/infer.py \
  --config src/configs/brats_group_e.yaml \
  --ckpt best \
  --split seed_0 \
  --out runs/brats_group_e/infer_best_seed0_ensemble \
  --source ensemble

python method/run_method_suite.py \
  --base-config src/configs/brats_group_e.yaml \
  --suite src/configs/method_suite.yaml

# 如需直接跑所有方法 + 全seed，取消下面注释
python method/run_method_suite.py \
 --base-config src/configs/brats_group_e.yaml \
 --suite src/configs/method_suite.yaml \
 --run

# 汇总方法对比结果
python method/summarize_method_suite.py \
  --suite src/configs/method_suite.yaml \
  --runs-root runs/method_suite \
  --out method/method_comparison.csv