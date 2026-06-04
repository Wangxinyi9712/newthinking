#!/bin/bash
#SBATCH -o slurm_out_new.log
#SBATCH -e slurm_err_new.log
#SBATCH --partition=gpu-a10
#SBATCH --qos=normal
#SBATCH -J newthinking
#SBATCH --nodes=1
#SBATCH -c 16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

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

python src/scripts/run_ablation.py --config-dir src/configs --prefix brats_group_

python src/scripts/collect_epoch_metrics.py \
  --run-dirs runs/brats_group_a runs/brats_group_b runs/brats_group_c runs/brats_group_d runs/brats_group_e runs/brats_group_r1 runs/brats_group_r2 runs/brats_group_r3 runs/brats_group_r4 \
  --out runs/ablation_summary.csv

python src/scripts/analyze_stats.py \
  --csv runs/ablation_summary.csv \
  --metric dice \
  --out runs/stats_report_dice.json