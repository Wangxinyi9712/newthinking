# NewThinking: Robust Semi-Supervised Medical Image Segmentation

面向 2D/3D 医学图像分割的半监督框架，核心包含：

- U-Net / 3D U-Net（`HybridUNet`）
- Mean Teacher (EMA Teacher)
- 动态伪标签权重（置信度+可靠性融合）
- 少数类敏感损失
- 结构先验损失（边界/拓扑/Hausdorff）
- 多尺度注意力 + 轻量 Transformer 编码器

支持数据集：

- **BraTS 2021/2023**（3D MRI）
- **ISIC 2018**（2D 皮肤病变）
- **MSD LiTS (Task03 Liver)**（3D CT）

---

## 1. 项目结构

```text
src/
  analysis/
    stats.py
  configs/
    brats_group_a.yaml
    brats_group_b.yaml
    brats_group_c.yaml
    brats_group_d.yaml
    brats_group_e.yaml
    brats_group_r1.yaml
    brats_group_r2.yaml
    brats_group_r3.yaml
    brats_group_r4.yaml
    method_suite.yaml
  data/
    datasets.py
    examples.py
    transforms.py
  engine/
    trainer.py
    infer.py
  losses/
    seg_losses.py
  models/
    modules.py
    seg_model.py
  scripts/
    train.py
    evaluate.py
    infer.py
    build_splits.py
    collect_epoch_metrics.py
    analyze_stats.py
    run_ablation.py
  utils/
    config.py
    metrics.py
    seed.py
method/
  run_method_suite.py              
  summarize_method_suite.py        
```

---

## 2. 环境安装

推荐 Python 3.10 + CUDA 环境。

```bash
conda create -n newthinking python=3.10 -y
conda activate newthinking

# 按你的 CUDA 版本安装 torch
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

pip install -r requirements.txt
# 或 pip install -e .
```

---

## 3. 数据准备

请先下载官方数据并解压到 `data/` 下（参考 `dataprecessing.sh`）。

### 3.1 生成 split 文件

> 已支持自动过滤 `._xxx`（macOS 资源叉）脏文件。

```bash
# BraTS
python src/scripts/build_splits.py --dataset brats --root data/BraTS2023 --out data_splits/brats_split.json --val-ratio 0.10 --labeled-ratio 0.10 --seed 3407

# ISIC
python src/scripts/build_splits.py --dataset isic --root data/ISIC2018 --out data_splits/isic_split.json --val-ratio 0.10 --labeled-ratio 0.40 --seed 3407

# MSD LiTS
python src/scripts/build_splits.py --dataset msd_liver --root data/MSD_Liver --out data_splits/msd_liver_split.json --val-ratio 0.10 --labeled-ratio 0.10 --seed 3407
```

---

## 4. 训练 / 评估 / 推理

### 4.1 单组训练（按 seed 运行）

```bash
python src/scripts/train.py --config src/configs/brats_group_e.yaml --seed 0
```

输出目录示例：

```text
runs/brats_group_e/
  seed_0/
    history.csv
    best.pt
    last.pt
    checkpoint_summary.txt
    config_used.json
  seed_1/...
  seed_2/...
```

### 4.2 评估（支持 best|last 别名）

```bash
python src/scripts/evaluate.py \
  --config src/configs/brats_group_e.yaml \
  --ckpt best \
  --split seed_0 \
  --source teacher
```

或指定绝对路径：

```bash
python src/scripts/evaluate.py \
  --config src/configs/brats_group_e.yaml \
  --ckpt runs/brats_group_e/seed_0/best.pt \
  --source teacher
```

`--source` 可选：`teacher | student | ensemble`

### 4.3 推理（支持 best|last 别名）

```bash
python src/scripts/infer.py \
  --config src/configs/brats_group_e.yaml \
  --ckpt last \
  --split seed_0 \
  --source teacher \
  --out runs/brats_group_e/infer_last_seed0
```

---

## 5. Ablation 实验

### 5.1 一键运行全部组

```bash
python src/scripts/run_ablation.py --config-dir src/configs --prefix brats_group_
```

默认组：`a b c d e r1 r2 r3 r4`

### 5.2 手动收集与统计

```bash
python src/scripts/collect_epoch_metrics.py \
  --run-dirs runs/brats_group_a runs/brats_group_b runs/brats_group_c runs/brats_group_d runs/brats_group_e runs/brats_group_r1 runs/brats_group_r2 runs/brats_group_r3 runs/brats_group_r4 \
  --out runs/ablation_summary.csv

python src/scripts/analyze_stats.py \
  --csv runs/ablation_summary.csv \
  --metric dice \
  --out runs/stats_report_dice.json
```

---

## 6. Method 对比实验（论文表格 Method 列）

本项目已支持将 Method 列设置为可复现实验协议（位于 `src/configs/method_suite.yaml`），并自动生成各方法配置、训练、汇总。

### 6.1 协议位置

- `src/configs/method_suite.yaml`

### 6.2 生成各方法配置

```bash
python method/run_method_suite.py \
  --base-config src/configs/brats_group_e.yaml
```

生成目录：

```text
method/generated/
  supervised.yaml
  self_training.yaml
  gan_ssl.yaml
  mean_teacher.yaml
  mt_reliability.yaml
  ours.yaml
```

### 6.3 一键运行 method suite

```bash
python method/run_method_suite.py \
  --base-config src/configs/brats_group_e.yaml \
  --run
```

输出目录：

```text
runs/method_suite/<method_name>/seed_0|1|2/...
```

### 6.4 汇总 method 对比结果

```bash
python method/summarize_method_suite.py
```

输出：

- `method/method_comparison.csv`

---

## 7. 实验设计（与代码对齐，BraTS 3D）

### 7.1 统一资源设置（公平比较）

- `spatial_size=[96,96,96]`
- `batch_size=1`
- `num_workers=0`
- `pin_memory=false`
- `cache_rate=0.1`
- `use_amp=true`
- `grad_accum_steps=2`
- `seed=[0,1,2]`

### 7.2 Group E（当前完整模型）

- Mean Teacher + EMA teacher
- Reliability fusion（confidence/entropy/consistency/ood）
- CPS（双学生交叉伪监督）
- 少数类敏感损失 + 过采样
- 结构先验（边界+拓扑+HD）
- SDM 形状分支
- 可选 adversarial 分支

---

## 8. 指标与统计

每个 epoch 记录：

- `train_loss`
- `val_dice / val_iou / val_precision / val_recall / val_f1 / val_minority_f1 / val_hd95`
- `unsup_conf_mean / reliability_mean / ood_mean / consistency_mean`

建议报告：

- 3 seeds 的 mean ± std
- 单因素 ANOVA / Tukey HSD / Cohen’s d / 95% CI

---

## 9. 复现实验建议流程

1. 生成 split（确保过滤 `._` 脏文件）
2. 跑单组 E 冒烟测试（seed=0）
3. 跑 E 组 seeds=[0,1,2]
4. 跑 method suite（Supervised/MT/Ours 等）
5. 汇总与统计
6. 导出可视化成功/失败样例与误差图

---

## 10. 引用说明

若用于论文，请在方法部分描述以下关键模块：

- Mean Teacher 半监督框架（EMA teacher）
- 动态伪标签权重与可靠性融合
- CPS 双学生交叉伪监督
- 少数类敏感损失
- 结构先验（边界+拓扑+HD）
- 多尺度注意力与轻量 Transformer 编码器
- （可选）对抗判别分支