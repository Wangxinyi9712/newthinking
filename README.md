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

### 4.1 单组训练（自动按 `seed: [0,1,2]` 重复）

```bash
python src/scripts/train.py --config src/configs/brats_group_e.yaml
```

输出目录示例：

```text
runs/brats_group_e/
  seed_0/
    history.csv
    best.pt
    config_used.json
  seed_1/...
  seed_2/...
  repeats_summary.csv
  repeats_aggregate.json
```

### 4.2 评估

```bash
python src/scripts/evaluate.py \
  --config src/configs/brats_group_e.yaml \
  --ckpt runs/brats_group_e/seed_0/best.pt \
  --source teacher
```

`--source` 可选：`teacher | student | ensemble`

### 4.3 推理可视化

```bash
python src/scripts/infer.py \
  --config src/configs/brats_group_e.yaml \
  --ckpt runs/brats_group_e/seed_0/best.pt \
  --out runs/brats_group_e/infer_seed0 \
  --source teacher
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

## 6. 实验设计（与代码对齐）

### 6.1 数据与预处理（BraTS 3D）

- 体素尺寸与 patch 统一：`96×96×96`（稳定训练，后续可升 112/128）
- 归一化：`NormalizeIntensityd`
- 增强：
  - `RandSpatialCropd`
  - `RandFlipd`
  - `RandAffined`
  - 无标注分支额外 `RandScaleIntensityd` + `RandGaussianNoised`
- 数据加载：`batch_size=1, num_workers=0, pin_memory=false`
- AMP + 梯度累积：`use_amp=true, grad_accum_steps=2`

---

## 7. 分组实验设置（BraTS）

> 统一资源设置（公平比较）：
>
> - `spatial_size=[96,96,96]`
> - `batch_size=1`
> - `num_workers=0`
> - `pin_memory=false`
> - `cache_rate=0.1`
> - `use_amp=true`
> - `grad_accum_steps=2`
> - `seed=[0,1,2]`

### Group A（全监督基线）
- 模型：U-Net（无 Transformer）
- 半监督：关闭（`lambda_ssl=0.0`）
- 少数类：关闭
- 结构先验：关闭
- 可靠性：关闭

### Group B（基础半监督 + 固定伪标签阈值）
- A + `lambda_ssl=1.0`
- `tau=0.95`（更严格伪标签）
- 少数类 / 结构 / 可靠性：关闭

### Group C（半监督 + 动态权重）
- B 但 `tau=0.7`（动态伪标签更积极）
- 少数类 / 结构 / 可靠性：关闭

### Group D（C + 少数类敏感）
- C + `lambda_minor=1.0`
- 开启少数类相关权重（`minor_class_weights=[2.0]`）
- 结构先验：关闭
- 可靠性：关闭

### Group E（完整模型）
- D + 结构先验（`lambda_struct=0.2`）
- + 特征一致性（`lambda_feat_consistency=0.1`）
- + 可靠性融合（confidence/entropy/consistency/ood）
- + 多尺度注意力 + Transformer 编码器
- + 少数类过采样

### R1（去除 OOD）
- 与 E 相同，`ood=false`
- 验证 OOD 组件贡献

### R2（去除少数类增强）
- 与 E 相同，但：
  - `minority_score=false`
  - `minority_oversample=false`
  - `lambda_minor=0.0`
- 验证少数类模块贡献

### R3（去除 consistency）
- 与 E 相同，但 `consistency=false`、`lambda_feat_consistency=0.0`
- 验证一致性分支贡献

### R4（仅 reliability 主体）
- 仅保留可靠性融合
- 关闭 minority / ood / consistency / struct / feat-consistency
- 验证 reliability 主干能力

---

## 8. 指标与统计

每个 epoch 记录：

- train_loss
- val_dice / iou / precision / recall / f1 / minority_f1 / hd95
- unsup_conf_mean / reliability_mean / ood_mean / consistency_mean

最终统计：

- 单因素 ANOVA
- Tukey HSD
- Cohen’s d
- 95% CI

见：

- `src/analysis/stats.py`
- `src/scripts/analyze_stats.py`

---

## 9. 复现实验建议流程

1. 生成 split（确保过滤 `._` 脏文件）
2. 跑单组 E 冒烟测试
3. 跑全部 ablation（A~E, R1~R4）
4. 汇总与统计
5. 导出可视化成功/失败样例与误差图

---

## 10. 引用说明

如果你在论文中使用本仓库，请在方法章节描述以下关键模块：

- Mean Teacher 半监督框架
- 动态伪标签权重与可靠性融合
- 少数类敏感损失
- 结构先验（边界+拓扑+HD）
- 多尺度注意力与轻量 Transformer 编码器