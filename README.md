# OBB 目标检测：模型选择与统一评估

本项目当前聚焦旋转目标框检测的模型选择，核心目标是提升严格 IoU 阈值下的定位精度，重点指标为 `mAP85`、`mAP90` 和 `mAP95`。当前主线比较单模型 YOLO11l-OBB baseline 与 Transformer 架构 RHINO R50-KLD，不再采用 deskew 或双模型 fusion 作为工程方案。

## 1. 当前检测流程

```text
原始图像
  -> OBB 检测模型（YOLO11l 或 RHINO）
  -> 旋转框、类别、置信度
  -> 统一 Ultralytics OBB 指标评估
  -> 按 label3/label5 检测框裁剪
  -> 下游 ResNet18 OK/NG 分类
```

检测类别共 7 类：

```text
label1_thin
label1_thick
label2
label3
label4
label5
label6
```

`label1` 按标注框上沿宽度拆分：

- `label1_thin`：上沿宽度 `< 164 px`
- `label1_thick`：上沿宽度 `>= 164 px`

## 2. 当前数据集

YOLO11l 和 RHINO 使用相同的样本、类别与 train/test 划分：

| 用途 | 路径 | 图像格式 | 标注格式 |
| --- | --- | --- | --- |
| YOLO11l | `datasets/obb_thin_thick` | BMP | YOLO OBB 归一化四点 |
| RHINO | `datasets/rhino_obb` | PNG | DOTA 四点 annfile |

RHINO 数据集由 YOLO 主数据集转换而来，只改变图像与标注的存储格式，不改变图像内容和划分。

| split | images | objects | label1_thin | label1_thick | label2-label6 每类 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 198 | 1188 | 132 | 66 | 198 |
| test | 53 | 318 | 35 | 18 | 53 |

数据来源口径：只保留时间戳 `20260121210219803` 之后的样本，并排除文件名末尾 index 为 `-1` 的样本。原 `other/label7` 不参与训练和评估。

## 3. 统一评估口径

所有模型最终统一使用 `ultralytics.utils.metrics.batch_probiou` 计算：

```text
precision, recall, mAP50, mAP80, mAP85, mAP90, mAP95
```

该评估器与具体检测模型无关。YOLO11l 可直接输出 OBB；RHINO 先把 `(cx, cy, w, h, angle)` 转换为四点 OBB，再进入相同评估器。

```text
YOLO11l best.pt
  -> YOLO OBB 预测
  -> custom_metrics.csv

RHINO best.pth
  -> predictions.pkl
  -> labels/*.txt
  -> custom_metrics.csv
```

RHINO 训练期间显示的 `dota/mAP` 仅为 polygon IoU 下的 AP50，用于训练监控和 checkpoint 选择，不能直接与最终的 mAP80-95 混用。

## 4. 模型与训练配置

| 项目 | YOLO11l-OBB baseline | RHINO R50-KLD |
| --- | --- | --- |
| 架构 | CNN 单阶段 OBB | ResNet50 + Transformer 检测器 |
| 输入尺寸 | 1280 | 1280 |
| epochs | 50 | 50 |
| batch | 8 | 2 |
| 主要损失 | Ultralytics OBB 默认损失 | Hausdorff matching + KLD |
| 最佳权重选择 | Ultralytics validation | DOTA AP50 |
| 部署形态 | 单模型、单环境 | 单模型，但依赖 RHINO/MMCV/MMDet 环境 |

当前权重：

```text
YOLO11l:
runs/obb/yolo11l_after212102_no_index1_label1_thin_thick_e50_img1280_b8_deg0_valtest/weights/best.pt

RHINO R50-KLD:
runs/rhino/rhino_r50_kld_e50_img1280_b2/best_dota_mAP_epoch_40.pth
```

RHINO 的 epoch 40 是训练阶段 AP50 最佳权重。由于目标指标是 mAP90，还需要把 `epoch_50.pth` 作为候选进行同口径评估。

## 5. 整体结果对比

以下结果来自相同 test split 和相同 Ultralytics OBB 评估器。

| model | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11l-OBB baseline | 0.997159 | 0.999863 | 0.995000 | 0.962807 | 0.894190 | **0.711124** | 0.183112 |
| RHINO R50-KLD epoch40 | 0.991563 | 0.998104 | 0.994762 | 0.923683 | 0.830400 | 0.648533 | **0.211504** |
| RHINO - YOLO11l | -0.005596 | -0.001759 | -0.000238 | -0.039124 | -0.063790 | -0.062591 | +0.028392 |

结果说明：

- 两个模型的 mAP50 几乎一致，检测和分类能力都接近饱和。
- RHINO 在 mAP80、mAP85、mAP90 均低于 YOLO11l，其中整体 mAP90 下降 `0.062591`。
- RHINO 的 mAP95 高 `0.028392`，说明部分框在极严格阈值下更准确，但收益没有覆盖 mAP80-90 的整体退化。
- 当前模型选择仍保留 YOLO11l-OBB 作为 baseline 和工程首选。

## 6. 分类别高 IoU 对比

| class | YOLO mAP85 | RHINO mAP85 | YOLO mAP90 | RHINO mAP90 | 差值 mAP90 | YOLO mAP95 | RHINO mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| label1_thin | 0.649216 | **0.690368** | 0.335726 | **0.446060** | +0.110334 | 0.014286 | **0.045196** |
| label1_thick | **0.775000** | 0.445073 | **0.643462** | 0.082473 | -0.560989 | 0.000000 | **0.005225** |
| label2 | **0.970283** | 0.905102 | **0.816797** | 0.732371 | -0.084426 | **0.264728** | 0.174382 |
| label3 | **0.943141** | 0.942571 | 0.586077 | **0.688773** | +0.102696 | **0.131533** | 0.120125 |
| label4 | 0.995000 | 0.995000 | **0.967453** | 0.953826 | -0.013627 | 0.492787 | **0.692595** |
| label5 | **0.942822** | 0.919599 | 0.674799 | **0.721139** | +0.046340 | 0.178752 | **0.187498** |
| label6 | **0.983868** | 0.915088 | **0.953552** | 0.915088 | -0.038464 | 0.199700 | **0.255507** |

RHINO 对 `label1_thin`、`label3`、`label5` 的 mAP90 有提升，但 `label1_thick` 从 `0.643462` 降到 `0.082473`，是整体指标下降的主要来源。下一步应先检查 RHINO 对 `label1_thick` 的宽高、角度规范化和边界偏差，而不是继续以 mAP50 调参。

## 7. 当前模型选择结论

| 模型 | 当前状态 | 结论 |
| --- | --- | --- |
| YOLO11l-OBB | baseline / 当前首选 | 整体 mAP90 最好，部署和环境最简单 |
| RHINO R50-KLD epoch40 | 已完成 | 局部类别改善，但整体 mAP90 未超过 baseline |
| RHINO R50-KLD epoch50 | 待统一评估 | 排除 AP50 checkpoint 选择对 mAP90 的影响 |
| RHINO R50-RIoU | 尚未训练 | 仅在 KLD 分析后仍有价值时继续 |
| deskew / fusion | 已停止 | 双模型推理成本和工程复杂度过高 |

当前决策标准：

1. 以整体 `mAP90` 为主要排序指标。
2. 同时检查所有类别，避免某一类别严重退化。
3. mAP90 接近时，优先选择单模型、依赖更少、推理流程更短的方案。
4. mAP95 作为辅助指标，不单独决定模型选择。

## 8. 当前评估入口

YOLO11l 重新评估：

```bash
CUDA_VISIBLE_DEVICES=6 python3 scripts/evaluate_yolo11_obb.py \
  --data datasets/obb_thin_thick/data.yaml \
  --model runs/obb/yolo11l_after212102_no_index1_label1_thin_thick_e50_img1280_b8_deg0_valtest/weights/best.pt \
  --split test \
  --imgsz 1280 \
  --batch 8 \
  --device 0 \
  --project runs/obb \
  --name yolo11l_baseline_test_reeval
```

RHINO 已生成预测标签后的统一评估：

```bash
python3 scripts/evaluate_obb_prediction_labels.py \
  --data datasets/obb_thin_thick/data.yaml \
  --pred-labels runs/rhino/rhino_r50_kld_e50_img1280_b2/test_epoch40/labels \
  --split test \
  --output runs/rhino/rhino_r50_kld_e50_img1280_b2/test_epoch40/custom_metrics.csv
```

## 9. 文档归档

前期依赖安装、数据筛选、YOLO 调参、deskew/fusion、RHINO 路线和分类实验已按项目阶段拆分。统一入口见 [项目文档索引](docs/README.md)。迁移前的完整 README 仅作为只读快照保留，不再作为当前运行说明。

主 README 只维护当前数据口径、当前候选模型和可直接比较的统一评估结果。

## 10. 下游 ResNet18 分类

OBB 检测框会按类别裁剪后送入下游 ResNet18，目前分类目标为 `label3` 和 `label5` 的 OK/NG。分类 GT 来自 `outputs/label1_6_description.xlsx` 对应 sheet 的 `tag1`，不是与 OBB 框重新计算 IoU。

```text
OBB预测框
  -> 按 label3/label5 裁剪并透视矫正
  -> ResNet18
  -> OK/NG
  -> 与分类数据集 test split 的 tag1 比较
```

当前人工 GT 框裁剪下的分类结果：

| label | dataset | test distribution | accuracy | macro F1 |
| --- | --- | --- | ---: | ---: |
| label3 | `datasets/classification/label3_ok_ng` | NG=36, OK=17 | 0.981132 | 0.977999 |
| label5 | `datasets/classification/label5_ok_ng` | NG=35, OK=18 | 1.000000 | 1.000000 |

训练、独立评测和预测默认统一保存在：

```text
runs/classification/<run_name>/
  weights/best.pt
  weights/last.pt
  train_*.csv / train_args.yaml
  eval_*.csv / eval_args.yaml
  predict_*.csv / predict_args.yaml
```

上述分类结果基于人工 OBB 框裁剪，主要验证分类器能力；接入检测框后的端到端结果还会同时受到 OBB 定位误差影响。
