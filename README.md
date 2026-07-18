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

| 项目 | YOLO11l-OBB | RHINO R50-KLD | RHINO R50-RIoU |
| --- | --- | --- | --- |
| 架构 | CNN 单阶段 OBB | ResNet50 + Transformer | ResNet50 + Transformer |
| 输入尺寸 | 1280 | 1280 | 1280 |
| epochs | 50 | 50 | 50 |
| batch | 8 | 2 | 2 |
| 回归与匹配 | Ultralytics OBB 默认损失 | Hausdorff + KLD | Hausdorff + Rotated IoU |
| 训练阶段 best 依据 | Ultralytics validation | DOTA AP50 | DOTA AP50 |
| 部署形态 | 单模型、单环境 | 单模型，RHINO 环境 | 单模型，RHINO 环境 |

当前权重：

```text
YOLO11l:
runs/obb/yolo11l_after212102_no_index1_label1_thin_thick_e50_img1280_b8_deg0_valtest/weights/best.pt

RHINO R50-KLD:
runs/rhino/rhino_r50_kld_e50_img1280_b2/best_dota_mAP_epoch_40.pth
runs/rhino/rhino_r50_kld_e50_img1280_b2/epoch_50.pth

RHINO R50-RIoU:
runs/rhino/rhino_r50_riou_e50_img1280_b2/best_dota_mAP_epoch_*.pth
runs/rhino/rhino_r50_riou_e50_img1280_b2/epoch_50.pth
```

RHINO 文件名中的 `best_dota_mAP` 只表示训练阶段 AP50 最佳，不表示 mAP90 最佳。当前统一评估结果中，KLD 的 mAP90 最佳候选是 epoch40，RIoU 的 mAP90 最佳候选是 epoch50。

## 5. 整体结果对比

以下结果来自相同 test split 和相同 Ultralytics OBB 评估器。

### 5.1 KLD 与 baseline

| model | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11l-OBB baseline | 0.997159 | 0.999863 | 0.995000 | 0.962807 | 0.894190 | **0.711124** | 0.183112 |
| RHINO R50-KLD epoch40 | 0.991563 | 0.998104 | 0.994762 | 0.923683 | 0.830400 | 0.648533 | 0.211504 |
| RHINO R50-KLD epoch50 | 0.992294 | 0.994560 | 0.994921 | 0.927507 | 0.828269 | 0.623374 | **0.226096** |

KLD 的 AP50-best 是 epoch40，其 mAP90 比 epoch50 高 `0.025159`；epoch50 的 mAP95 高 `0.014592`，但没有改善主要目标 mAP90。两个 KLD checkpoint 的整体 mAP90 都低于 YOLO11l。

### 5.2 RIoU 与 baseline

| model | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11l-OBB baseline | 0.997159 | 0.999863 | 0.995000 | **0.962807** | **0.894190** | **0.711124** | 0.183112 |
| RHINO R50-RIoU AP50-best | 0.985410 | 0.995654 | 0.995000 | 0.911655 | 0.788412 | 0.596849 | 0.163323 |
| RHINO R50-RIoU epoch50 | 0.992466 | 0.991982 | 0.994948 | 0.959565 | 0.878536 | 0.659244 | **0.264438** |

RIoU epoch50 比其 AP50-best 的 mAP90 高 `0.062395`、mAP95 高 `0.101115`，是当前表现最好的 RHINO checkpoint。与 YOLO11l 相比，RIoU epoch50 的 mAP90 仍低 `0.051880`，但 mAP95 高 `0.081326`。

综合结论：

- 所有模型的 mAP50 几乎一致，主要差异来自旋转框定位精度。
- RIoU epoch50 是当前 RHINO 最佳结果，mAP90 为 `0.659244`，比 KLD epoch40 高 `0.010711`，比 KLD epoch50 高 `0.035870`。
- KLD 的 AP50-best 在已评估的两个 checkpoint 中同时取得更高 mAP90，但 RIoU 的结果证明这种对应关系并不稳定，后续不能只根据 `best_dota_mAP` 文件名选择 checkpoint。
- 当前模型选择仍保留 YOLO11l-OBB 作为 baseline 和工程首选。

## 6. 分类别 mAP90 对比

### 6.1 KLD 与 baseline

| class | YOLO11l | KLD AP50-best e40 | KLD e50 |
| --- | ---: | ---: | ---: |
| label1_thin | 0.335726 | **0.446060** | 0.413432 |
| label1_thick | **0.643462** | 0.082473 | 0.114522 |
| label2 | **0.816797** | 0.732371 | 0.671747 |
| label3 | 0.586077 | **0.688773** | 0.587144 |
| label4 | **0.967453** | 0.953826 | 0.929727 |
| label5 | 0.674799 | 0.721139 | **0.725400** |
| label6 | **0.953552** | 0.915088 | 0.921647 |

KLD epoch40 对 `label1_thin` 和 `label3` 有明显提升，但 `label1_thick` 严重退化；epoch50 没有修复该问题。

### 6.2 RIoU 与 baseline

| class | YOLO11l | RIoU AP50-best | RIoU e50 |
| --- | ---: | ---: | ---: |
| label1_thin | **0.335726** | 0.272419 | 0.280394 |
| label1_thick | **0.643462** | 0.104691 | 0.300384 |
| label2 | **0.816797** | 0.605898 | 0.719197 |
| label3 | 0.586077 | 0.549698 | **0.639405** |
| label4 | 0.967453 | **0.973491** | 0.966698 |
| label5 | 0.674799 | 0.735962 | **0.779512** |
| label6 | **0.953552** | 0.935784 | 0.929118 |

RIoU epoch50 显著修复了 AP50-best 在 `label1_thick` 上的退化，并提高了 `label3` 和 `label5`；但 `label1_thick`、`label2` 和 `label6` 仍低于 YOLO11l。

## 7. 当前模型选择结论

| 模型 | 当前状态 | 结论 |
| --- | --- | --- |
| YOLO11l-OBB | baseline / 当前首选 | 整体 mAP90 最好，部署和环境最简单 |
| RHINO R50-KLD epoch40 | KLD 当前代表 | mAP90 为 0.648533，优于 KLD epoch50 |
| RHINO R50-KLD epoch50 | 已完成 | mAP95 提高，但 mAP90 降至 0.623374 |
| RHINO R50-RIoU AP50-best | 已完成 | AP50-best 与高 IoU 目标不一致 |
| RHINO R50-RIoU epoch50 | RHINO 当前代表 | RHINO 中整体 mAP90/mAP95 最好，但仍未超过 YOLO mAP90 |
| Oriented R-CNN R50-FPN | 待训练 | 新增两阶段旋转框精修对照，结果仍需统一 mAP90 评测 |
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

Oriented R-CNN R50-FPN 先生成配置并检查命令：

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_oriented_rcnn.py \
  --rhino-root ~/RHINO \
  --data datasets/rhino_obb \
  --imgsz 1280 \
  --epochs 50 \
  --batch 2 \
  --project runs/oriented_rcnn \
  --name oriented_rcnn_r50_fpn_e50_img1280_b2 \
  --dry-run
```

完整训练、预测转换和统一 mAP90 评测命令见 [代码架构文档](docs/architecture.md)。顶层脚本提供简洁命令，具体实现按 `scripts/models/<model>/` 分类；原有脚本和 `yolo11_obb` import 暂时保留兼容，不影响历史训练与权重。

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
