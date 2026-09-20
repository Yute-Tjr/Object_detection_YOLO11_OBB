# OBB 目标检测：模型选择与统一评估

本项目当前聚焦旋转目标框检测的模型选择，核心目标是提升严格 IoU 阈值下的定位精度，`mAP85` 和 `mAP90` 是两个主要指标，`mAP95` 仅作为极严格定位的辅助诊断。当前主线比较 YOLO11l/YOLO26l/YOLO26m-OBB、Transformer 架构 RHINO 和两阶段 Oriented R-CNN R50-FPN，不再采用 deskew 或双模型 fusion 作为工程方案。

## 1. 当前检测流程

```text
原始图像
  -> OBB 检测模型（YOLO11l、YOLO26l/m、RHINO 或 Oriented R-CNN）
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

YOLO11l、YOLO26l/m、RHINO 和 Oriented R-CNN 使用相同的样本、类别与 train/test 划分：

| 用途 | 路径 | 图像格式 | 标注格式 |
| --- | --- | --- | --- |
| YOLO11l | `datasets/obb_thin_thick` | BMP | YOLO OBB 归一化四点 |
| YOLO26l/m | `datasets/obb_thin_thick` | BMP | YOLO OBB 归一化四点 |
| RHINO | `datasets/rhino_obb` | PNG | DOTA 四点 annfile |
| Oriented R-CNN | `datasets/rhino_obb` | PNG | DOTA 四点 annfile |

MMRotate 数据集由 YOLO 主数据集转换而来，供 RHINO 和 Oriented R-CNN 共用；转换只改变图像与标注的存储格式，不改变图像内容和划分。

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

该评估器与具体检测模型无关。YOLO11l 和 YOLO26l/m 可直接输出 OBB；RHINO 和 Oriented R-CNN 先把 MMRotate 预测转换为四点 OBB，再进入相同评估器。

```text
YOLO best/last.pt
  -> YOLO OBB 预测
  -> custom_metrics.csv

RHINO best.pth
  -> predictions.pkl
  -> labels/*.txt
  -> custom_metrics.csv

Oriented R-CNN epoch_*.pth
  -> predictions.pkl
  -> labels/*.txt
  -> custom_metrics.csv
```

MMRotate 训练期间显示的 `dota/mAP` 仅为 polygon IoU 下的 AP50，用于训练监控和 checkpoint 选择，不能直接与最终的 mAP80-95 混用。

## 4. 模型与训练配置

| 项目 | YOLO11l-OBB | YOLO26l/m-OBB | RHINO R50-KLD | RHINO R50-RIoU | Oriented R-CNN R50-FPN |
| --- | --- | --- | --- | --- | --- |
| 架构 | CNN 单阶段 OBB | YOLO26 单阶段 OBB（l/m 容量消融） | ResNet50 + Transformer | ResNet50 + Transformer | ResNet50-FPN + RPN + Rotated RoI Head |
| 输入尺寸 | 1280 | 1280 | 1280 | 1280 | 1280 |
| epochs | 50 | 50 | 50 | 50 | 50 |
| batch | 8 | 8 | 2、4（消融） | 2、4（消融） | 2、4（消融） |
| 回归与匹配 | Ultralytics OBB 默认损失 | Ultralytics YOLO26 OBB 默认损失 | Hausdorff + KLD | Hausdorff + Rotated IoU | 两阶段 proposal + rotated box refinement |
| 训练阶段 best 依据 | Ultralytics validation | Ultralytics validation | DOTA AP50 | DOTA AP50 | DOTA AP50 |
| 部署形态 | 单模型、单环境 | 单模型、单环境 | 单模型，RHINO 环境 | 单模型，RHINO 环境 | 单模型，MMRotate/RHINO 环境 |

当前权重：

```text
YOLO11l:
runs/obb/yolo11l_after212102_no_index1_label1_thin_thick_e50_img1280_b8_deg0_valtest/weights/best.pt

YOLO26l:
runs/obb/yolo26l_after212102_no_index1_label1_thin_thick_e50_img1280_b8_deg0_valtest/weights/best.pt
runs/obb/yolo26l_after212102_no_index1_label1_thin_thick_e50_img1280_b8_deg0_valtest/weights/last.pt

YOLO26m:
runs/obb/yolo26m_after212102_no_index1_label1_thin_thick_e50_img1280_b8_deg0_valtest/weights/best.pt
runs/obb/yolo26m_after212102_no_index1_label1_thin_thick_e50_img1280_b8_deg0_valtest/weights/last.pt

RHINO R50-KLD:
runs/rhino/rhino_r50_kld_e50_img1280_b2/best_dota_mAP_epoch_40.pth
runs/rhino/rhino_r50_kld_e50_img1280_b2/epoch_50.pth
runs/rhino/rhino_r50_kld_e50_img1280_b4/best_dota_mAP_epoch_*.pth
runs/rhino/rhino_r50_kld_e50_img1280_b4/epoch_50.pth

RHINO R50-RIoU:
runs/rhino/rhino_r50_riou_e50_img1280_b2/best_dota_mAP_epoch_*.pth
runs/rhino/rhino_r50_riou_e50_img1280_b2/epoch_50.pth
runs/rhino/rhino_r50_riou_e50_img1280_b4/best_dota_mAP_epoch_*.pth
runs/rhino/rhino_r50_riou_e50_img1280_b4/epoch_50.pth

Oriented R-CNN R50-FPN:
runs/oriented_rcnn/oriented_rcnn_r50_fpn_e50_img1280_b2/best_dota_mAP_epoch_6.pth
runs/oriented_rcnn/oriented_rcnn_r50_fpn_e50_img1280_b2/epoch_50.pth
runs/oriented_rcnn/oriented_rcnn_r50_fpn_e50_img1280_b4/best_dota_mAP_epoch_*.pth
runs/oriented_rcnn/oriented_rcnn_r50_fpn_e50_img1280_b4/epoch_50.pth
```

文件名中的 `best_dota_mAP` 只表示训练阶段 AP50 最佳，不表示 mAP85/mAP90 最佳。batch4 统一评估中，KLD 和 Oriented R-CNN 的 mAP90 最佳候选是 epoch50，RIoU 的 mAP90 最佳候选则是 AP50-best；YOLO26l 的 best 双指标优于 last，而 YOLO26m 的 last 双指标等权分数略高于 best。后续每次训练都应至少同时评估训练框架选出的 best 和最后一个 checkpoint。

## 5. 整体结果对比

以下结果来自相同 test split 和相同 Ultralytics OBB 评估器。

### 5.1 KLD batch2 与 baseline

| model | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11l-OBB baseline | 0.997159 | 0.999863 | 0.995000 | 0.962807 | 0.894190 | **0.711124** | 0.183112 |
| RHINO R50-KLD epoch40 | 0.991563 | 0.998104 | 0.994762 | 0.923683 | 0.830400 | 0.648533 | 0.211504 |
| RHINO R50-KLD epoch50 | 0.992294 | 0.994560 | 0.994921 | 0.927507 | 0.828269 | 0.623374 | **0.226096** |

KLD 的 AP50-best 是 epoch40，其 mAP90 比 epoch50 高 `0.025159`；epoch50 的 mAP95 高 `0.014592`，但没有改善主要目标 mAP90。两个 KLD checkpoint 的整体 mAP90 都低于 YOLO11l。

### 5.2 RIoU batch2 与 baseline

| model | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11l-OBB baseline | 0.997159 | 0.999863 | 0.995000 | **0.962807** | **0.894190** | **0.711124** | 0.183112 |
| RHINO R50-RIoU AP50-best | 0.985410 | 0.995654 | 0.995000 | 0.911655 | 0.788412 | 0.596849 | 0.163323 |
| RHINO R50-RIoU epoch50 | 0.992466 | 0.991982 | 0.994948 | 0.959565 | 0.878536 | 0.659244 | **0.264438** |

RIoU epoch50 比其 AP50-best 的 mAP90 高 `0.062395`、mAP95 高 `0.101115`，是 batch2 中表现最好的 RHINO checkpoint。与 YOLO11l 相比，RIoU epoch50 的 mAP90 仍低 `0.051880`，但 mAP95 高 `0.081326`。

### 5.3 Oriented R-CNN batch2 与 baseline

| model | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11l-OBB baseline | 0.997159 | **0.999863** | **0.995000** | **0.962807** | **0.894190** | **0.711124** | **0.183112** |
| Oriented R-CNN AP50-best epoch6 | 0.997468 | 0.996280 | **0.995000** | 0.892713 | 0.745466 | 0.415088 | 0.033228 |
| Oriented R-CNN epoch50 | **0.998536** | 0.996358 | **0.995000** | 0.933204 | 0.888799 | 0.686510 | 0.182745 |

Oriented R-CNN epoch50 比 AP50-best epoch6 的 mAP85 高 `0.143333`、mAP90 高 `0.271422`、mAP95 高 `0.149517`，说明 AP50-best checkpoint 完全不适合作为该模型的高 IoU 代表。epoch50 的 mAP90 距 YOLO11l 仅 `0.024614`，并比 RHINO R50-RIoU epoch50 高 `0.027266`；其 mAP95 与 YOLO11l 只差 `0.000367`。

### 5.4 batch4 完整结果

| model | checkpoint | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RHINO R50-KLD b4 | AP50-best | 0.990190 | 0.996487 | 0.994974 | 0.933258 | 0.845754 | 0.649115 | **0.207403** |
| RHINO R50-KLD b4 | epoch50 | 0.996755 | 0.995905 | 0.995000 | **0.939198** | **0.856959** | **0.685661** | 0.198077 |
| RHINO R50-RIoU b4 | AP50-best | 0.965274 | 0.979237 | 0.989985 | **0.938151** | **0.872691** | **0.662231** | 0.213762 |
| RHINO R50-RIoU b4 | epoch50 | 0.961146 | 0.973812 | 0.987300 | 0.932056 | 0.861031 | 0.655601 | **0.231351** |
| Oriented R-CNN b4 | AP50-best | 0.997165 | 0.995740 | 0.995000 | 0.912147 | 0.794084 | 0.532095 | 0.075518 |
| Oriented R-CNN b4 | epoch50 | **0.998666** | **0.996384** | **0.995000** | **0.931711** | **0.861925** | **0.689797** | **0.208225** |

表内 mAP80-95 的加粗值表示同一模型 batch4 的两个 checkpoint 中更好的结果。若单独按 mAP90 选择，batch4 的模型代表分别为 KLD epoch50、RIoU AP50-best 和 Oriented R-CNN epoch50；双指标选型见 5.7 节。

### 5.5 batch2/batch4 消融

| model / checkpoint | b2 mAP85 | b4 mAP85 | mAP85 变化 | b2 mAP90 | b4 mAP90 | mAP90 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KLD AP50-best | 0.830400 | 0.845754 | +0.015354 | 0.648533 | 0.649115 | +0.000582 |
| KLD epoch50 | 0.828269 | 0.856959 | **+0.028690** | 0.623374 | 0.685661 | **+0.062287** |
| RIoU AP50-best | 0.788412 | 0.872691 | **+0.084279** | 0.596849 | 0.662231 | **+0.065382** |
| RIoU epoch50 | 0.878536 | 0.861031 | -0.017505 | 0.659244 | 0.655601 | -0.003643 |
| Oriented R-CNN AP50-best | 0.745466 | 0.794084 | **+0.048618** | 0.415088 | 0.532095 | **+0.117007** |
| Oriented R-CNN epoch50 | 0.888799 | 0.861925 | -0.026874 | 0.686510 | 0.689797 | +0.003287 |

若不绑定 checkpoint 名称，而是取每个 batch 实验中双指标等权分数最高的权重，结果更能反映扩大 batch 对最终模型选择的真实价值：

| model | batch2 最佳双指标分数 | batch4 最佳双指标分数 | 变化 |
| --- | ---: | ---: | ---: |
| RHINO R50-KLD | 0.739467 | 0.771310 | **+0.031844** |
| RHINO R50-RIoU | 0.768890 | 0.767461 | -0.001429 |
| Oriented R-CNN | 0.787655 | 0.775861 | -0.011794 |

扩大 batch 并未让三个模型一致受益。它明确改善了 KLD；RIoU 的最佳双指标分数基本不变，Oriented R-CNN 则因为 mAP85 的明显损失而下降。因此当前没有直接扩大到 batch8 的依据。

### 5.6 YOLO26l/m 与 YOLO11l baseline

YOLO26l、YOLO26m 与 YOLO11l 使用同一数据集、输入尺寸 1280、50 epoch、batch8、seed42 和 `degrees=0`。两个 YOLO26 容量版本的 best 与 last 统一评估结果如下：

| model | checkpoint | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11l | best | **0.997159** | **0.999863** | **0.995000** | **0.962807** | **0.894190** | **0.711124** | 0.183112 |
| YOLO26l | best | 0.983001 | 0.979877 | 0.992925 | 0.952319 | 0.869726 | 0.676807 | 0.229832 |
| YOLO26l | last | 0.971175 | 0.981281 | 0.990271 | 0.941601 | 0.862519 | 0.659412 | 0.237696 |
| YOLO26m | best | 0.989232 | 0.986867 | 0.994363 | 0.927486 | 0.861608 | 0.689270 | 0.256848 |
| YOLO26m | last | 0.993045 | 0.980019 | 0.993815 | 0.936864 | 0.851910 | 0.703123 | **0.260011** |

YOLO26l best 相比 last 的 mAP85 高 `0.007207`、mAP90 高 `0.017395`，因此 l 版本应选择 best。YOLO26m 则呈现权衡：last 的 mAP85 比 best 低 `0.009698`，但 mAP90 高 `0.013853`，双指标等权分数从 `0.775439` 小幅升至 `0.777517`，因此在等权规则下选择 last。YOLO26m last 的 mAP90 距 YOLO11l 仅 `0.008001`，但 mAP85 仍低 `0.042280`，没有全面刷新 baseline。

容量从 l 降到 m 后，所选 checkpoint 的双指标等权分数从 `0.773267` 提高到 `0.777517`，增加 `0.004250`。这说明较小容量对当前数据有一定收益，但提升来自 mAP90、同时伴随 mAP85 下降，不能简单归因于 YOLO26l 过拟合；更准确的结论是两个容量版本产生了不同的高 IoU 定位权衡。

### 5.7 当前整体排序与结论

当前同时关注 mAP85 和 mAP90。为便于排序，表中的双指标等权分数定义为 `(mAP85 + mAP90) / 2`，并为每个模型选择该分数最高的已评估 checkpoint：

| rank | model | checkpoint | mAP85 | mAP90 | 双指标等权分数 | mAP95 |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | YOLO11l-OBB baseline | best | **0.894190** | **0.711124** | **0.802657** | 0.183112 |
| 2 | Oriented R-CNN | b2 epoch50 | 0.888799 | 0.686510 | 0.787655 | 0.182745 |
| 3 | YOLO26m-OBB | last | 0.851910 | 0.703123 | 0.777517 | 0.260011 |
| 4 | YOLO26l-OBB | best | 0.869726 | 0.676807 | 0.773267 | 0.229832 |
| 5 | RHINO R50-KLD | b4 epoch50 | 0.856959 | 0.685661 | 0.771310 | 0.198077 |
| 6 | RHINO R50-RIoU | b2 epoch50 | 0.878536 | 0.659244 | 0.768890 | **0.264438** |

综合结论：

- 所有模型的 mAP50 几乎一致，主要差异来自旋转框定位精度。
- YOLO11l-OBB 在 mAP85、mAP90 和双指标等权分数上均保持第一，仍是工程首选。
- Oriented R-CNN batch2 epoch50 是当前最均衡的 MMRotate 对照；batch4 只增加 `0.003287` mAP90，却损失 `0.026874` mAP85，因此双指标下不如 batch2。
- YOLO26m last 的 mAP90 为 `0.703123`，是除 YOLO11l 外最高的 mAP90，距 baseline 仅 `0.008001`；但 mAP85 只有 `0.851910`，双指标等权后仍低于 Oriented R-CNN batch2 epoch50。
- YOLO26m 的双指标等权分数比 YOLO26l 提高 `0.004250`，说明缩小容量有小幅综合收益，但 mAP85/mAP90 的反向变化不支持把 YOLO26l 的结果简单归因于过拟合。
- KLD batch4 epoch50 同时改善了 KLD 自身的 mAP85 和 mAP90，但两个指标仍分别低于 Oriented R-CNN batch4 epoch50，当前不作为最终首选。
- RIoU batch2 epoch50 比 batch4 更均衡，并保留所有实验中最高的 mAP95；但 mAP95 不作为当前主要排序依据。
- batch4 提升具有模型和 checkpoint 依赖性，并不支持直接扩大 batch 就一定改善双指标的结论。
- Oriented R-CNN 的 AP50-best 与 epoch50 仍存在巨大差异，进一步证明后续不能只根据 `best_dota_mAP` 文件名选择 checkpoint。
- 当前模型选择保留 YOLO11l-OBB 作为工程首选、Oriented R-CNN batch2 epoch50 作为最强均衡对照。

## 6. 分类别 mAP85/mAP90 对比

### 6.1 KLD batch2 与 baseline

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

### 6.2 RIoU batch2 与 baseline

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

### 6.3 Oriented R-CNN batch2 与 baseline

| class | YOLO11l | Oriented R-CNN AP50-best e6 | Oriented R-CNN e50 |
| --- | ---: | ---: | ---: |
| label1_thin | 0.335726 | 0.209263 | **0.495807** |
| label1_thick | **0.643462** | 0.206333 | 0.200833 |
| label2 | 0.816797 | 0.324057 | **0.843347** |
| label3 | 0.586077 | 0.230687 | **0.659070** |
| label4 | **0.967453** | 0.906104 | 0.943856 |
| label5 | 0.674799 | 0.667775 | **0.729621** |
| label6 | **0.953552** | 0.361399 | 0.933039 |

Oriented R-CNN epoch50 在 `label1_thin`、`label2`、`label3` 和 `label5` 上超过 YOLO11l，但 `label1_thick` 的 mAP90 只有 `0.200833`，比 YOLO11l 低 `0.442629`，是当前最主要的类别短板。

### 6.4 batch4 最佳 checkpoint 分类别 mAP90

| class | YOLO11l | KLD b4 e50 | RIoU b4 AP50-best | Oriented R-CNN b4 e50 |
| --- | ---: | ---: | ---: | ---: |
| label1_thin | 0.335726 | 0.382594 | 0.388212 | **0.392427** |
| label1_thick | **0.643462** | 0.340332 | 0.204502 | 0.235543 |
| label2 | 0.816797 | **0.818590** | 0.755904 | 0.778780 |
| label3 | 0.586077 | 0.587596 | 0.625475 | **0.742877** |
| label4 | 0.967453 | 0.930790 | 0.939248 | **0.969906** |
| label5 | 0.674799 | **0.816540** | 0.795844 | 0.779928 |
| label6 | **0.953552** | 0.923182 | 0.926431 | 0.929118 |

batch4 后，三个 MMRotate 模型在 `label1_thin` 和 `label5` 上都超过 YOLO11l；KLD 在 `label2` 上基本追平并略微超过 YOLO11l，Oriented R-CNN 在 `label3` 和 `label4` 上最好。共同瓶颈仍是 `label1_thick`：三者 mAP90 只有 `0.204502` 至 `0.340332`，显著低于 YOLO11l 的 `0.643462`。由于整体 mAP90 是 7 类宏平均，若仅把 KLD batch4 epoch50 的 `label1_thick` 提升到 YOLO11l 水平、其他类别保持不变，整体 mAP90 将从 `0.685661` 算术上升到约 `0.728965`；这说明该类足以改变最终模型排序。该类训练集只有 66 个实例、测试集只有 18 个实例，严格 IoU 指标波动也会更大；在继续更换模型或扩大 batch 前，应优先检查该类标注一致性并补充样本。

### 6.5 YOLO26l best/last 分类别对比

| class | YOLO11l mAP85 | YOLO11l mAP90 | YOLO26l best mAP85 | YOLO26l best mAP90 | YOLO26l last mAP85 | YOLO26l last mAP90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| label1_thin | **0.649216** | **0.335726** | 0.474323 | 0.195302 | 0.542392 | 0.235182 |
| label1_thick | 0.775000 | **0.643462** | **0.776538** | 0.428417 | 0.727333 | 0.246667 |
| label2 | 0.970283 | **0.816797** | **0.977453** | 0.776892 | 0.942568 | 0.795240 |
| label3 | 0.943141 | 0.586077 | **0.945484** | **0.751112** | 0.942554 | 0.717438 |
| label4 | **0.995000** | **0.967453** | **0.995000** | 0.938585 | **0.995000** | 0.956564 |
| label5 | 0.942822 | 0.674799 | **0.944663** | 0.700053 | 0.942736 | **0.742846** |
| label6 | **0.983868** | **0.953552** | 0.974623 | 0.947290 | 0.945051 | 0.921945 |

YOLO26l best 的主要损失来自 `label1`：相比 YOLO11l，`label1_thin` 的 mAP85/mAP90 分别下降 `0.174893/0.140424`，`label1_thick` 的 mAP90 下降 `0.215045`。与此同时，YOLO26l best 将 `label3` mAP90 从 `0.586077` 提高到 `0.751112`，并小幅提高 `label5` mAP90。last 并非所有类别都更差，但 `label1_thick`、`label3` 和 `label6` 的退化使整体 mAP85/mAP90 低于 best。

### 6.6 YOLO26m best/last 分类别对比

| class | YOLO11l mAP85 | YOLO11l mAP90 | YOLO26m best mAP85 | YOLO26m best mAP90 | YOLO26m last mAP85 | YOLO26m last mAP90 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| label1_thin | 0.649216 | 0.335726 | 0.600858 | 0.465265 | **0.661834** | **0.467542** |
| label1_thick | **0.775000** | **0.643462** | 0.647083 | 0.167500 | 0.557441 | 0.280265 |
| label2 | **0.970283** | **0.816797** | 0.930660 | 0.790577 | 0.932170 | 0.778848 |
| label3 | 0.943141 | 0.586077 | 0.949748 | **0.783581** | **0.949794** | 0.764147 |
| label4 | **0.995000** | 0.967453 | 0.994259 | 0.933394 | 0.991187 | **0.968868** |
| label5 | **0.942822** | 0.674799 | 0.938746 | **0.746923** | 0.901981 | 0.726153 |
| label6 | **0.983868** | **0.953552** | 0.969906 | 0.937648 | 0.968962 | 0.936035 |

YOLO26m 明显改变了 `label1` 内部的定位分布：last 的 `label1_thin` 同时超过 YOLO11l 的 mAP85/mAP90，但 `label1_thick` mAP90 只有 `0.280265`，比 YOLO11l 低 `0.363197`，仍是最主要瓶颈。YOLO26m 还显著提高了 `label3` 和 `label5` 的 mAP90。best 与 last 之间的整体权衡主要来自 last 改善 `label1_thin`、`label1_thick` 和 `label4` mAP90，同时损失多个类别的 mAP85。

## 7. 当前模型选择结论

| 模型 | 当前状态 | 结论 |
| --- | --- | --- |
| YOLO11l-OBB | baseline / 当前首选 | mAP85=0.894190、mAP90=0.711124，两个主要指标均为当前最佳 |
| YOLO26m-OBB last | YOLO26 当前代表 | mAP85=0.851910、mAP90=0.703123；mAP90 接近 baseline，但 mAP85 明显较低 |
| YOLO26l-OBB best | 已完成 / 容量对照 | 双指标等权分数 0.773267，低于 YOLO26m 的 0.777517 |
| RHINO R50-KLD b4 epoch50 | KLD 当前代表 | mAP85=0.856959、mAP90=0.685661；扩大 batch 对 KLD 有效，但综合仍非首选 |
| RHINO R50-RIoU b2 epoch50 | RIoU 均衡代表 | mAP85=0.878536、mAP90=0.659244，双指标比 b4 更均衡 |
| Oriented R-CNN b2 epoch50 | MMRotate 均衡代表 | mAP85=0.888799、mAP90=0.686510，当前最强非 YOLO11l 对照 |
| batch4 三组实验 | 已完成 / batch 消融 | KLD 收益明显；RIoU 和 Oriented R-CNN 的双指标收益不足 |
| deskew / fusion | 已停止 | 双模型推理成本和工程复杂度过高 |

当前决策标准：

1. `mAP85` 和 `mAP90` 是两个同等重要的主要指标；需要单值排序时使用二者等权平均。
2. 同时检查所有类别，避免整体平均掩盖 `label1_thin/label1_thick` 等类别的严重退化。
3. 双指标接近时，优先选择单模型、依赖更少、推理流程更短的方案。
4. `mAP95` 仅作为极严格定位的辅助诊断，不单独决定模型选择。

## 8. 各模型训练与 checkpoint 完整分类别结果

以下各表均来自相同 test split 和统一 Ultralytics OBB 评估器，并按模型、batch、checkpoint 分层保留，不再只展示代表权重。`all` 是 7 个类别的总体结果，其余各行保留每个标签的 precision、recall 与完整高 IoU 指标，便于定位整体平均背后的类别差异。

### 8.1 YOLO11l-OBB baseline best

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.997159 | 0.999863 | 0.995000 | 0.962807 | 0.894190 | 0.711124 | 0.183112 |
| label1_thin | 1.000000 | 0.999041 | 0.995000 | 0.940558 | 0.649216 | 0.335726 | 0.014286 |
| label1_thick | 0.980111 | 1.000000 | 0.995000 | 0.875694 | 0.775000 | 0.643462 | 0.000000 |
| label2 | 1.000000 | 1.000000 | 0.995000 | 0.995000 | 0.970283 | 0.816797 | 0.264728 |
| label3 | 1.000000 | 1.000000 | 0.995000 | 0.983113 | 0.943141 | 0.586077 | 0.131533 |
| label4 | 1.000000 | 1.000000 | 0.995000 | 0.995000 | 0.995000 | 0.967453 | 0.492787 |
| label5 | 1.000000 | 1.000000 | 0.995000 | 0.966415 | 0.942822 | 0.674799 | 0.178752 |
| label6 | 1.000000 | 1.000000 | 0.995000 | 0.983868 | 0.983868 | 0.953552 | 0.199700 |

### 8.2 YOLO26l-OBB best/last

#### best

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.983001 | 0.979877 | 0.992925 | 0.952319 | 0.869726 | 0.676807 | 0.229832 |
| label1_thin | 0.969873 | 0.919918 | 0.982470 | 0.908569 | 0.474323 | 0.195302 | 0.001190 |
| label1_thick | 0.971692 | 1.000000 | 0.995000 | 0.842647 | 0.776538 | 0.428417 | 0.018333 |
| label2 | 0.984520 | 1.000000 | 0.995000 | 0.995000 | 0.977453 | 0.776892 | 0.235577 |
| label3 | 0.997005 | 1.000000 | 0.995000 | 0.974623 | 0.945484 | 0.751112 | 0.211187 |
| label4 | 0.988155 | 1.000000 | 0.995000 | 0.995000 | 0.995000 | 0.938585 | 0.577189 |
| label5 | 1.000000 | 0.939224 | 0.993008 | 0.975774 | 0.944663 | 0.700053 | 0.225116 |
| label6 | 0.969765 | 1.000000 | 0.995000 | 0.974623 | 0.974623 | 0.947290 | 0.340232 |

#### last

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.971175 | 0.981281 | 0.990271 | 0.941601 | 0.862519 | 0.659412 | 0.237696 |
| label1_thin | 0.969446 | 0.906701 | 0.965149 | 0.790351 | 0.542392 | 0.235182 | 0.011965 |
| label1_thick | 0.945403 | 1.000000 | 0.995000 | 0.895556 | 0.727333 | 0.246667 | 0.035357 |
| label2 | 0.969855 | 1.000000 | 0.995000 | 0.995000 | 0.942568 | 0.795240 | 0.315793 |
| label3 | 0.937744 | 0.962264 | 0.991748 | 0.967942 | 0.942554 | 0.717438 | 0.145346 |
| label4 | 0.996322 | 1.000000 | 0.995000 | 0.995000 | 0.995000 | 0.956564 | 0.537849 |
| label5 | 0.990735 | 1.000000 | 0.995000 | 0.973491 | 0.942736 | 0.742846 | 0.175577 |
| label6 | 0.988721 | 1.000000 | 0.995000 | 0.973868 | 0.945051 | 0.921945 | 0.441988 |

### 8.3 YOLO26m-OBB best/last

#### best

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.989232 | 0.986867 | 0.994363 | 0.927486 | 0.861608 | 0.689270 | 0.256848 |
| label1_thin | 0.991008 | 0.942857 | 0.991282 | 0.790274 | 0.600858 | 0.465265 | 0.038333 |
| label1_thick | 1.000000 | 0.987314 | 0.995000 | 0.792778 | 0.647083 | 0.167500 | 0.000000 |
| label2 | 0.983757 | 1.000000 | 0.995000 | 0.995000 | 0.930660 | 0.790577 | 0.313961 |
| label3 | 0.997197 | 1.000000 | 0.995000 | 0.974245 | 0.949748 | 0.783581 | 0.126612 |
| label4 | 0.963110 | 0.985242 | 0.994259 | 0.994259 | 0.994259 | 0.933394 | 0.699352 |
| label5 | 1.000000 | 0.992655 | 0.995000 | 0.975943 | 0.938746 | 0.746923 | 0.310427 |
| label6 | 0.989550 | 1.000000 | 0.995000 | 0.969906 | 0.969906 | 0.937648 | 0.309250 |

#### last

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.993045 | 0.980019 | 0.993815 | 0.936864 | 0.851910 | 0.703123 | 0.260011 |
| label1_thin | 1.000000 | 0.936053 | 0.990516 | 0.849041 | 0.661834 | 0.467542 | 0.038333 |
| label1_thick | 0.988043 | 1.000000 | 0.995000 | 0.803667 | 0.557441 | 0.280265 | 0.006875 |
| label2 | 0.997356 | 1.000000 | 0.995000 | 0.995000 | 0.932170 | 0.778848 | 0.339574 |
| label3 | 0.976395 | 1.000000 | 0.995000 | 0.973868 | 0.949794 | 0.764147 | 0.148950 |
| label4 | 1.000000 | 0.924076 | 0.991187 | 0.991187 | 0.991187 | 0.968868 | 0.659465 |
| label5 | 0.994112 | 1.000000 | 0.995000 | 0.976321 | 0.901981 | 0.726153 | 0.303995 |
| label6 | 0.995413 | 1.000000 | 0.995000 | 0.968962 | 0.968962 | 0.936035 | 0.322888 |

### 8.4 RHINO R50-KLD batch2/batch4

#### batch2 AP50-best epoch40

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.991563 | 0.998104 | 0.994762 | 0.923683 | 0.830400 | 0.648533 | 0.211504 |
| label1_thin | 1.000000 | 0.991081 | 0.995000 | 0.841769 | 0.690368 | 0.446060 | 0.045196 |
| label1_thick | 0.977986 | 1.000000 | 0.995000 | 0.740154 | 0.445073 | 0.082473 | 0.005225 |
| label2 | 0.994286 | 1.000000 | 0.995000 | 0.995000 | 0.905102 | 0.732371 | 0.174382 |
| label3 | 0.993530 | 1.000000 | 0.995000 | 0.942571 | 0.942571 | 0.688773 | 0.120125 |
| label4 | 1.000000 | 0.995648 | 0.995000 | 0.995000 | 0.995000 | 0.953826 | 0.692595 |
| label5 | 0.998487 | 1.000000 | 0.995000 | 0.973491 | 0.919599 | 0.721139 | 0.187498 |
| label6 | 0.976653 | 1.000000 | 0.993333 | 0.977795 | 0.915088 | 0.915088 | 0.255507 |

#### batch2 epoch50

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.992294 | 0.994560 | 0.994921 | 0.927507 | 0.828269 | 0.623374 | 0.226096 |
| label1_thin | 1.000000 | 0.974865 | 0.995000 | 0.832805 | 0.678232 | 0.413432 | 0.017351 |
| label1_thick | 0.976145 | 1.000000 | 0.995000 | 0.772286 | 0.432337 | 0.114522 | 0.009461 |
| label2 | 0.996283 | 1.000000 | 0.995000 | 0.995000 | 0.899794 | 0.671747 | 0.213464 |
| label3 | 0.996436 | 1.000000 | 0.995000 | 0.941917 | 0.941917 | 0.587144 | 0.098690 |
| label4 | 1.000000 | 0.997258 | 0.995000 | 0.995000 | 0.995000 | 0.929727 | 0.704876 |
| label5 | 0.995902 | 1.000000 | 0.995000 | 0.975359 | 0.928960 | 0.725400 | 0.183996 |
| label6 | 0.981293 | 0.989801 | 0.994444 | 0.980185 | 0.921647 | 0.921647 | 0.354835 |

#### batch4 AP50-best

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.990190 | 0.996487 | 0.994974 | 0.933258 | 0.845754 | 0.649115 | 0.207403 |
| label1_thin | 0.999155 | 1.000000 | 0.995000 | 0.864784 | 0.587612 | 0.421472 | 0.005418 |
| label1_thick | 0.993082 | 1.000000 | 0.995000 | 0.775840 | 0.623392 | 0.166506 | 0.036800 |
| label2 | 0.998182 | 1.000000 | 0.995000 | 0.995000 | 0.956878 | 0.801760 | 0.290458 |
| label3 | 0.998331 | 1.000000 | 0.995000 | 0.968208 | 0.901294 | 0.577447 | 0.109397 |
| label4 | 1.000000 | 0.975408 | 0.995000 | 0.995000 | 0.995000 | 0.919822 | 0.473049 |
| label5 | 0.962645 | 1.000000 | 0.995000 | 0.967095 | 0.929460 | 0.730404 | 0.239902 |
| label6 | 0.979936 | 1.000000 | 0.994815 | 0.966883 | 0.926642 | 0.926391 | 0.296797 |

#### batch4 epoch50

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.996755 | 0.995905 | 0.995000 | 0.939198 | 0.856959 | 0.685661 | 0.198077 |
| label1_thin | 0.997101 | 1.000000 | 0.995000 | 0.865526 | 0.653353 | 0.382594 | 0.001920 |
| label1_thick | 0.991944 | 1.000000 | 0.995000 | 0.798571 | 0.616846 | 0.340332 | 0.004112 |
| label2 | 0.997749 | 1.000000 | 0.995000 | 0.995000 | 0.959340 | 0.818590 | 0.248536 |
| label3 | 1.000000 | 0.998738 | 0.995000 | 0.967830 | 0.883916 | 0.587596 | 0.112467 |
| label4 | 1.000000 | 0.972597 | 0.995000 | 0.995000 | 0.995000 | 0.930790 | 0.471688 |
| label5 | 0.991475 | 1.000000 | 0.995000 | 0.967459 | 0.967075 | 0.816540 | 0.295281 |
| label6 | 0.999017 | 1.000000 | 0.995000 | 0.985000 | 0.923182 | 0.923182 | 0.252535 |

### 8.5 RHINO R50-RIoU batch2/batch4

#### batch2 AP50-best

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.985410 | 0.995654 | 0.995000 | 0.911655 | 0.788412 | 0.596849 | 0.163323 |
| label1_thin | 0.990466 | 1.000000 | 0.995000 | 0.770304 | 0.499512 | 0.272419 | 0.006049 |
| label1_thick | 1.000000 | 0.978152 | 0.995000 | 0.739914 | 0.434611 | 0.104691 | 0.003235 |
| label2 | 1.000000 | 0.991425 | 0.995000 | 0.995000 | 0.909048 | 0.605898 | 0.044622 |
| label3 | 0.963602 | 1.000000 | 0.995000 | 0.939669 | 0.851440 | 0.549698 | 0.083417 |
| label4 | 0.982935 | 1.000000 | 0.995000 | 0.995000 | 0.995000 | 0.973491 | 0.557014 |
| label5 | 0.983880 | 1.000000 | 0.995000 | 0.976698 | 0.893491 | 0.735962 | 0.092018 |
| label6 | 0.976992 | 1.000000 | 0.995000 | 0.965000 | 0.935784 | 0.935784 | 0.356904 |

#### batch2 epoch50

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.992466 | 0.991982 | 0.994948 | 0.959565 | 0.878536 | 0.659244 | 0.264438 |
| label1_thin | 1.000000 | 0.979560 | 0.995000 | 0.896074 | 0.770482 | 0.280394 | 0.017062 |
| label1_thick | 0.958624 | 1.000000 | 0.995000 | 0.938889 | 0.652715 | 0.300384 | 0.125000 |
| label2 | 0.998650 | 1.000000 | 0.995000 | 0.995000 | 0.904713 | 0.719197 | 0.220429 |
| label3 | 1.000000 | 0.983185 | 0.995000 | 0.945566 | 0.918196 | 0.639405 | 0.136161 |
| label4 | 0.997590 | 1.000000 | 0.995000 | 0.995000 | 0.995000 | 0.966698 | 0.772240 |
| label5 | 0.995354 | 1.000000 | 0.995000 | 0.981428 | 0.979528 | 0.779512 | 0.261836 |
| label6 | 0.997044 | 0.981132 | 0.994636 | 0.965000 | 0.929118 | 0.929118 | 0.318336 |

#### batch4 AP50-best

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.965274 | 0.979237 | 0.989985 | 0.938151 | 0.872691 | 0.662231 | 0.213762 |
| label1_thin | 0.941425 | 0.971429 | 0.972068 | 0.790346 | 0.696377 | 0.388212 | 0.019675 |
| label1_thick | 0.946023 | 0.974441 | 0.989211 | 0.908432 | 0.610384 | 0.204502 | 0.015927 |
| label2 | 0.981141 | 0.981601 | 0.994444 | 0.994444 | 0.994444 | 0.755904 | 0.220268 |
| label3 | 0.925820 | 0.981132 | 0.991518 | 0.943886 | 0.912554 | 0.625475 | 0.112245 |
| label4 | 0.981121 | 0.980561 | 0.994444 | 0.994444 | 0.994444 | 0.939248 | 0.668540 |
| label5 | 0.981385 | 0.994768 | 0.994074 | 0.970886 | 0.970778 | 0.795844 | 0.189983 |
| label6 | 1.000000 | 0.970728 | 0.994138 | 0.964615 | 0.929854 | 0.926431 | 0.269699 |

#### batch4 epoch50

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.961146 | 0.973812 | 0.987300 | 0.932056 | 0.861031 | 0.655601 | 0.231351 |
| label1_thin | 0.939908 | 0.942857 | 0.970528 | 0.779690 | 0.653566 | 0.410220 | 0.041065 |
| label1_thick | 0.943803 | 0.933544 | 0.986667 | 0.893922 | 0.593663 | 0.230778 | 0.008607 |
| label2 | 0.981348 | 0.992782 | 0.993704 | 0.993704 | 0.956073 | 0.730986 | 0.236348 |
| label3 | 0.936321 | 1.000000 | 0.993909 | 0.945910 | 0.912862 | 0.624973 | 0.150245 |
| label4 | 0.955890 | 1.000000 | 0.979064 | 0.979064 | 0.979064 | 0.920636 | 0.631676 |
| label5 | 0.980848 | 0.966369 | 0.992417 | 0.967104 | 0.966990 | 0.744455 | 0.211574 |
| label6 | 0.989902 | 0.981132 | 0.994815 | 0.965000 | 0.965000 | 0.927157 | 0.339945 |

### 8.6 Oriented R-CNN R50-FPN batch2/batch4

#### batch2 AP50-best epoch6

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.997468 | 0.996280 | 0.995000 | 0.892713 | 0.745466 | 0.415088 | 0.033228 |
| label1_thin | 1.000000 | 0.973958 | 0.995000 | 0.749513 | 0.538218 | 0.209263 | 0.033571 |
| label1_thick | 0.991186 | 1.000000 | 0.995000 | 0.602778 | 0.308333 | 0.206333 | 0.000000 |
| label2 | 0.997910 | 1.000000 | 0.995000 | 0.995000 | 0.769893 | 0.324057 | 0.003465 |
| label3 | 0.999528 | 1.000000 | 0.995000 | 0.971792 | 0.731795 | 0.230687 | 0.014559 |
| label4 | 0.997111 | 1.000000 | 0.995000 | 0.995000 | 0.995000 | 0.906104 | 0.123120 |
| label5 | 0.998127 | 1.000000 | 0.995000 | 0.969906 | 0.942571 | 0.667775 | 0.049739 |
| label6 | 0.998413 | 1.000000 | 0.995000 | 0.965000 | 0.932451 | 0.361399 | 0.008143 |

#### batch2 epoch50

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.998536 | 0.996358 | 0.995000 | 0.933204 | 0.888799 | 0.686510 | 0.182745 |
| label1_thin | 1.000000 | 0.974509 | 0.995000 | 0.863184 | 0.678018 | 0.495807 | 0.050625 |
| label1_thick | 0.996671 | 1.000000 | 0.995000 | 0.792167 | 0.740500 | 0.200833 | 0.007857 |
| label2 | 0.998889 | 1.000000 | 0.995000 | 0.995000 | 0.952503 | 0.843347 | 0.217161 |
| label3 | 0.998953 | 1.000000 | 0.995000 | 0.949343 | 0.949343 | 0.659070 | 0.119511 |
| label4 | 0.997639 | 1.000000 | 0.995000 | 0.995000 | 0.995000 | 0.943856 | 0.505316 |
| label5 | 0.998711 | 1.000000 | 0.995000 | 0.972736 | 0.941226 | 0.729621 | 0.178646 |
| label6 | 0.998887 | 1.000000 | 0.995000 | 0.965000 | 0.965000 | 0.933039 | 0.200100 |

#### batch4 AP50-best

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.997165 | 0.995740 | 0.995000 | 0.912147 | 0.794084 | 0.532095 | 0.075518 |
| label1_thin | 1.000000 | 0.976487 | 0.995000 | 0.833587 | 0.685459 | 0.428448 | 0.077857 |
| label1_thick | 0.992708 | 1.000000 | 0.995000 | 0.627951 | 0.237333 | 0.100833 | 0.000000 |
| label2 | 0.997423 | 1.000000 | 0.995000 | 0.995000 | 0.928585 | 0.512838 | 0.026435 |
| label3 | 1.000000 | 0.993694 | 0.995000 | 0.973491 | 0.892481 | 0.490048 | 0.035906 |
| label4 | 0.997822 | 1.000000 | 0.995000 | 0.995000 | 0.995000 | 0.838506 | 0.184216 |
| label5 | 0.993914 | 1.000000 | 0.995000 | 0.995000 | 0.854729 | 0.474879 | 0.081554 |
| label6 | 0.998290 | 1.000000 | 0.995000 | 0.965000 | 0.965000 | 0.879113 | 0.122660 |

#### batch4 epoch50

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.998666 | 0.996384 | 0.995000 | 0.931711 | 0.861925 | 0.689797 | 0.208225 |
| label1_thin | 1.000000 | 0.974686 | 0.995000 | 0.895635 | 0.721242 | 0.392427 | 0.009804 |
| label1_thick | 0.996758 | 1.000000 | 0.995000 | 0.736765 | 0.555679 | 0.235543 | 0.000000 |
| label2 | 0.998687 | 1.000000 | 0.995000 | 0.995000 | 0.952441 | 0.778780 | 0.236529 |
| label3 | 0.999207 | 1.000000 | 0.995000 | 0.968962 | 0.948363 | 0.742877 | 0.107400 |
| label4 | 0.998696 | 1.000000 | 0.995000 | 0.995000 | 0.995000 | 0.969906 | 0.604377 |
| label5 | 0.998580 | 1.000000 | 0.995000 | 0.945993 | 0.895748 | 0.779928 | 0.225600 |
| label6 | 0.998735 | 1.000000 | 0.995000 | 0.984623 | 0.965000 | 0.929118 | 0.273866 |

当前共整理六组模型路线、十七张完整分类别表：YOLO11l baseline best；YOLO26l/m best 与 last；KLD、RIoU、Oriented R-CNN 的 batch2/batch4 及对应 AP50-best/epoch50。所有已提供并完成统一评估的训练结果均已按 `all + 7 个标签 × 7 项指标` 完整收录，第 5/6 节保留用于快速比较的摘要，第 8 节作为详细结果总表。

## 9. 当前评估入口

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
  --pred-labels runs/rhino/rhino_r50_kld_e50_img1280_b4/test_epoch50/labels \
  --split test \
  --output runs/rhino/rhino_r50_kld_e50_img1280_b4/test_epoch50/custom_metrics.csv
```

Oriented R-CNN R50-FPN epoch50 统一评估：

```bash
CUDA_VISIBLE_DEVICES=2 ~/miniconda3/envs/rhino/bin/python scripts/evaluate_oriented_rcnn.py \
  --rhino-root ~/RHINO \
  --config runs/oriented_rcnn/oriented_rcnn_r50_fpn_e50_img1280_b4/config.py \
  --weights runs/oriented_rcnn/oriented_rcnn_r50_fpn_e50_img1280_b4/epoch_50.pth \
  --data datasets/obb_thin_thick/data.yaml \
  --split test \
  --min-conf 0.001 \
  --run-dir runs/oriented_rcnn/oriented_rcnn_r50_fpn_e50_img1280_b4/eval_epoch50 \
  --mmrotate-python ~/miniconda3/envs/rhino/bin/python \
  --metric-python ~/miniconda3/bin/python
```

完整训练、预测转换和统一 mAP85/mAP90 评测命令见 [代码架构文档](docs/architecture.md)。顶层脚本提供简洁命令，具体实现按 `scripts/models/<model>/` 分类；原有脚本和 `yolo11_obb` import 暂时保留兼容，不影响历史训练与权重。

## 10. 文档归档

前期依赖安装、数据筛选、YOLO 调参、deskew/fusion、RHINO 路线和分类实验已按项目阶段拆分。统一入口见 [项目文档索引](docs/README.md)。迁移前的完整 README 仅作为只读快照保留，不再作为当前运行说明。

主 README 只维护当前数据口径、当前候选模型和可直接比较的统一评估结果。

## 11. 下游 ResNet18 分类

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

## 12. 端子检测网页

网页第一版已接通完整闭环：1–100 张图片上传、YOLO11l-OBB 分区域检测、label3/label5 ResNet18 OK/NG 分类、实时阶段与进度、检测前后对比、任务状态筛选、失败图片重试和可刷新恢复的持久化历史。其他区域以灰框标记为“暂不支持分类”。颜色分类仅预留 API 与 UI 位置，当前不会生成或猜测颜色结果。

### 12.1 权重和运行环境

固定使用以下三个本地权重：

```text
weights/detector/yolo11l_obb_best.pt
weights/classifiers/label3/resnet18_best.pt
weights/classifiers/label5/resnet18_best.pt
```

创建环境并配置服务：

```bash
python3.11 -m venv .venv
.venv/bin/pip install -r requirements-web.txt
cp .env.example .env
```

`.env` 至少要设置 PostgreSQL URL、持久化目录、三个权重路径和设备：

```dotenv
DATABASE_URL=postgresql+psycopg://terminal:请替换密码@127.0.0.1:5432/terminal_inspection
INSPECTION_STORAGE_ROOT=/srv/terminal-inspection/artifacts
DETECTOR_WEIGHTS=./weights/detector/yolo11l_obb_best.pt
LABEL3_CLASSIFIER_WEIGHTS=./weights/classifiers/label3/resnet18_best.pt
LABEL5_CLASSIFIER_WEIGHTS=./weights/classifiers/label5/resnet18_best.pt
DETECTION_DEVICE=0
CLASSIFICATION_DEVICE=0
```

不要把包含真实密码的 `.env` 提交到 Git。

### 12.2 开发启动

先迁移数据库，再分别启动 API、单个 Worker 和前端：

```bash
.venv/bin/python -m alembic upgrade head
.venv/bin/python scripts/run_terminal_api.py --host 127.0.0.1 --port 8000
.venv/bin/python scripts/run_terminal_worker.py
```

```bash
cd web_frontend
npm ci
npm run dev -- --host 127.0.0.1 --port 5173
```

访问 `http://127.0.0.1:5173/tasks`，健康接口为 `http://127.0.0.1:8000/api/v1/health`。Vite 开发服务已代理 `/api`；生产环境由 Nginx 代理。

### 12.3 Compose 与学校服务器部署

Compose 启动命令：

```bash
docker compose config
docker compose up -d postgres
docker compose run --rm api alembic upgrade head
docker compose up -d api worker frontend
curl -fsS http://127.0.0.1:8080/api/v1/health
```

Compose 使用 `postgres-data` 保存 PostgreSQL，使用 `inspection-data` 保存原图和结果图。原生服务器建议将 artifact 目录固定为 `/srv/terminal-inspection/artifacts`，并用 `pg_dump` 与 `rsync` 同时备份数据库和图像。完整 systemd、Nginx、日志、备份和回滚步骤见 [网页部署文档](docs/web-deployment.md)。

### 12.4 验证命令

```bash
.venv/bin/python -m unittest discover -s tests -v
cd web_frontend
npm test -- --run
npm run build
```

真实三模型 CPU smoke：

```bash
.venv/bin/python scripts/smoke_terminal_pipeline.py \
  --detector weights/detector/yolo11l_obb_best.pt \
  --label3 weights/classifiers/label3/resnet18_best.pt \
  --label5 weights/classifiers/label5/resnet18_best.pt \
  --image datasets/rhino_obb/test/images/CropImage_20260128141159842_F3-I0_OK-3.png \
  --output runs/web-smoke-final \
  --det-device cpu \
  --cls-device cpu
```
