# YOLO11-OBB Training

## 1. 项目数据集:

```text
datasets/154843_obb_train_val_test
```

## 2. 数据集配置:

```text
datasets/154843_obb_train_val_test/data.yaml
```

## 3. 依赖下载

```bash
python3 -m pip install -r requirements.txt
```

## 4. 检查数据

只检查数据完整性，不启动训练

```bash
python3 scripts/train_yolo11_obb.py --dry-run
```

## 5. Train

```bash
python3 scripts/train_yolo11_obb.py \
  --model yolo11n-obb.pt \
  --data datasets/154843_obb_train_val_test/data.yaml \
  --imgsz 1024 \
  --epochs 150 \
  --batch 8
```

当前训练中最好模型权重

```text
runs/obb/terminal_obb_yolo11n/weights/best.pt
```

## 6. Evaluate

使用test数据集进行评估

```bash
python3 scripts/evaluate_yolo11_obb.py \
  --model runs/obb/terminal_obb_yolo11n/weights/best.pt \
  --split test
```

## 7. Predict

```bash
python3 scripts/predict_yolo11_obb.py \
  --model runs/obb/terminal_obb_yolo11n/weights/best.pt \
  --source datasets/154843_obb_train_val_test/images/test
```

Ultralytics YOLO11 supports OBB models such as `yolo11n-obb.pt`,
`yolo11s-obb.pt`, `yolo11m-obb.pt`, `yolo11l-obb.pt`, and `yolo11x-obb.pt`.

## 8. AnyLabeling OBB 数据集

主要修改标注框的精确度 \
新标注转换后的完整 OBB 目录：

```text
已打标的数据202604/user1_2026-03-16_154843_obb_converted
```

按母样本分组后，只保留 `label1-label6`，并按 `train:test = 8:2` 生成的数据集：

```text
datasets/154843_obb_converted_label1_6_train_test
```

该数据集没有单独的 `images/val` 和 `labels/val` 目录。`data.yaml` 中的 `val` 指向 `test`，只用于兼容 Ultralytics 数据配置；训练时使用 `--no-val` 关闭训练期间验证，训练结束后再单独对 `test` 做 evaluate。

重新生成数据集：

```bash
python3 scripts/create_train_test_dataset.py
```

检查数据：

```bash
python3 scripts/train_yolo11_obb.py \
  --data datasets/154843_obb_converted_label1_6_train_test/data.yaml \
  --no-val \
  --dry-run
```

训练：

```bash
python3 scripts/train_yolo11_obb.py \
  --model yolo11n-obb.pt \
  --data datasets/154843_obb_converted_label1_6_train_test/data.yaml \
  --imgsz 1024 \
  --epochs 150 \
  --batch 8 \
  --device mps \
  --name yolo11n_154843_converted_label1_6_train_test \
  --no-val
```

训练结束后使用 test 集评估：

```bash
python3 scripts/evaluate_yolo11_obb.py \
  --model runs/obb/yolo11n_154843_converted_label1_6_train_test/weights/best.pt \
  --data datasets/154843_obb_converted_label1_6_train_test/data.yaml \
  --split test \
  --name yolo11n_154843_converted_label1_6_test_eval
```

评估脚本会在本次 evaluate 输出目录中额外写入：

```text
custom_metrics.csv
```

该文件只包含当前关注的指标：

```text
class,precision,recall,mAP50,mAP80,mAP85,mAP90,mAP95
```

`custom_metrics.csv` 不输出 `mAP50-95`；如需查看 Ultralytics 原始默认指标，仍可查看 evaluate 命令自身打印和生成的默认结果文件。

## 9. 训练日志

使用模型：`yolo11n-obb`

### 9.1 数据集划分

当前使用的数据集：

```text
datasets/154843_obb_train_val_test
```

划分方式是按母样本分组，避免同一个母样本的不同裁片同时进入训练集和验证/测试集。

```text
train: 294 images, 47 groups
val:    66 images, 10 groups
test:   69 images, 11 groups
```

类别数量存在明显不均衡：

| 类别ID | 类别名 | train实例 | val实例 | test实例 | 总实例 | train图片 | val图片 | test图片 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | label1 | 283 | 64 | 65 | 412 | 283 | 64 | 65 |
| 1 | label2 | 268 | 59 | 63 | 390 | 268 | 59 | 63 |
| 2 | label3 | 268 | 59 | 63 | 390 | 268 | 59 | 63 |
| 3 | label4 | 268 | 59 | 63 | 390 | 268 | 59 | 63 |
| 4 | label5 | 268 | 59 | 63 | 390 | 268 | 59 | 63 |
| 5 | label6 | 268 | 59 | 63 | 390 | 268 | 59 | 63 |
| 6 | label7 | 1 | 0 | 1 | 2 | 1 | 0 | 1 |
| 7 | lable7 | 1 | 0 | 0 | 1 | 1 | 0 | 0 |
| 8 | other | 11 | 2 | 4 | 17 | 11 | 2 | 4 |

其中 `lable7` 是原始标注里的拼写问题，`label7/lable7/other` 样本数量过少，后续不适合作为主要评估依据。

### 9.2 第一次训练：10 轮 baseline

训练目录：

```text
runs/obb/terminal_obb_yolo11n
```

训练命令：

```bash
python3 scripts/train_yolo11_obb.py \
  --model yolo11n-obb.pt \
  --epochs 10 \
  --imgsz 1024 \
  --batch 8 \
  --device mps \
  --name terminal_obb_yolo11n
```

训练结果：

```text
best epoch: 10
Precision: 0.87664
Recall:    0.91741
mAP50:     0.93088
mAP50-95:  0.73026
```

对应权重：

```text
runs/obb/terminal_obb_yolo11n/weights/best.pt
```

结论：10 轮后模型已经能较好识别 `label1-label6`，但 `mAP50-95` 仍有提升空间，因此继续基于 `best.pt` 做第二阶段训练。

### 9.3 第二次训练：基于 10 轮 best.pt 继续训练 30 轮

训练目录：

```text
runs/obb/yolo11n_from_e10_e30
```

训练命令：

```bash
python3 scripts/train_yolo11_obb.py \
  --model runs/obb/terminal_obb_yolo11n/weights/best.pt \
  --epochs 30 \
  --imgsz 1024 \
  --batch 8 \
  --device mps \
  --name yolo11n_from_e10_e30
```

第二阶段最佳结果：

```text
best epoch: 25
Precision: 0.99120
Recall:    0.91736
mAP50:     0.94383
mAP50-95:  0.79170
```

第二阶段最后一轮结果：

```text
last epoch: 30
Precision: 0.98314
Recall:    0.91741
mAP50:     0.94723
mAP50-95:  0.78143
```

对应权重：

```text
runs/obb/yolo11n_from_e10_e30/weights/best.pt
```

### 9.4 两次训练结果对比

| 阶段 | 最佳 epoch | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 轮 baseline | 10 | 0.87664 | 0.91741 | 0.93088 | 0.73026 |
| 继续 30 轮 | 25 | 0.99120 | 0.91736 | 0.94383 | 0.79170 |

提升幅度：

```text
Precision: +0.11456
Recall:    -0.00005
mAP50:     +0.01295
mAP50-95:  +0.06144
```

结论：

- 继续训练 30 轮后，`mAP50-95` 从 `0.73026` 提升到 `0.79170`，说明第二阶段训练有效。
- `Precision` 明显提升，说明误检减少。
- `Recall` 基本不变，说明漏检问题没有明显改善。
- 第二阶段第 25 轮是最佳点，第 30 轮 `mAP50-95` 降到 `0.78143`，说明后续继续训练的收益可能变小，并开始出现波动。

### 9.5 Test evaluate 结果

评估目录：

```text
runs/obb/yolo11n_from_e10_e30_test
```

该 evaluate 目录保存了图像结果，包括：

```text
BoxPR_curve.png
confusion_matrix.png
confusion_matrix_normalized.png
val_batch*_labels.jpg
val_batch*_pred.jpg
```

使用 `runs/obb/yolo11n_from_e10_e30/weights/best.pt` 在 test 集上的评估结果：

| 类别 | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 69 | 385 | 0.868 | 0.875 | 0.871 | 0.708 |
| label1 | 65 | 65 | 0.992 | 1.000 | 0.995 | 0.739 |
| label2 | 63 | 63 | 0.992 | 1.000 | 0.995 | 0.914 |
| label3 | 63 | 63 | 0.993 | 1.000 | 0.995 | 0.840 |
| label4 | 63 | 63 | 0.993 | 1.000 | 0.995 | 0.901 |
| label5 | 63 | 63 | 0.993 | 1.000 | 0.995 | 0.835 |
| label6 | 63 | 63 | 0.993 | 1.000 | 0.995 | 0.884 |
| label7 | 1 | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| other | 4 | 4 | 0.986 | 1.000 | 0.995 | 0.547 |

`lable7` 在 test 集中没有实例，因此该次 test evaluate 没有单独输出 `lable7` 行。

从 `confusion_matrix.png` 看：

```text
label1: 65 hit
label2: 63 hit
label3: 63 hit
label4: 63 hit
label5: 63 hit
label6: 63 hit
other:  4 hit
label7: 1 missed as background
background: 1 false positive as label1
```

结论：

- `label1-label6` 在 test 上表现稳定。
- `label7` 样本太少，test 中 1 个目标被漏检，AP 为 0。
- `other` 在当前 test 中命中了 4 个，`mAP50` 为 `0.995`，说明能大致定位；但 `mAP50-95` 只有 `0.547`，说明严格 IoU 下框的位置、角度或边界仍不够稳定。
- `all` 的 `mAP50-95` 为 `0.708`，被 `label7` 和 `other` 这两个少样本类别明显拉低。
- 当前最可靠的模型是：

```text
runs/obb/yolo11n_from_e10_e30/weights/best.pt
```

### 8.6 当前遇到的问题和后续优化方向

1. `label7/lable7/other` 样本太少。
   - `label7` 和 `lable7` 在训练集中各只有 1 个样本。
   - `other` 在训练集中只有 11 个样本。
   - 这些类别会拖低整体指标，并让评估结果不稳定。

2. `lable7` 是拼写错误类别。
   - 后续应统一为 `label7`，或者直接从当前检测任务中移除。

3. 当前 9 类模型主要适合检测 `label1-label6` 六个区域。
   - 如果业务目标是定位主要区域，建议先训练一个只包含 `label1-label6` 的干净数据集版本。
   - 如果业务必须识别 `label7/other`，需要补充足够样本后再训练。

4. 继续增加 epoch 的收益可能有限。
   - 第二阶段最佳点是 epoch 25。
   - epoch 30 的 `mAP50-95` 已经低于最佳点。
   - 后续应优先清理类别和数据，而不是盲目增加训练轮数。

### 8.7 仅保留 label1-label6 后重新训练 50 轮

为避免 `label7/lable7/other` 这几个少样本类别拉低整体评估结果，重新生成了只包含 `label1-label6` 的数据集，原始数据集保持不变。

新数据集：

```text
datasets/154843_obb_label1_6_train_val_test
```

新数据配置：

```text
datasets/154843_obb_label1_6_train_val_test/data.yaml
```

类别只保留：

```text
0: label1
1: label2
2: label3
3: label4
4: label5
5: label6
```

过滤后的数据分布：

| split | Images | Labels | 保留目标 | 移除目标 | 空标签图片 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 294 | 294 | 1623 | 13 | 11 |
| val | 66 | 66 | 359 | 2 | 2 |
| test | 69 | 69 | 380 | 5 | 4 |

训练目录：

```text
runs/obb/yolo11n_label1_6_e50
```

训练命令：

```bash
python3 scripts/train_yolo11_obb.py \
  --model yolo11n-obb.pt \
  --data datasets/154843_obb_label1_6_train_val_test/data.yaml \
  --epochs 50 \
  --imgsz 1024 \
  --batch 8 \
  --device mps \
  --name yolo11n_label1_6_e50
```

50 轮训练中的最佳验证集结果：

```text
best epoch: 36
Precision: 0.99394
Recall:    1.00000
mAP50:     0.99485
mAP50-95:  0.85876
```

第 50 轮最后结果：

```text
last epoch: 50
Precision: 0.99561
Recall:    0.99780
mAP50:     0.99497
mAP50-95:  0.85215
```

对应权重：

```text
runs/obb/yolo11n_label1_6_e50/weights/best.pt
```

结论：

- 第 36 轮是本次训练的最佳点，因此后续 evaluate 和 predict 应优先使用 `best.pt`。
- 第 50 轮的 `mAP50-95` 比最佳点略低，说明 36 轮之后继续训练的收益不明显，指标进入小幅波动。
- 去掉少样本类别后，验证集整体指标明显更稳定。

### 8.8 label1-label6 模型的 test evaluate 结果

评估目录：

```text
runs/obb/yolo11n_label1_6_e50_test
```

评估命令：

```bash
python3 scripts/evaluate_yolo11_obb.py \
  --data datasets/154843_obb_label1_6_train_val_test/data.yaml \
  --model runs/obb/yolo11n_label1_6_e50/weights/best.pt \
  --split test \
  --imgsz 1024 \
  --batch 8 \
  --device mps \
  --name yolo11n_label1_6_e50_test
```

test evaluate 结果：

| 类别 | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 69 | 380 | 0.997 | 0.997 | 0.995 | 0.852 |
| label1 | 65 | 65 | 1.000 | 0.984 | 0.995 | 0.715 |
| label2 | 63 | 63 | 0.996 | 1.000 | 0.995 | 0.909 |
| label3 | 63 | 63 | 0.992 | 1.000 | 0.995 | 0.847 |
| label4 | 63 | 63 | 0.998 | 1.000 | 0.995 | 0.911 |
| label5 | 63 | 63 | 0.998 | 1.000 | 0.995 | 0.843 |
| label6 | 63 | 63 | 0.998 | 1.000 | 0.995 | 0.885 |

与 9 类模型的 test evaluate 对比：

| 模型 | 类别范围 | test Instances | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `yolo11n_from_e10_e30` | label1-label7, lable7, other | 385 | 0.868 | 0.875 | 0.871 | 0.708 |
| `yolo11n_label1_6_e50` | label1-label6 | 380 | 0.997 | 0.997 | 0.995 | 0.852 |

提升幅度：

```text
Precision: +0.129
Recall:    +0.122
mAP50:     +0.124
mAP50-95:  +0.144
```

结论：

- 只训练 `label1-label6` 后，test 集整体 `mAP50-95` 从 `0.708` 提升到 `0.852`。
- `label2-label6` 的 `mAP50-95` 都在 `0.843` 以上，定位质量较稳定。
- `label1` 的 `mAP50-95` 为 `0.715`，明显低于其他 5 类，是下一步需要重点查看预测图和误差来源的类别。
- 当前最可靠的六分类模型是：

```text
runs/obb/yolo11n_label1_6_e50/weights/best.pt
```

### 8.9 后续模型实验建议

尝试参数更大的 YOLO11-OBB 模型

```text
yolo11s-obb.pt
```

原因：

- 当前 `yolo11n-obb` 已经达到较高的 `mAP50`，主要提升空间在更严格的 `mAP50-95`，也就是框的位置、角度和边界质量。
- 更大的模型可能提升 OBB 框的精细定位能力。
- 数据量仍然不大，直接使用 `yolo11m/l/x` 容易训练更慢，也更容易过拟合。

建议实验命令：

```bash
python3 scripts/train_yolo11_obb.py \
  --model yolo11s-obb.pt \
  --data datasets/154843_obb_label1_6_train_val_test/data.yaml \
  --epochs 50 \
  --imgsz 1024 \
  --batch 8 \
  --device mps \
  --name yolo11s_label1_6_e50
```

如果显存或内存不足，可以把 `batch` 改小：

```bash
--batch 4
```

是否更换更大参数模型，比较：

```text
val mAP50-95
test mAP50-95
label1 的 mAP50-95
```

如果 `yolo11s-obb` 的 test `mAP50-95` 没有明显超过 `0.852`，就不建议继续盲目加大模型，应该转向检查 `label1` 的预测图、标注一致性和数据增强参数。

### 8.10 yolo11s-obb 50 轮训练与 test evaluate 结果

训练目录：

```text
runs/obb/yolo11s_label1_6_e50
```

训练命令：

```bash
python3 scripts/train_yolo11_obb.py \
  --model yolo11s-obb.pt \
  --data datasets/154843_obb_label1_6_train_val_test/data.yaml \
  --epochs 50 \
  --imgsz 1024 \
  --batch 8 \
  --device mps \
  --name yolo11s_label1_6_e50
```

50 轮训练中的最佳验证集结果：

```text
best epoch: 47
Precision: 0.99491
Recall:    0.99951
mAP50:     0.99485
mAP50-95:  0.85857
```

第 50 轮最后结果：

```text
last epoch: 50
Precision: 0.99359
Recall:    0.99917
mAP50:     0.99485
mAP50-95:  0.85438
```

对应权重：

```text
runs/obb/yolo11s_label1_6_e50/weights/best.pt
```

评估目录：

```text
runs/obb/yolo11s_label1_6_e50_test
```

评估命令：

```bash
python3 scripts/evaluate_yolo11_obb.py \
  --data datasets/154843_obb_label1_6_train_val_test/data.yaml \
  --model runs/obb/yolo11s_label1_6_e50/weights/best.pt \
  --split test \
  --imgsz 1024 \
  --batch 8 \
  --device mps \
  --name yolo11s_label1_6_e50_test
```

test evaluate 结果：

| 类别 | Images | Instances | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 69 | 380 | 0.991 | 0.996 | 0.995 | 0.833 |
| label1 | 65 | 65 | 0.970 | 0.978 | 0.993 | 0.717 |
| label2 | 63 | 63 | 0.994 | 1.000 | 0.995 | 0.887 |
| label3 | 63 | 63 | 0.994 | 1.000 | 0.995 | 0.833 |
| label4 | 63 | 63 | 0.993 | 1.000 | 0.995 | 0.885 |
| label5 | 63 | 63 | 0.999 | 1.000 | 0.995 | 0.809 |
| label6 | 63 | 63 | 0.995 | 1.000 | 0.995 | 0.866 |

与 `yolo11n_label1_6_e50` 的 test evaluate 对比：

| 模型 | Precision | Recall | mAP50 | test mAP50-95 |
| --- | ---: | ---: | ---: | ---: |
| `yolo11n_label1_6_e50` | 0.997 | 0.997 | 0.995 | 0.852 |
| `yolo11s_label1_6_e50` | 0.991 | 0.996 | 0.995 | 0.833 |

`yolo11s` 相比 `yolo11n` 的变化：

```text
Precision: -0.006
Recall:    -0.001
mAP50:      0.000
mAP50-95:  -0.019
```

分类别 `mAP50-95` 对比：

| 类别 | yolo11n | yolo11s | 变化 |
| --- | ---: | ---: | ---: |
| label1 | 0.715 | 0.717 | +0.002 |
| label2 | 0.909 | 0.887 | -0.022 |
| label3 | 0.847 | 0.833 | -0.014 |
| label4 | 0.911 | 0.885 | -0.026 |
| label5 | 0.843 | 0.809 | -0.034 |
| label6 | 0.885 | 0.866 | -0.019 |

结论：

- `yolo11s-obb` 的验证集最佳 `mAP50-95` 为 `0.85857`，与 `yolo11n-obb` 的 `0.85876` 基本持平。
- 在 test 集上，`yolo11s-obb` 的整体 `mAP50-95` 为 `0.833`，低于 `yolo11n-obb` 的 `0.852`。
- `label1` 只从 `0.715` 提升到 `0.717`，提升非常小；其余类别的 `mAP50-95` 都下降。
- 本轮实验不支持继续加大模型。当前更推荐保留 `yolo11n_label1_6_e50/weights/best.pt` 作为六分类任务的主要模型。
- 下一步应优先检查 `label1` 的预测图、标注边界一致性和增强策略，而不是继续尝试 `yolo11m/l/x`。
