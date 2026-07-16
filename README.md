# YOLO11-OBB Training

## 1. 当前实验数据集

当前稳定基线和模型选型对比都使用原图坐标系下的 thin/thick 数据集：

```text
datasets/154843_after_20260121210219803_no_index1_label1_thin_thick_train_test
```

旧的 deskew 数据集和更早的数据集已归档到：

```text
archive/datasets_legacy
```

## 2. 数据集配置

```text
datasets/154843_after_20260121210219803_no_index1_label1_thin_thick_train_test/data.yaml
```

## 3. 依赖下载

```bash
python3 -m pip install -r requirements.txt
```

## 4. 检查数据

只检查数据完整性，不启动训练

```bash
python3 scripts/train_yolo11_obb.py \
  --data datasets/154843_after_20260121210219803_no_index1_label1_thin_thick_train_test/data.yaml \
  --dry-run
```

## 5. Train

模型选型的首个轻量基线。它与原 YOLO11l baseline 使用完全相同的数据集、图像尺寸、训练轮数、批大小、随机种子和数据增强参数；仅将初始权重换为 `yolo11n-obb.pt`。

```bash
python3 -u scripts/train_yolo11_obb.py \
  --model yolo11n-obb.pt \
  --data datasets/154843_after_20260121210219803_no_index1_label1_thin_thick_train_test/data.yaml \
  --imgsz 1280 \
  --epochs 50 \
  --batch 8 \
  --device mps \
  --workers 8 \
  --seed 42 \
  --name yolo11n_no_index1_thin_thick_e50_img1280_b8_deg0_compare \
  --degrees 0.0
```

新基线完成后权重位于：

```text
runs/obb/yolo11n_no_index1_thin_thick_e50_img1280_b8_deg0_compare/weights/best.pt
```

## 6. Evaluate

使用test数据集进行评估

```bash
python3 scripts/evaluate_yolo11_obb.py \
  --data datasets/154843_after_20260121210219803_no_index1_label1_thin_thick_train_test/data.yaml \
  --model runs/obb/yolo11n_no_index1_thin_thick_e50_img1280_b8_deg0_compare/weights/best.pt \
  --split test \
  --imgsz 1280 \
  --batch 8 \
  --device mps \
  --name yolo11n_no_index1_thin_thick_e50_img1280_b8_deg0_compare_eval
```

## 7. Predict

```bash
python3 scripts/predict_yolo11_obb.py \
  --model runs/obb/yolo11n_no_index1_thin_thick_e50_img1280_b8_deg0_compare/weights/best.pt \
  --source datasets/154843_after_20260121210219803_no_index1_label1_thin_thick_train_test/images/test \
  --imgsz 1280 \
  --device mps \
  --name yolo11n_no_index1_thin_thick_e50_img1280_b8_deg0_compare_pred
```

Ultralytics YOLO11 supports OBB models such as `yolo11n-obb.pt`,
`yolo11s-obb.pt`, `yolo11m-obb.pt`, `yolo11l-obb.pt`, and `yolo11x-obb.pt`.

## 8. AnyLabeling OBB 数据集

主要修改标注框的精确度。新标注转换后的完整 OBB 目录：

```text
已打标的数据202604/user1_2026-03-16_154843_obb_converted
```

当前主数据集把原 `label1` 按标注框上沿宽度拆成两类：

- `label1_thin`: 原 `label1` 上沿宽度 `< 164 px`
- `label1_thick`: 原 `label1` 上沿宽度 `>= 164 px`

其余类别从原 `label2-label6` 顺延为当前 `label2-label6`。原 `other` 类仍然过滤，不参与训练和评估。

当前主数据集：

```text
datasets/154843_obb_converted_label1_thin_thick_train_test
```

该数据集没有单独的 `images/val` 和 `labels/val` 目录。`data.yaml` 中的 `val` 指向 `images/test`，训练和 evaluate 都使用 `test` 作为验证/测试集。

```text
train: 342 images, 342 labels
test:   87 images, 87 labels
```

类别分布：

| split | objects | label1_thin | label1_thick | label2 | label3 | label4 | label5 | label6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1884 | 209 | 120 | 311 | 311 | 311 | 311 | 311 |
| test | 478 | 54 | 29 | 79 | 79 | 79 | 79 | 79 |

重新生成数据集：

```bash
python3 scripts/create_label1_thin_thick_dataset.py \
  --top-edge-threshold-px 164
```

检查数据：

```bash
python3 scripts/train_yolo11_obb.py \
  --data datasets/154843_obb_converted_label1_thin_thick_train_test/data.yaml \
  --dry-run
```

评估脚本会在 evaluate 输出目录中额外写入：

```text
custom_metrics.csv
```

该文件只包含当前关注的指标：

```text
class,precision,recall,mAP50,mAP80,mAP85,mAP90,mAP95
```

`custom_metrics.csv` 不输出 `mAP50-95`。

## 8.1 Label5 ResNet18 分类

使用人工标注的 AnyLabeling OBB 框先裁剪 `label5` 区域，并用 Excel 中的 `tag1` 作为 OK/NG 分类标签：

```bash
python3 scripts/create_label_classification_dataset.py \
  --excel outputs/label1_6_description.xlsx \
  --source '已打标的数据202604/user1_2026-03-16_154843_anylabeling' \
  --output datasets/classification/label5_ok_ng \
  --label label5 \
  --target-column tag1 \
  --train-ratio 0.8 \
  --seed 42 \
  --overwrite
```

重新评测已有权重：

```bash
.venv/bin/python scripts/evaluate_resnet18_classifier.py \
  --data datasets/classification/label5_ok_ng \
  --weights runs/classification/label5_resnet18_e30/weights/best.pt \
  --split test \
  --batch 8 \
  --workers 0 \
  --device cpu \
  --name label5_resnet18_eval \
  --exist-ok
```

对任意图片或目录做预测：

```bash
.venv/bin/python scripts/predict_resnet18_classifier.py \
  --weights runs/classification/label5_resnet18_e30/weights/best.pt \
  --source datasets/classification/label5_ok_ng/images/test \
  --batch 8 \
  --workers 0 \
  --device cpu \
  --name label5_resnet18_predict \
  --exist-ok
```

训练、评测和预测都会在 `runs/` 下写出 CSV 结果。训练脚本每个 epoch 会在终端输出 loss、accuracy 和 macro F1。

### 8.1.1 label3/label5 原始标注框分类结果

以下结果都基于 AnyLabeling 的原始人工 OBB 标注框裁剪，不接入 YOLO11-OBB 上游检测框。分类标签来自 `outputs/label1_6_description.xlsx` 中对应 label sheet 的 `tag1` 列，数据按 parent group 做 8:2 划分。

| label | dataset | total | train | test | train run | evaluate run | best epoch | test accuracy | test macro F1 |
| --- | --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: |
| label3 | `datasets/classification/label3_ok_ng` | 251 | NG=129, OK=69 | NG=36, OK=17 | `runs/classification/label3_resnet18_cpu_e30` | `runs/classification_eval/label3_resnet18_eval_cpu_e30` | 1 | 0.981132 | 0.977999 |
| label5 | `datasets/classification/label5_ok_ng` | 251 | NG=118, OK=80 | NG=35, OK=18 | `runs/classification/label5_resnet18_mps_e30` | `runs/classification_eval/label5_resnet18_eval_smoke` | 2 | 1.000000 | 1.000000 |

训练配置：

| label | epochs | batch | imgsz | lr | weight_decay | device | pretrained |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| label3 | 30 | 8 | 224 | 0.0001 | 0.0001 | cpu | true |
| label5 | 30 | 8 | 224 | 0.0001 | 0.0001 | mps | true |

`best.pt` 独立 evaluate 混淆矩阵：

label3:

```text
true\pred,NG,OK
NG,36,0
OK,1,16
```

label5:

```text
true\pred,NG,OK
NG,35,0
OK,0,18
```

label3 的训练 `last.pt` 最后一轮结果为 `test_accuracy=0.962264`、`test_macro_f1=0.955236`；上表中的结果是加载 `weights/best.pt` 后用独立 evaluate 脚本复评得到的结果。label5 的 `last.pt` 与 `best.pt` 在 test split 上均为 1.0。

当前第一版按用户要求使用 8:2 划分，`best.pt` 依据测试集 macro F1 选择。这个结果适合验证首版流程，但因为没有单独 validation split，指标会偏乐观。

## 9. label1_thin/thick 训练日志

当前日志基于 `label1_thin/label1_thick` 数据集实验，包含完整 AnyLabeling 数据集和按时间截取后的新子集。重点指标仍是 `mAP50`、`mAP80`、`mAP85`、`mAP90`、`mAP95`。

### 9.1 yolo11l-obb 50 轮 imgsz=1280 baseline

配置：

```text
model: yolo11l-obb.pt
epochs: 50
imgsz: 1280
batch: 4
device: 0
degrees: 0.0
val: test
```

`custom_metrics.csv` 结果：

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.993263 | 1.000000 | 0.994857 | 0.917490 | 0.805687 | 0.618882 | 0.195674 |
| label1_thin | 0.993305 | 1.000000 | 0.995000 | 0.812355 | 0.526756 | 0.371545 | 0.023200 |
| label1_thick | 0.964738 | 1.000000 | 0.994000 | 0.734939 | 0.423884 | 0.143500 | 0.047654 |
| label2 | 1.000000 | 1.000000 | 0.995000 | 0.995000 | 0.981076 | 0.870748 | 0.289849 |
| label3 | 1.000000 | 1.000000 | 0.995000 | 0.979684 | 0.927963 | 0.620615 | 0.146949 |
| label4 | 1.000000 | 1.000000 | 0.995000 | 0.995000 | 0.995000 | 0.928924 | 0.495581 |
| label5 | 1.000000 | 1.000000 | 0.995000 | 0.910454 | 0.800128 | 0.491211 | 0.078691 |
| label6 | 0.994795 | 1.000000 | 0.995000 | 0.995000 | 0.985000 | 0.905628 | 0.287796 |

IoU85 可视化统计：

```text
analysis: runs/analysis/iou85_overlay_yolo11l_label1_thin_thick_e50_img1280_b4
GT rows: 478
failed rows: 146
failed images: 71
missing predictions: 0
```

| class | GT | IoU85 failed | failed rate | mean IoU |
| --- | ---: | ---: | ---: | ---: |
| label1_thick | 29 | 20 | 69.0% | 0.778 |
| label1_thin | 54 | 33 | 61.1% | 0.796 |
| label2 | 79 | 9 | 11.4% | 0.897 |
| label3 | 79 | 27 | 34.2% | 0.870 |
| label4 | 79 | 10 | 12.7% | 0.901 |
| label5 | 79 | 34 | 43.0% | 0.848 |
| label6 | 79 | 13 | 16.5% | 0.888 |

角度段统计：

| GT top-edge angle | GT | IoU85 failed | failed rate | mean IoU |
| --- | ---: | ---: | ---: | ---: |
| 0-2 deg | 426 | 123 | 28.9% | 0.868 |
| 2-5 deg | 21 | 6 | 28.6% | 0.861 |
| 5-10 deg | 28 | 16 | 57.1% | 0.826 |
| 10+ deg | 3 | 1 | 33.3% | 0.849 |

结论：拆分后 `label1_thin` 的 `mAP90` 高于原 `label1` baseline，但 `label1_thick` 样本少且严格 IoU 仍弱。5-10 度样本失败率明显升高，说明角度一致性是非 label1 类别的重要问题。

### 9.2 yolo11l-obb 50 轮 imgsz=1280 degrees=5

配置：

```text
model: yolo11l-obb.pt
epochs: 50
imgsz: 1280
batch: 8
device: 0
degrees: 5.0
val: test
```

`custom_metrics.csv` 结果：

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.992978 | 0.999534 | 0.994857 | 0.919840 | 0.821532 | 0.621163 | 0.171728 |
| label1_thin | 0.990835 | 1.000000 | 0.995000 | 0.809321 | 0.526985 | 0.163999 | 0.004081 |
| label1_thick | 0.966558 | 0.996735 | 0.994000 | 0.734238 | 0.423700 | 0.270346 | 0.039765 |
| label2 | 1.000000 | 1.000000 | 0.995000 | 0.995000 | 0.995000 | 0.883671 | 0.224622 |
| label3 | 1.000000 | 1.000000 | 0.995000 | 0.972532 | 0.950316 | 0.670539 | 0.071438 |
| label4 | 0.993454 | 1.000000 | 0.995000 | 0.995000 | 0.995000 | 0.930823 | 0.509387 |
| label5 | 1.000000 | 1.000000 | 0.995000 | 0.937789 | 0.864720 | 0.542588 | 0.083300 |
| label6 | 1.000000 | 1.000000 | 0.995000 | 0.995000 | 0.995000 | 0.886178 | 0.269503 |

IoU85 可视化统计：

```text
analysis: runs/analysis/iou85_overlay_yolo11l_label1_thin_thick_e50_img1280_b8_deg5_valtest
GT rows: 478
failed rows: 162
failed images: 76
missing predictions: 0
```

| class | GT | IoU85 failed | failed rate | mean IoU |
| --- | ---: | ---: | ---: | ---: |
| label1_thick | 29 | 19 | 65.5% | 0.787 |
| label1_thin | 54 | 41 | 75.9% | 0.775 |
| label2 | 79 | 16 | 20.3% | 0.893 |
| label3 | 79 | 26 | 32.9% | 0.863 |
| label4 | 79 | 8 | 10.1% | 0.897 |
| label5 | 79 | 37 | 46.8% | 0.851 |
| label6 | 79 | 15 | 19.0% | 0.889 |

角度段统计：

| GT top-edge angle | GT | IoU85 failed | failed rate | mean IoU |
| --- | ---: | ---: | ---: | ---: |
| 0-2 deg | 426 | 142 | 33.3% | 0.862 |
| 2-5 deg | 21 | 6 | 28.6% | 0.879 |
| 5-10 deg | 28 | 12 | 42.9% | 0.847 |
| 10+ deg | 3 | 2 | 66.7% | 0.855 |

结论：`degrees=5` 改善了 5-10 度样本的固定阈值 IoU85 失败率，并提升 `label3/label5` 的 mAP85/mAP90；但它显著损伤 `label1_thin mAP90/mAP95`，且固定 `conf=0.25` 可视化下 IoU85 失败数从 146 增加到 162。因此 `degrees=5` 不适合作为默认增强强度。

### 9.3 thin/thick 实验对比

| 实验 | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 | IoU85 failed rows | failed images |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| b4 degrees0 | 0.993263 | 1.000000 | 0.994857 | 0.917490 | 0.805687 | 0.618882 | 0.195674 | 146 | 71 |
| b8 degrees5 | 0.992978 | 0.999534 | 0.994857 | 0.919840 | 0.821532 | 0.621163 | 0.171728 | 162 | 76 |
| b8 degrees5 - b4 degrees0 | -0.000285 | -0.000466 | +0.000000 | +0.002350 | +0.015845 | +0.002281 | -0.023946 | +16 | +5 |

按 test 集数量加权合并 `label1_thin/thick` 后：

| 实验 | weighted label1 mAP80 | weighted label1 mAP85 | weighted label1 mAP90 | weighted label1 mAP95 |
| --- | ---: | ---: | ---: | ---: |
| 原 label1 baseline | 0.792929 | 0.427479 | 0.283031 | 0.008705 |
| thin/thick b4 degrees0 | 0.785306 | 0.490813 | 0.291867 | 0.031744 |
| thin/thick b8 degrees5 | 0.783087 | 0.490897 | 0.201156 | 0.016549 |

结论：

- `label1` 拆分是有效方向：`b4 degrees0` 的加权 `label1 mAP85/mAP90/mAP95` 均高于原 `label1` baseline。
- `degrees=5` 对倾斜样本和 `label3/label5` 有帮助，但对 `label1_thin` 的高 IoU 定位损伤太大。
- 当前主要问题不是漏检，两个可视化结果 `missing predictions` 都是 0；瓶颈是框边界、角度规则和 thin/thick 边界定义在高 IoU 下不稳定。

### 9.4 yolo11l-obb 50 轮 after_20260121210219803 子集训练

数据集：

```text
datasets/154843_after_20260121210219803_label1_thin_thick_train_test
```

该数据集从原 AnyLabeling 标注中排除 `20260121210219803` 时间戳本身及之前样本，只保留其后的 273 张图，再转换为 OBB，并继续把 `label1` 拆成 `label1_thin/label1_thick`。

```text
train: 216 images, 216 labels
test:   57 images, 57 labels
```

配置：

```text
model: yolo11l-obb.pt
epochs: 50
imgsz: 1280
batch: 8
device: 0
degrees: 0.0
val: test
```

`custom_metrics.csv` 结果：

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.999303 | 1.000000 | 0.995000 | 0.930040 | 0.841242 | 0.700215 | 0.216485 |
| label1_thin | 0.998076 | 1.000000 | 0.995000 | 0.850524 | 0.605308 | 0.469059 | 0.015000 |
| label1_thick | 0.997043 | 1.000000 | 0.995000 | 0.764306 | 0.509383 | 0.367969 | 0.047500 |
| label2 | 1.000000 | 1.000000 | 0.995000 | 0.995000 | 0.949898 | 0.836529 | 0.256333 |
| label3 | 1.000000 | 1.000000 | 0.995000 | 0.939410 | 0.883192 | 0.600875 | 0.113356 |
| label4 | 1.000000 | 1.000000 | 0.995000 | 0.995000 | 0.995000 | 0.942428 | 0.583023 |
| label5 | 1.000000 | 1.000000 | 0.995000 | 0.981038 | 0.960912 | 0.729835 | 0.173103 |
| label6 | 1.000000 | 1.000000 | 0.995000 | 0.985000 | 0.985000 | 0.954808 | 0.327079 |

IoU85 可视化输出：

```text
images processed: 57
overlays: /home/tjr/Object_detection_YOLO11_OBB/runs/analysis/iou85_overlay/all
failed overlays: /home/tjr/Object_detection_YOLO11_OBB/runs/analysis/iou85_overlay/failed_iou85
matches: /home/tjr/Object_detection_YOLO11_OBB/runs/analysis/iou85_overlay/matches.csv
```

与完整 thin/thick baseline 相比：

| 实验 | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| full b4 degrees0 | 0.993263 | 1.000000 | 0.994857 | 0.917490 | 0.805687 | 0.618882 | 0.195674 |
| after212102 b8 degrees0 | 0.999303 | 1.000000 | 0.995000 | 0.930040 | 0.841242 | 0.700215 | 0.216485 |
| after212102 - full b4 | +0.006040 | +0.000000 | +0.000143 | +0.012550 | +0.035555 | +0.081333 | +0.020811 |

按 test 集数量加权合并 `label1_thin/thick` 后：

| 实验 | weighted label1 mAP80 | weighted label1 mAP85 | weighted label1 mAP90 | weighted label1 mAP95 |
| --- | ---: | ---: | ---: | ---: |
| 原 label1 baseline | 0.792929 | 0.427479 | 0.283031 | 0.008705 |
| thin/thick b4 degrees0 | 0.785306 | 0.490813 | 0.291867 | 0.031744 |
| after212102 b8 degrees0 | 0.817247 | 0.568284 | 0.430042 | 0.027544 |

结论：

- 这轮是目前最有价值的方向：整体 `mAP90` 达到 `0.700215`，高于完整 thin/thick baseline 的 `0.618882`。
- `label1_thin` 和 `label1_thick` 的严格 IoU 都明显改善，尤其 `label1_thick mAP90` 从 `0.143500` 提升到 `0.367969`。
- 加权后的 `label1 mAP90` 从原始 `label1` baseline 的 `0.283031` 提升到 `0.430042`，说明重新截取数据源比继续调增强参数更接近根因。
- `label3 mAP85/mAP90` 仍偏低，下一步需要结合本轮 `matches.csv` 看失败样本是否集中在少量角度或边界不一致样本。

### 9.5 yolo11l-obb 50 轮 after_20260121210219803 去除 -1 样本

数据集：

```text
datasets/154843_after_20260121210219803_no_index1_label1_thin_thick_train_test
```

该数据集在 `after_20260121210219803` 子集基础上，额外去掉文件名末尾 index 为 `-1` 的样本。这样做的原因是部分 `-1` 样本在原 AnyLabeling JSON 中被写成 `label1`，但从图像语义看更接近已排除的 `other` 大框，会污染 `label1_thick`。

```text
selected json files: 251
excluded json files: 22
train: 198 images, 198 labels
test:   53 images, 53 labels
```

类别分布：

| split | objects | label1_thin | label1_thick | label2 | label3 | label4 | label5 | label6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1188 | 132 | 66 | 198 | 198 | 198 | 198 | 198 |
| test | 318 | 35 | 18 | 53 | 53 | 53 | 53 | 53 |

配置：

```text
model: yolo11l-obb.pt
epochs: 50
imgsz: 1280
batch: 8
device: 0
degrees: 0.0
val: test
```

`custom_metrics.csv` 结果：

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

IoU85 可视化输出：

```text
images processed: 53
overlays: /home/tjr/Object_detection_YOLO11_OBB/runs/analysis/iou85_overlay_yolo11l_after212102_no_index1_label1_thin_thick_e50_img1280_b8_deg0_valtest/all
failed overlays: /home/tjr/Object_detection_YOLO11_OBB/runs/analysis/iou85_overlay_yolo11l_after212102_no_index1_label1_thin_thick_e50_img1280_b8_deg0_valtest/failed_iou85
matches: /home/tjr/Object_detection_YOLO11_OBB/runs/analysis/iou85_overlay_yolo11l_after212102_no_index1_label1_thin_thick_e50_img1280_b8_deg0_valtest/matches.csv
```

与 `after212102` 未去除 `-1` 的结果相比：

| 实验 | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| after212102 b8 degrees0 | 0.999303 | 1.000000 | 0.995000 | 0.930040 | 0.841242 | 0.700215 | 0.216485 |
| after212102 no-index1 b8 degrees0 | 0.997159 | 0.999863 | 0.995000 | 0.962807 | 0.894190 | 0.711124 | 0.183112 |
| no-index1 - after212102 | -0.002144 | -0.000137 | +0.000000 | +0.032767 | +0.052948 | +0.010909 | -0.033373 |

按 test 集数量加权合并 `label1_thin/thick` 后：

| 实验 | weighted label1 mAP80 | weighted label1 mAP85 | weighted label1 mAP90 | weighted label1 mAP95 |
| --- | ---: | ---: | ---: | ---: |
| 原 label1 baseline | 0.792929 | 0.427479 | 0.283031 | 0.008705 |
| thin/thick b4 degrees0 | 0.785306 | 0.490813 | 0.291867 | 0.031744 |
| after212102 b8 degrees0 | 0.817247 | 0.568284 | 0.430042 | 0.027544 |
| after212102 no-index1 b8 degrees0 | 0.918529 | 0.691935 | 0.440240 | 0.009434 |

结论：

- 去掉 `-1` 样本是有效的：整体 `mAP80/mAP85/mAP90` 分别提升到 `0.962807/0.894190/0.711124`，说明原先的 `-1` 大框确实在干扰定位。
- `label1_thick` 明显受益，`mAP80/mAP85/mAP90` 从 `0.764306/0.509383/0.367969` 提升到 `0.875694/0.775000/0.643462`。
- `label1_thin mAP90` 从 `0.469059` 降到 `0.335726`，说明拆分后 thin 类在高 IoU 阈值下仍不稳定；这可能来自样本少、边界定义细、或 thin/thick 分界规则不够贴合视觉语义。
- 同一 cleaned 数据集上的未拆分 `label1` 对照见 9.6；对照结果显示，单一 `label1` 的整体 `all` 指标更高，但 `label1` 自身的 `mAP85/mAP90` 和 IoU85 通过率更差。

### 9.6 yolo11l-obb 50 轮 no-index1 单一 label1 对照

数据集：

```text
datasets/154843_after_20260121210219803_no_index1_label1_6_train_test
```

该数据集与 9.5 使用同一个 no-index1 数据源和同一组 train/test 图片，只是不再把 `label1` 拆成 `label1_thin/label1_thick`，而是保留单一 `label1`。原 `other/label7` 仍过滤，不参与训练和评估。

```text
train: 198 images, 198 labels
test:   53 images, 53 labels
```

类别分布：

| split | objects | label1 | label2 | label3 | label4 | label5 | label6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 1188 | 198 | 198 | 198 | 198 | 198 | 198 |
| test | 318 | 53 | 53 | 53 | 53 | 53 | 53 |

配置：

```text
model: yolo11l-obb.pt
epochs: 50
imgsz: 1280
batch: 8
device: 0
degrees: 0.0
val: test
```

`custom_metrics.csv` 结果：

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.999845 | 1.000000 | 0.995000 | 0.978489 | 0.896184 | 0.736873 | 0.220707 |
| label1 | 0.999072 | 1.000000 | 0.995000 | 0.937068 | 0.579722 | 0.349609 | 0.000405 |
| label2 | 1.000000 | 1.000000 | 0.995000 | 0.995000 | 0.985000 | 0.778777 | 0.294268 |
| label3 | 1.000000 | 1.000000 | 0.995000 | 0.981038 | 0.949271 | 0.625321 | 0.104852 |
| label4 | 1.000000 | 1.000000 | 0.995000 | 0.995000 | 0.995000 | 0.969906 | 0.534817 |
| label5 | 1.000000 | 1.000000 | 0.995000 | 0.995000 | 0.900283 | 0.806119 | 0.188774 |
| label6 | 1.000000 | 1.000000 | 0.995000 | 0.967830 | 0.967830 | 0.891509 | 0.201127 |

IoU85 可视化输出：

```text
analysis: runs/analysis/iou85_overlay_yolo11l_after212102_no_index1_label1_6_e50_img1280_b8_deg0_valtest
GT rows: 318
failed rows: 90
failed images: 44
missing predictions: 0
```

| class | GT | IoU85 failed | failed rate | mean IoU | median IoU | min IoU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| label1 | 53 | 36 | 67.9% | 0.795 | 0.817 | 0.518 |
| label2 | 53 | 10 | 18.9% | 0.891 | 0.900 | 0.758 |
| label3 | 53 | 16 | 30.2% | 0.869 | 0.888 | 0.523 |
| label4 | 53 | 2 | 3.8% | 0.919 | 0.928 | 0.821 |
| label5 | 53 | 14 | 26.4% | 0.884 | 0.900 | 0.719 |
| label6 | 53 | 12 | 22.6% | 0.887 | 0.887 | 0.682 |

与 no-index1 thin/thick 结果相比：

| 实验 | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 | IoU85 failed rows | failed images |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| no-index1 thin/thick | 0.997159 | 0.999863 | 0.995000 | 0.962807 | 0.894190 | 0.711124 | 0.183112 | 84 | 41 |
| no-index1 single label1 | 0.999845 | 1.000000 | 0.995000 | 0.978489 | 0.896184 | 0.736873 | 0.220707 | 90 | 44 |
| single - thin/thick | +0.002686 | +0.000137 | +0.000000 | +0.015682 | +0.001994 | +0.025749 | +0.037595 | +6 | +3 |

`label1` 单独对比：

| 实验 | label1 mAP80 | label1 mAP85 | label1 mAP90 | label1 mAP95 | label1 IoU85 failed |
| --- | ---: | ---: | ---: | ---: | ---: |
| no-index1 thin/thick 加权合并 | 0.918529 | 0.691935 | 0.440240 | 0.009434 | 29/53 |
| no-index1 single label1 | 0.937068 | 0.579722 | 0.349609 | 0.000405 | 36/53 |
| single - thin/thick | +0.018539 | -0.112213 | -0.090631 | -0.009029 | +7/53 |

逐图对齐结论：

```text
thin/thick 过 IoU85、single label1 未过: 8 images
single label1 过 IoU85、thin/thick 未过: 1 image
```

角度段统计：

| GT top-edge angle | GT | IoU85 failed | failed rate | mean IoU |
| --- | ---: | ---: | ---: | ---: |
| 0-2 deg | 301 | 79 | 26.2% | 0.877 |
| 2-5 deg | 5 | 1 | 20.0% | 0.896 |
| 5-10 deg | 6 | 5 | 83.3% | 0.786 |
| 10+ deg | 6 | 5 | 83.3% | 0.810 |

结论：

- 单一 `label1` 的整体 `all mAP80/mAP90/mAP95` 更高，但这主要来自其他类别波动，不能说明 `label1` 合并更好。
- 对当前最关注的 `label1` 高 IoU 定位，thin/thick 更好：`mAP85/mAP90` 分别高 `0.112213/0.090631`，IoU85 失败数少 7 个。
- 单一 `label1` 没有解决主要瓶颈；失败仍是检测到了但边界不够贴合，`missing predictions` 仍为 0。
- 5 度以上倾斜样本仍是主要硬点，12 个 GT 中失败 10 个，说明瓶颈是角度和边界定位，而不是 thin/thick 分类本身。
- 建议训练阶段保留 `label1_thin/label1_thick`。如果最终业务只需要输出单一 `label1`，更合理的方案是训练时保留 thin/thick，推理或评估后把 `label1_thin/label1_thick` 合并为 `label1`。

### 9.7 已停止的 deskew / fusion 路线

deskew 和双模型预测级融合的历史数据、权重与结果仍保留在 `archive/` 或 `runs/`，但不再作为工程方案维护。原因是线上部署需要两个 OBB 模型并行推理，成本和复杂度不符合当前工程约束。

后续模型选型统一采用单模型、相同数据划分、相同训练配置和相同 Ultralytics OBB 指标（重点比较 `mAP90`）。首个对照为 `YOLO11n-OBB`，原 `YOLO11l-OBB` baseline 权重和运行目录均保留，作为参照，不会被覆盖。

## 10. 筛选后数据统计

统计口径：

```text
source: 已打标的数据202604/user1_2026-03-16_154843_anylabeling
after_stem: 20260121210219803
exclude_indices: 1
selected_json_files: 251
```

即只统计 `20260121210219803` 之后的数据，并排除文件名末尾为 `-1` 的样本。该口径与当前主线 no-index1 训练集一致。

统计输出：

```text
runs/analysis/description_distribution_after_20260121210219803_no_index1
```

### 10.1 label1-label6 原始标签分布

原始 AnyLabeling JSON 中，`label1-label6` 完全均衡：

| label | shapes |
| --- | ---: |
| label1 | 251 |
| label2 | 251 |
| label3 | 251 |
| label4 | 251 |
| label5 | 251 |
| label6 | 251 |
| label7 | 2 |
| lable7 | 1 |

`label7/lable7` 是原始 JSON 中残留的排除类，不参与训练和评估。

当前主线 thin/thick 训练数据分布：

| split | label1_thin | label1_thick | label2 | label3 | label4 | label5 | label6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 132 | 66 | 198 | 198 | 198 | 198 | 198 |
| test | 35 | 18 | 53 | 53 | 53 | 53 | 53 |
| total | 167 | 84 | 251 | 251 | 251 | 251 | 251 |

如果把 `label1_thin/label1_thick` 合并回单一 `label1`，训练数据仍然均衡：

| split | label1 | label2 | label3 | label4 | label5 | label6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 198 | 198 | 198 | 198 | 198 | 198 |
| test | 53 | 53 | 53 | 53 | 53 | 53 |
| total | 251 | 251 | 251 | 251 | 251 | 251 |

### 10.2 description 分类分布

`description` 来自原始 AnyLabeling JSON 中每个 shape 的 `flags.description`。本表按原始 `label1-label6` 统计，`label1` 不拆 thin/thick。`EMPTY` 代表 description 为空，已计入总数。

| label | total_shapes | category_sum | missing_from_sum | EMPTY | OK | OK_B | OK_G | OK_R | OK_W | NG_芯线露出多 | NG_芯线露出少 | NG_芯线未露出 | NG_铜线露出多 | NG_铜线露出少 | NG_铜线飞出 | NG_OTHER | OTHER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| label1 | 251 | 251 | 0 | 1 | 4 | 126 | 43 | 40 | 37 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| label2 | 251 | 251 | 0 | 1 | 250 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| label3 | 251 | 251 | 0 | 1 | 85 | 0 | 0 | 0 | 0 | 162 | 1 | 0 | 2 | 0 | 0 | 0 | 0 |
| label4 | 251 | 251 | 0 | 1 | 249 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| label5 | 251 | 251 | 0 | 1 | 97 | 0 | 0 | 0 | 0 | 0 | 141 | 3 | 0 | 2 | 6 | 1 | 0 |
| label6 | 251 | 251 | 0 | 5 | 243 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 0 |
| label7 | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 | 0 |
| lable7 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |

校验结果：所有 label 的 `missing_from_sum` 都是 `0`，说明 `EMPTY` 和其它 description 分类相加后没有遗漏。

空 description 明细：

| label | empty descriptions |
| --- | ---: |
| label1 | 1 |
| label2 | 1 |
| label3 | 1 |
| label4 | 1 |
| label5 | 1 |
| label6 | 5 |
| label7 | 0 |
| lable7 | 0 |

`label4` 中唯一的 `OTHER` 来自：

```text
CropImage_20260128140200491_F3-I0_OK-4.json
label: label4
description: 0空气能
```

该项应视为 description 误填，建议后续修正为 `OK` 或正确 NG 描述。

## 11. 训练日志归档

旧训练日志已经迁移到：

```text
docs/training_logs/legacy_labelme_training_log.md
docs/training_logs/anylabeling_label1_6_training_log.md
```

其中：

- `legacy_labelme_training_log.md` 基于更早的 LabelMe 数据集。
- `anylabeling_label1_6_training_log.md` 基于未拆分 `label1` 的 AnyLabeling `label1-label6` 数据集。

当前 README 主线只记录 `label1_thin/label1_thick` 数据集实验。
