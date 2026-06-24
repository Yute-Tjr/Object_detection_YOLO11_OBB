# YOLO11-OBB Training

## 1. 项目数据集:

```text
datasets/154843_obb_converted_label1_thin_thick_train_test
```

## 2. 数据集配置:

```text
datasets/154843_obb_converted_label1_thin_thick_train_test/data.yaml
```

## 3. 依赖下载

```bash
python3 -m pip install -r requirements.txt
```

## 4. 检查数据

只检查数据完整性，不启动训练

```bash
python3 scripts/train_yolo11_obb.py \
  --data datasets/154843_obb_converted_label1_thin_thick_train_test/data.yaml \
  --dry-run
```

## 5. Train

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/train_yolo11_obb.py \
  --model yolo11l-obb.pt \
  --data datasets/154843_obb_converted_label1_thin_thick_train_test/data.yaml \
  --imgsz 1280 \
  --epochs 50 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --name yolo11l_label1_thin_thick_e50_img1280_b8_deg5_valtest \
  --degrees 5.0
```

当前 50 轮实验权重

```text
runs/obb/yolo11l_label1_thin_thick_e50_img1280_b8_deg5_valtest/weights/best.pt
```

## 6. Evaluate

使用test数据集进行评估

```bash
python3 scripts/evaluate_yolo11_obb.py \
  --data datasets/154843_obb_converted_label1_thin_thick_train_test/data.yaml \
  --model runs/obb/yolo11l_label1_thin_thick_e50_img1280_b8_deg5_valtest/weights/best.pt \
  --split test \
  --imgsz 1280 \
  --batch 8 \
  --device 0 \
  --name yolo11l_label1_thin_thick_e50_img1280_b8_deg5_valtest_eval
```

## 7. Predict

```bash
python3 scripts/predict_yolo11_obb.py \
  --model runs/obb/yolo11l_label1_thin_thick_e50_img1280_b8_deg5_valtest/weights/best.pt \
  --source datasets/154843_obb_converted_label1_thin_thick_train_test/images/test
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

## 9. label1_thin/thick 训练日志

当前日志基于 `datasets/154843_obb_converted_label1_thin_thick_train_test`。重点指标仍是 `mAP50`、`mAP80`、`mAP85`、`mAP90`、`mAP95`。

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

### 9.4 下一步调整

优先级：

1. 补跑 `batch=8, degrees=0`，隔离 batch 变化影响。当前 `b4 degrees0` 和 `b8 degrees5` 同时改变了 batch 和旋转增强，不能把差异全部归因于 `degrees`。
2. 跑 `degrees=3`，验证较弱旋转增强是否能保留 5-10 度样本收益，同时减少 `label1_thin` 高 IoU 损伤。
3. 复查 `label1_thin_failed_iou85` 和 `label1_thick_failed_iou85`，重点检查上沿/下沿边界规则是否一致，尤其是 `CropImage_20260121180233350_*`、`CropImage_20260126135006021_*`、`CropImage_20260127201733495_*`。
4. 对明显整体倾斜的母样本，优先统一标注角度规则；如果训练和部署图像都存在稳定倾斜，再考虑预矫正数据集，而不是继续叠加增强。

建议下一轮命令：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/train_yolo11_obb.py \
  --model yolo11l-obb.pt \
  --data datasets/154843_obb_converted_label1_thin_thick_train_test/data.yaml \
  --imgsz 1280 \
  --epochs 50 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --name yolo11l_label1_thin_thick_e50_img1280_b8_deg0_valtest \
  --degrees 0.0
```

随后再跑单变量 `degrees=3.0`：

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/train_yolo11_obb.py \
  --model yolo11l-obb.pt \
  --data datasets/154843_obb_converted_label1_thin_thick_train_test/data.yaml \
  --imgsz 1280 \
  --epochs 50 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --name yolo11l_label1_thin_thick_e50_img1280_b8_deg3_valtest \
  --degrees 3.0
```

## 10. 训练日志归档

旧训练日志已经迁移到：

```text
docs/training_logs/legacy_labelme_training_log.md
docs/training_logs/anylabeling_label1_6_training_log.md
```

其中：

- `legacy_labelme_training_log.md` 基于更早的 LabelMe 数据集。
- `anylabeling_label1_6_training_log.md` 基于未拆分 `label1` 的 AnyLabeling `label1-label6` 数据集。

当前 README 主线只记录 `label1_thin/label1_thick` 数据集实验。
