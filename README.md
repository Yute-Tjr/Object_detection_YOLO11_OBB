# YOLO11-OBB Training

## 1. 项目数据集:

```text
datasets/154843_obb_converted_label1_6_train_test
```

## 2. 数据集配置:

```text
datasets/154843_obb_converted_label1_6_train_test/data.yaml
```

## 3. 依赖下载

```bash
python3 -m pip install -r requirements.txt
```

## 4. 检查数据

只检查数据完整性，不启动训练

```bash
python3 scripts/train_yolo11_obb.py \
  --data datasets/154843_obb_converted_label1_6_train_test/data.yaml \
  --no-val \
  --dry-run
```

## 5. Train

```bash
CUDA_VISIBLE_DEVICES=0 python3 -u scripts/train_yolo11_obb.py \
  --model yolo11m-obb.pt \
  --data datasets/154843_obb_converted_label1_6_train_test/data.yaml \
  --imgsz 1024 \
  --epochs 30 \
  --batch 16 \
  --device 0 \
  --workers 8 \
  --name yolo11m_154843_converted_label1_6_e30_train_test \
  --no-val
```

当前 30 轮实验权重

```text
runs/obb/yolo11m_154843_converted_label1_6_e30_train_test/weights/best.pt
```

## 6. Evaluate

使用test数据集进行评估

```bash
python3 scripts/evaluate_yolo11_obb.py \
  --data datasets/154843_obb_converted_label1_6_train_test/data.yaml \
  --model runs/obb/yolo11m_154843_converted_label1_6_e30_train_test/weights/best.pt \
  --split test \
  --imgsz 1024 \
  --batch 16 \
  --device 0 \
  --name yolo11m_154843_converted_label1_6_e30_test_eval
```

## 7. Predict

```bash
python3 scripts/predict_yolo11_obb.py \
  --model runs/obb/yolo11m_154843_converted_label1_6_e30_train_test/weights/best.pt \
  --source datasets/154843_obb_converted_label1_6_train_test/images/test
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

该数据集没有单独的 `images/val` 和 `labels/val` 目录。`data.yaml` 中的 `val` 指向 `images/test`，只用于兼容 Ultralytics 数据配置；训练时使用 `--no-val` 关闭训练期间验证，训练结束后再单独对 `test` 做 evaluate。

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

评估脚本会在 evaluate 输出目录中额外写入：

```text
custom_metrics.csv
```

该文件只包含当前关注的指标：

```text
class,precision,recall,mAP50,mAP80,mAP85,mAP90,mAP95
```

`custom_metrics.csv` 不输出 `mAP50-95`。

## 9. AnyLabeling 数据集训练日志

当前日志基于新的 AnyLabeling OBB 数据集，只保留 `label1-label6`，按 `train:test = 8:2` 划分。

```text
datasets/154843_obb_converted_label1_6_train_test
```

该数据集没有单独划分 `val`。为了满足 Ultralytics 的数据配置字段，`data.yaml` 中的 `val` 指向 `images/test`；正式指标以训练结束后单独执行的 `test evaluate` 为准。

```text
train: 342 images, 342 labels
test:   87 images, 87 labels
```

当前重点关注指标：`mAP50`、`mAP80`、`mAP85`、`mAP90`、`mAP95`。不再把 `mAP50-95` 作为主要对比指标。

### 9.1 yolo11m-obb 30 轮训练

`custom_metrics.csv` 结果：

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.999066 | 0.997992 | 0.994961 | 0.938099 | 0.866518 | 0.690608 | 0.204737 |
| label1 | 0.998217 | 0.987952 | 0.994765 | 0.753901 | 0.500505 | 0.261939 | 0.009079 |
| label2 | 0.997938 | 1.000000 | 0.995000 | 0.995000 | 0.995000 | 0.891359 | 0.309233 |
| label3 | 1.000000 | 1.000000 | 0.995000 | 0.983734 | 0.954011 | 0.579002 | 0.175633 |
| label4 | 1.000000 | 1.000000 | 0.995000 | 0.995000 | 0.980190 | 0.944519 | 0.365026 |
| label5 | 0.998243 | 1.000000 | 0.995000 | 0.905957 | 0.788579 | 0.561387 | 0.167898 |
| label6 | 1.000000 | 1.000000 | 0.995000 | 0.995000 | 0.980823 | 0.905445 | 0.201553 |

### 9.2 yolo11m-obb 60 轮训练

`custom_metrics.csv` 结果：

| class | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all | 0.995575 | 1.000000 | 0.994821 | 0.941476 | 0.858661 | 0.678999 | 0.228278 |
| label1 | 0.985392 | 1.000000 | 0.993929 | 0.780943 | 0.422103 | 0.169866 | 0.007727 |
| label2 | 0.995776 | 1.000000 | 0.995000 | 0.995000 | 0.995000 | 0.895415 | 0.279622 |
| label3 | 1.000000 | 1.000000 | 0.995000 | 0.979430 | 0.968413 | 0.593519 | 0.206165 |
| label4 | 1.000000 | 1.000000 | 0.995000 | 0.995000 | 0.995000 | 0.933481 | 0.538481 |
| label5 | 0.996449 | 1.000000 | 0.995000 | 0.903485 | 0.786451 | 0.582529 | 0.082702 |
| label6 | 0.995835 | 1.000000 | 0.995000 | 0.995000 | 0.985000 | 0.899187 | 0.254968 |

### 9.3 30 轮与 60 轮对比

| 轮数 | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 30 | 0.999066 | 0.997992 | 0.994961 | 0.938099 | 0.866518 | 0.690608 | 0.204737 |
| 60 | 0.995575 | 1.000000 | 0.994821 | 0.941476 | 0.858661 | 0.678999 | 0.228278 |
| 60-30 | -0.003491 | +0.002008 | -0.000140 | +0.003377 | -0.007857 | -0.011609 | +0.023541 |

结论：

- `mAP50` 基本持平，30 轮略高。
- 60 轮的 `mAP80` 和 `mAP95` 有小幅提升，但 `mAP85`、`mAP90` 下降。
- `label1` 在严格 IoU 指标下仍是主要短板，60 轮的 `label1 mAP85/mAP90/mAP95` 没有改善。
- 当前结果不支持继续单纯增加 epoch。下一步更应该检查 `label1` 的预测图、标注框一致性，或者单独尝试更高输入分辨率。

## 10. 旧训练日志归档

旧的 LabelMe 数据集训练日志已经迁移到：

```text
docs/training_logs/legacy_labelme_training_log.md
```

这些实验基于 `datasets/154843_obb_train_val_test` 和 `datasets/154843_obb_label1_6_train_val_test`，与当前 AnyLabeling train/test 数据集不同。
