# ResNet18 分类实验记录

本文归档下游 OK/NG 分类实验。当前 pipeline 说明位于根目录 [README](../../README.md) 的最后一章。

## 1. 分类流程

当前分类器处理 `label3` 和 `label5`：

```text
OBB 框
  -> 按类别提取目标
  -> 四点透视矫正并裁剪
  -> ResNet18
  -> OK / NG
```

分类 GT 来自 `outputs/label1_6_description.xlsx` 对应类别 sheet 的 `tag1`。分类结果与分类数据集 test split 的 `tag1` 比较，不再与 OBB 检测框计算 IoU。

## 2. 数据集

首轮数据使用 AnyLabeling 人工 OBB 框裁剪，按 parent group 做 8:2 划分：

| label | dataset | total | train | test |
| --- | --- | ---: | --- | --- |
| label3 | `datasets/classification/label3_ok_ng` | 251 | NG=129, OK=69 | NG=36, OK=17 |
| label5 | `datasets/classification/label5_ok_ng` | 251 | NG=118, OK=80 | NG=35, OK=18 |

## 3. 训练配置

| label | epochs | batch | imgsz | lr | weight decay | pretrained | device |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| label3 | 30 | 8 | 224 | 0.0001 | 0.0001 | true | CPU |
| label5 | 30 | 8 | 224 | 0.0001 | 0.0001 | true | MPS |

训练、评测和预测产物统一写入 `runs/classification/<run_name>/`，权重固定保存在 `weights/best.pt` 和 `weights/last.pt`。

## 4. 独立评测结果

| label | best epoch | test accuracy | test macro F1 |
| --- | ---: | ---: | ---: |
| label3 | 1 | 0.981132 | 0.977999 |
| label5 | 2 | 1.000000 | 1.000000 |

label3 的 best checkpoint 混淆矩阵：

```text
true\pred,NG,OK
NG,36,0
OK,1,16
```

label5 的 best checkpoint 混淆矩阵：

```text
true\pred,NG,OK
NG,35,0
OK,0,18
```

label3 `last.pt` 最后一轮为 `accuracy=0.962264`、`macro F1=0.955236`；表中数值是加载 `best.pt` 后的独立复评结果。label5 的 `best.pt` 与 `last.pt` 在该 test split 上均为 1.0。

## 5. 输出约定

```text
runs/classification/<run_name>/
  weights/best.pt
  weights/last.pt
  train_*.csv
  train_args.yaml
  eval_*.csv
  eval_args.yaml
  predict_*.csv
  predict_args.yaml
```

完整 OBB -> ResNet pipeline 的端到端结果存放在 `runs/pipeline_eval/<run_name>/`，与仅使用人工 GT 框裁剪的分类器结果分开。

## 6. 局限

- 首轮只有 train/test，没有独立 validation split。
- `best.pt` 依据 test macro F1 选择，因此当前结果偏乐观，只适合验证分类流程可行性。
- 上表使用人工 OBB 框裁剪，尚未包含检测框角度和边界误差。
- 真正工程指标应使用上游模型预测框完成裁剪，再在固定端到端 test split 上统计分类结果。
