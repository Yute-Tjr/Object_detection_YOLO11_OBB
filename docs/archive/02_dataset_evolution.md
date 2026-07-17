# 数据集清理与版本演进

本文记录数据集从原始 AnyLabeling 标注到当前检测基线数据的演进。当前训练统一使用 `datasets/obb_thin_thick`。

## 1. 原始标注口径

原始数据来自 AnyLabeling JSON。有效检测类别为 `label1-label6`，每张图通常同时包含多个类别框；`other/label7` 不参与当前训练。

因此图像数和 GT 框数并不相等。例如当前 test split 有 53 张图，但每张图包含 `label1` 和 `label2-label6` 共 6 个目标，所以总实例数为：

```text
53 images x 6 objects = 318 GT instances
```

原始候选数据共有 251 张图并不意味着只有 251 个 GT 框。

## 2. label1 thin/thick 拆分

为减少不同宽度工件共用一个类别造成的边界回归波动，原 `label1` 按标注框上沿宽度拆分：

- `label1_thin`：上沿宽度 `< 164 px`
- `label1_thick`：上沿宽度 `>= 164 px`

其余类别保持为 `label2-label6`。该拆分只改变类别标签，不旋转图像，也不自动重新摆正标注框。

## 3. 数据版本演进

### 3.1 完整 thin/thick 数据

最初完整版本包含：

| split | images | objects | label1_thin | label1_thick | label2-label6 每类 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 342 | 1884 | 209 | 120 | 311 |
| test | 87 | 478 | 54 | 29 | 79 |

该版本包含较早的数据来源，标注边界和工件姿态一致性较弱。

### 3.2 时间筛选版本

仅保留 `after_20260121210219803` 数据后：

- 训练图像：216 张
- 测试图像：57 张

该步骤显著提高高 IoU 指标，说明数据源和标注批次的一致性比继续增强更关键。

### 3.3 去除 index=-1 的当前版本

在时间筛选版本上进一步删除文件名末尾 index 为 `-1` 的 22 张样本。这些样本中的大框语义更接近已排除的 `other`，会污染 `label1_thick`。

当前 `datasets/obb_thin_thick` 统计如下：

| split | images | objects | label1_thin | label1_thick | label2 | label3 | label4 | label5 | label6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 198 | 1188 | 132 | 66 | 198 | 198 | 198 | 198 | 198 |
| test | 53 | 318 | 35 | 18 | 53 | 53 | 53 | 53 | 53 |

`data.yaml` 的 `val` 指向 test split，因此当前实验没有独立 validation split。所有模型对比必须使用完全相同的 53 张测试图。

## 4. RHINO 格式数据

`datasets/rhino_obb` 是 `datasets/obb_thin_thick` 的格式转换副本：

- 图像划分不变：train 198 张、test 53 张。
- 实例数不变：train 1188 个、test 318 个。
- 类别顺序不变。
- 图像转换为 PNG，标注转换为 DOTA 四点文本。

这不是新采样的数据集，因此 RHINO 和 YOLO 的最终指标可以在同一 test split 上直接比较。

## 5. 阶段结论

- 当前主要提升来自清理标注语义不一致的数据，而不是加大旋转增强。
- `label1_thick` 的 test 样本只有 18 个，单个样本就会明显影响该类高 IoU AP，跨 seed 稳定性仍需关注。
- thin/thick 拆分类别改善了 label1 在 mAP85/mAP90 下的定位，但没有解决极严格 mAP95 的边界敏感问题。
- 后续数据变更必须保留固定 test 清单，否则不同模型的 mAP 对比失去意义。
