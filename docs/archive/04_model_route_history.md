# deskew、fusion 与 RHINO 路线记录

本文记录 YOLO11l 基线之后尝试过的模型路线。当前模型选择结论以根目录 [README](../../README.md) 为准。

## 1. deskew 路线

deskew 数据集尝试根据工件方向统一图像或标注坐标，目标是降低正放与倾斜样本混合导致的角度回归波动。

该路线没有成为当前主线，原因包括：

- 不同类别和不同工件的参考方向并不总能由同一规则稳定定义。
- 图像、OBB 标注和预测结果之间需要额外坐标变换及逆变换。
- 部分类别改善时，另一些类别会下降，整体 mAP90 没有稳定超过原图基线。
- 新增的数据维护成本和实际部署步骤较多。

deskew 数据和权重只作为历史实验归档，不应与当前 `obb_thin_thick` 原图数据混用。

## 2. 双模型 fusion 路线

历史上曾融合原图 YOLO11l 与 deskew YOLO11l 的预测。统一为 Ultralytics OBB 口径后，较好的一次融合结果为：

| mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| ---: | ---: | ---: | ---: | ---: |
| 0.995000 | 0.952791 | 0.900863 | 0.747498 | 0.246202 |

该结果说明互补预测能够提高部分高 IoU 指标，但工程方案需要：

- 同时维护两个检测模型。
- 对同一图像执行两次推理。
- 处理 deskew 和原图坐标映射。
- 增加融合阈值、匹配策略和异常情况。

由于实际部署对算力和流程复杂度敏感，该路线已停止。融合结果仍可作为算法上限参考，但不参与当前单模型选型。

## 3. RHINO 单模型路线

停止 fusion 后，模型选择转向单模型 Transformer 检测器。当前候选为 RHINO R50：

- KLD：使用 RHINO 官方 `GDLoss/GDCost(KLD)` 回归和匹配配置。
- RIoU：保持 R50、Hausdorff matching 和其他参数一致，仅替换为 `RotatedIoULoss/RotatedIoUCost`，用于受控消融。

训练阶段 RHINO 原生 DOTA mAP@0.5 只用于观察收敛和选择候选 checkpoint。正式对比需完成：

```text
.pth checkpoint
  -> test predictions.pkl
  -> OBB labels/*.txt
  -> Ultralytics OBB 固定阈值 mAP
```

这样可避免把 RHINO 的原生 AP50 与 YOLO 的 mAP80/mAP85/mAP90/mAP95 混为一谈。

## 4. RHINO KLD 首轮结论

RHINO R50-KLD epoch40 在当前 test split 上：

| model | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: |
| YOLO11l-OBB | 0.962807 | 0.894190 | 0.711124 | 0.183112 |
| RHINO R50-KLD epoch40 | 0.923683 | 0.830400 | 0.648533 | 0.211504 |

RHINO 改善了 `label1_thin`、`label3` 和 `label5` 的 mAP90，也提高了整体 mAP95；但 `label1_thick` mAP90 从 0.643462 降至 0.082473，导致整体 mAP90 低于 YOLO11l。

epoch40 是按训练阶段 AP50 选出的权重，不一定是 mAP90 最佳 checkpoint。因此 `epoch_50.pth` 仍需要按同一流程评估，才能确认 checkpoint 选择是否影响结论。

## 5. 当前模型选择原则

1. 所有模型使用相同 train/test 图像和类别顺序。
2. 最终比较只使用 Ultralytics OBB 统一口径。
3. 以整体 mAP90 为主要指标，同时检查每个类别是否出现严重退化。
4. mAP90 接近时，再比较 mAP85、mAP95、速度、显存和部署依赖。
5. 优先选择单模型方案；双模型融合不再作为当前工程候选。
