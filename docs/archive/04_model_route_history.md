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

## 4. RHINO checkpoint 对比

所有结果使用相同 test split 和统一 Ultralytics OBB 指标：

### 4.1 KLD 与 baseline

| model | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11l-OBB | 0.997159 | 0.999863 | 0.995000 | 0.962807 | 0.894190 | 0.711124 | 0.183112 |
| RHINO KLD epoch40 | 0.991563 | 0.998104 | 0.994762 | 0.923683 | 0.830400 | 0.648533 | 0.211504 |
| RHINO KLD epoch50 | 0.992294 | 0.994560 | 0.994921 | 0.927507 | 0.828269 | 0.623374 | 0.226096 |

- KLD epoch40 的 mAP90 比 KLD epoch50 高 0.025159，因此 KLD 保留 epoch40 作为高 IoU 代表。
- 两个 KLD checkpoint 的整体 mAP90 都没有超过 YOLO11l。

### 4.2 RIoU 与 baseline

| model | precision | recall | mAP50 | mAP80 | mAP85 | mAP90 | mAP95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLO11l-OBB | 0.997159 | 0.999863 | 0.995000 | 0.962807 | 0.894190 | 0.711124 | 0.183112 |
| RHINO RIoU AP50-best | 0.985410 | 0.995654 | 0.995000 | 0.911655 | 0.788412 | 0.596849 | 0.163323 |
| RHINO RIoU epoch50 | 0.992466 | 0.991982 | 0.994948 | 0.959565 | 0.878536 | 0.659244 | 0.264438 |

- RIoU epoch50 的 mAP90 比其 AP50-best 高 0.062395，mAP95 高 0.101115，因此 RIoU 保留 epoch50 作为高 IoU 代表。
- RIoU epoch50 是当前最好的 RHINO checkpoint，但 mAP90 仍比 YOLO11l 低 0.051880。
- RIoU epoch50 的 mAP95 比 YOLO11l 高 0.081326，说明其在一部分目标上能产生更严格贴合的框，但类别稳定性和整体 mAP90 仍不足。

### 4.3 checkpoint 选择结论

- `best_dota_mAP` 只代表 AP50 最佳。后续 RHINO 训练需要同时评估最后若干 epoch，不能用该文件名直接决定最终权重。

## 5. 当前模型选择原则

1. 所有模型使用相同 train/test 图像和类别顺序。
2. 最终比较只使用 Ultralytics OBB 统一口径。
3. 以整体 mAP90 为主要指标，同时检查每个类别是否出现严重退化。
4. mAP90 接近时，再比较 mAP85、mAP95、速度、显存和部署依赖。
5. 优先选择单模型方案；双模型融合不再作为当前工程候选。
