# 代码架构与兼容策略

本项目最初只有 YOLO11-OBB，因此 Python 包命名为 `yolo11_obb`。随着 RHINO、分类器、数据转换和端到端流水线加入，这个名称已经不能准确表达职责。当前采用渐进式迁移：新功能进入模型无关的 `obb_detection` 包，旧包继续提供兼容入口。

## 当前正式结构

```text
obb_detection/
  common/                         # 模型共用的预测转换等能力
  models/
    oriented_rcnn/                # Oriented R-CNN 配置与集成

scripts/
  common/                         # 模型无关命令
  models/
    yolo11/                       # YOLO11 分类入口
    rhino/                        # RHINO 分类入口
    oriented_rcnn/                # Oriented R-CNN 训练、预测、评测

yolo11_obb/                       # 旧代码兼容包，暂不删除
tests/                            # 单元测试
datasets/                         # 数据集，不随代码迁移
runs/                             # 训练产物，不随代码迁移
```

`scripts/models/yolo11` 和 `scripts/models/rhino` 当前是对旧命令的薄包装，因此新旧路径行为一致。后续可以逐模块把实现迁入 `obb_detection`，而不需要一次性修改所有历史命令。

## Oriented R-CNN 入口

训练入口：

```bash
CUDA_VISIBLE_DEVICES=0 python3 scripts/train_oriented_rcnn.py \
  --rhino-root ~/RHINO \
  --data datasets/rhino_obb \
  --imgsz 1280 \
  --epochs 50 \
  --batch 2 \
  --workers 4 \
  --project runs/oriented_rcnn \
  --name oriented_rcnn_r50_fpn_e50_img1280_b2 \
  --dry-run
```

确认生成的配置和命令正确后移除 `--dry-run`。默认继承 MMRotate 1.x：

```text
configs/oriented_rcnn/oriented-rcnn-le90_r50_fpn_1x_dota.py
```

如果外部仓库的位置不同，可用 `--base-config` 指定配置。训练输出保存到：

```text
runs/oriented_rcnn/<run_name>/
  config.py
  train_command.txt
  epoch_*.pth
  best_dota_mAP_epoch_*.pth
```

统一高 IoU 评测入口：

```bash
python3 scripts/evaluate_oriented_rcnn.py \
  --rhino-root ~/RHINO \
  --config runs/oriented_rcnn/oriented_rcnn_r50_fpn_e50_img1280_b2/config.py \
  --weights runs/oriented_rcnn/oriented_rcnn_r50_fpn_e50_img1280_b2/epoch_50.pth \
  --data datasets/obb_thin_thick/data.yaml \
  --split test \
  --run-dir runs/oriented_rcnn/oriented_rcnn_r50_fpn_e50_img1280_b2/test_epoch50 \
  --mmrotate-python /path/to/mmrotate-env/bin/python \
  --metric-python /path/to/yolo-env/bin/python
```

评测链路为：

```text
MMRotate test.py
  -> predictions.pkl
  -> labels/*.txt（YOLO OBB 四点 + confidence）
  -> Ultralytics batch_probiou
  -> custom_metrics.csv
```

训练阶段保存的 `best_dota_mAP` 仍只代表 AP50 最佳。为了选择 mAP90 checkpoint，需要用上述统一评测命令检查候选 epoch。

## 对历史训练的影响

本次整理不会改变之前训练的数值或权重：

- 没有移动或改写 `runs/`、`.pt`、`.pth`、数据集和历史生成配置。
- 原有 `scripts/train_yolo11_obb.py`、`scripts/train_rhino.py` 等命令仍然可用。
- `yolo11_obb` 的历史 import 路径仍然存在，测试与旧脚本无需立即修改。
- 已完成的 YOLO 和 RHINO 权重加载不依赖新的目录结构。
- 已经启动的外部服务器训练进程不会读取本地后续代码修改，因此不受影响。

唯一有意收紧的行为是 MMRotate 预测导出：现在会确认预测 pickle 覆盖所选 split 的每一张图片。如果预测不完整会直接报错，避免旧标签残留造成虚假的 mAP90。

## 后续迁移规则

- 新模型放到 `obb_detection/models/<model>/`。
- 多模型共用代码放到 `obb_detection/common/`。
- 命令入口按 `scripts/models/<model>/` 分类。
- 旧顶层命令至少保留一个完整实验周期，再考虑删除。
- 不通过重构移动 `datasets/` 或 `runs/`，保证历史实验可追溯。
