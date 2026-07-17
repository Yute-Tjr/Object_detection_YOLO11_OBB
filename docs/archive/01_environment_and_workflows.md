# 环境、脚本入口与产物约定

本文归档项目运行环境和脚本职责。当前可直接执行的命令以根目录 [README](../../README.md) 为准。

## 1. 环境隔离

YOLO11 OBB 与 RHINO 建议使用独立 Python 环境：

- YOLO 环境安装 `ultralytics`，负责 YOLO 训练、预测以及统一的 Ultralytics OBB 指标计算。
- RHINO 环境安装 RHINO 官方要求的 MMDetection、MMRotate 和 MMCV 版本，负责 `.pth` 推理和 `.pkl` 预测生成。
- RHINO 依赖较旧，不应直接覆盖可正常运行的 YOLO 环境。

历史上 RHINO 环境因 YAPF API 兼容问题固定过 `yapf==0.40.1`。使用 `--no-deps` 安装时还需要确保 `importlib-metadata` 可用；该约束属于 RHINO 旧依赖链，不是本项目算法逻辑的一部分。

## 2. 当前 YOLO 基线配置

当前基线使用：

```yaml
model: yolo11l-obb.pt
data: datasets/obb_thin_thick/data.yaml
epochs: 50
imgsz: 1280
batch: 8
degrees: 0.0
seed: 42
```

主要入口：

- `scripts/train_yolo11_obb.py`：训练。
- `scripts/evaluate_yolo11_obb.py`：调用 Ultralytics 验证流程。
- `scripts/evaluate_obb_prediction_labels.py`：对已经导出的 OBB 文本预测统一计算固定 IoU 阈值指标。
- `scripts/analyze_obb_iou.py`：生成逐实例匹配、IoU 统计和可视化分析。

正式训练前可以使用脚本支持的 `--dry-run` 检查数据路径和最终参数。

## 3. RHINO 数据与执行链路

RHINO 使用 `datasets/rhino_obb`：

- 图像由原始格式转换为 RHINO DOTA loader 要求的 PNG。
- YOLO OBB 归一化四点标注转换为 DOTA 四点文本标注。
- train/test 图像划分和类别定义与 `datasets/obb_thin_thick` 保持一致。

主要入口：

- `scripts/create_rhino_dataset.py`：创建 RHINO 数据格式。
- `scripts/train_rhino.py`：生成配置并启动 KLD 或 RIoU 训练。
- `scripts/predict_rhino.py`：使用 `.pth` 对指定 split 推理，产生 `.pkl`。
- `scripts/export_rhino_predictions.py`：把 `.pkl` 转换为统一的 YOLO OBB 预测文本。
- `scripts/evaluate_rhino.py`：串联导出与统一指标评估。

RHINO 的完整评估链路为：

```text
checkpoint.pth
  -> RHINO 推理 predictions.pkl
  -> 每张图一个 OBB labels/*.txt
  -> Ultralytics OBB mAP50/mAP80/mAP85/mAP90/mAP95
```

其中 `.pkl` 相当于 RHINO 原生的整批预测容器，转换后的 `labels/*.txt` 才与 YOLO 导出的逐图预测文件处于同一层级。

## 4. 输出目录约定

- YOLO 检测：`runs/obb/<run_name>/`
- RHINO 检测：`runs/rhino/<run_name>/`
- 分类：`runs/classification/<run_name>/`
- 完整 pipeline 评估：`runs/pipeline_eval/<run_name>/`

训练权重、预测结果和评估结果应放在同一 run 根目录下的子目录中，避免训练、验证、测试产物散落到不同顶层目录。
