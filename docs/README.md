# 项目文档索引

当前可复现的目标检测流程、模型结论和统一指标见项目根目录的 [README](../README.md)。本目录按项目阶段保存历史背景，避免早期安装记录、数据实验和当前结论混在同一份文档中。

## 当前代码结构

- [代码架构、兼容策略与 Oriented R-CNN 入口](architecture.md)

说明新的 `obb_detection` 包、按模型分类的命令目录，以及为什么暂时保留 `yolo11_obb` 兼容层。

## 阶段一：环境与运行方式

- [环境、脚本入口与产物约定](archive/01_environment_and_workflows.md)

记录 YOLO 与 RHINO 的环境隔离、训练/预测/评估入口，以及各阶段输出文件的含义。

## 阶段二：数据集演进

- [数据集清理与版本演进](archive/02_dataset_evolution.md)

记录从原始 AnyLabeling 数据到当前 `obb_thin_thick` 数据集的筛选、类别拆分、样本数量和 RHINO 格式转换。

## 阶段三：YOLO 定位优化

- [YOLO11 OBB 高 IoU 优化实验](archive/03_yolo_optimization_experiments.md)

记录分辨率、batch、旋转增强、时间筛选、`index=-1` 清理和 label1 thin/thick 拆分等历史实验。

## 阶段四：模型路线选择

- [deskew、fusion 与 RHINO 路线记录](archive/04_model_route_history.md)

记录已停止的 deskew/fusion 路线，以及 RHINO KLD/RIoU 单模型对照路线的由来和评估链路。

## 阶段五：下游分类

- [ResNet18 分类实验记录](archive/05_resnet_classification_experiments.md)

记录 label3、label5 裁剪分类的数据来源、训练配置、结果和局限。

## 原始训练日志

- [早期 LabelMe 训练日志](training_logs/legacy_labelme_training_log.md)
- [AnyLabeling label1-label6 训练日志](training_logs/anylabeling_label1_6_training_log.md)

## 只读快照

- [2026-07-17 前的原始 README](archive/raw_readme_snapshot_20260717.md)

该快照仅用于追溯，不再更新，也不作为当前训练和模型选择依据。
