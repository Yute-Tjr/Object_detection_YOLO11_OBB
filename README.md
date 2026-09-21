# 端子智能检测系统

本项目提供一套面向端子生产质检的端到端检测网页：上传 1–100 张端子图片后，系统先用旋转目标检测模型定位各功能区域，再对 `label3` 和 `label5` 执行 OK/NG 异常分类，最终保存原图、标注结果、任务状态和检测历史。

当前工程重点是稳定完成生产检测闭环。模型选型、batch 消融和完整分类别评测结果已经归档，不再放在项目首页。

## 1. 当前检测流程

```text
浏览器上传图片
  -> FastAPI 创建检测任务
  -> PostgreSQL 保存任务和队列状态
  -> 单个 Worker 领取任务
  -> YOLO11l-OBB 检测 7 类端子区域
  -> 裁剪 label3 / label5 区域
  -> ResNet18 分别判断 OK / NG
  -> 生成标注结果图并持久化
  -> 网页展示进度、结果对比和历史记录
```

检测类别为：

```text
label1_thin
label1_thick
label2
label3
label4
label5
label6
```

- `label3`、`label5`：当前支持 ResNet18 OK/NG 分类，OK 使用绿色框，NG 使用红色框。
- 其他区域：完成分区域检测，但暂不进行异常分类，使用灰色框。
- 颜色分类：后端和网页已经预留接口，当前没有接入颜色模型，不会生成或猜测颜色结果。

## 2. 当前使用模型

| 用途 | 模型 | 权重路径 | 当前结果或说明 |
| --- | --- | --- | --- |
| 分区域检测 | YOLO11l-OBB | `weights/detector/yolo11l_obb_best.pt` | 当前最强基线；mAP85=`0.894190`，mAP90=`0.711124` |
| label3 异常分类 | ResNet18 | `weights/classifiers/label3/resnet18_best.pt` | 人工框裁剪测试：accuracy=`0.981132`，macro F1=`0.977999` |
| label5 异常分类 | ResNet18 | `weights/classifiers/label5/resnet18_best.pt` | 人工框裁剪测试：accuracy=`1.000000`，macro F1=`1.000000` |

检测模型在当前统一评估中同时取得最高的 mAP85 和 mAP90，因此网页部署固定使用 YOLO11l-OBB。分类指标基于人工 OBB 框裁剪，用于说明分类器本身的能力；实际端到端表现还会受到检测框定位误差影响。

完整模型对比、各类别指标和 checkpoint 结果见 [模型选型与网页状态归档](docs/archive/06_model_selection_and_web_snapshot_20260921.md)。

## 3. 快速启动

### 3.1 环境要求

- Python 3.11
- Node.js 18 或更高版本
- PostgreSQL
- CPU、CUDA GPU 或支持 PyTorch MPS 的 Apple 芯片

推荐使用 uv 创建并同步项目环境：

```bash
uv sync
cd web_frontend && npm ci && cd ..
cp .env.example .env
```

如果需要使用 Conda，也可以根据同一份 `requirements.txt` 建立环境：

```bash
conda create -n terminal-inspection python=3.11 -y
conda activate terminal-inspection
python -m pip install -r requirements.txt
cd web_frontend && npm ci && cd ..
cp .env.example .env
```

`requirements.txt` 与 `pyproject.toml` 保存相同的项目直接依赖；`uv.lock` 进一步锁定 uv 路线的全部传递依赖。不要再单独维护网页依赖文件。

确认三个模型权重已放到第 2 节列出的固定路径。启动器只检查依赖和权重，不会自动下载模型或修改 Python/Node 环境。

### 3.2 配置 `.env`

至少配置数据库、持久化目录、模型权重和运行设备：

```dotenv
DATABASE_URL=postgresql+psycopg://terminal:请替换密码@127.0.0.1:5432/terminal_inspection
INSPECTION_STORAGE_ROOT=./var/terminal-inspection

DETECTOR_WEIGHTS=./weights/detector/yolo11l_obb_best.pt
LABEL3_CLASSIFIER_WEIGHTS=./weights/classifiers/label3/resnet18_best.pt
LABEL5_CLASSIFIER_WEIGHTS=./weights/classifiers/label5/resnet18_best.pt

DETECTION_DEVICE=cpu
CLASSIFICATION_DEVICE=cpu
```

设备配置示例：

| 运行设备 | `DETECTION_DEVICE` | `CLASSIFICATION_DEVICE` |
| --- | --- | --- |
| CPU | `cpu` | `cpu` |
| 第一块 CUDA GPU | `0` | `0` |
| Apple MPS | `mps` | `mps` |

不要把包含真实数据库密码的 `.env` 提交到 Git。

### 3.3 一键启动

在项目根目录执行：

```bash
uv run python scripts/start_terminal_web.py
```

启动器会依次用中文输出以下检查结果：

1. `.env` 配置；
2. 三个模型权重；
3. Python 环境和核心依赖；
4. Node.js、npm 和 Vite 前端依赖；
5. 可用 GPU 及检测/分类模型实际配置；
6. PostgreSQL 连接；
7. `alembic upgrade head` 数据库迁移。

全部通过后，启动器会同时运行 API、Worker 和 Vite，并使用 `[API]`、`[WORKER]`、`[WEB]` 区分日志。任一服务异常退出时，其余服务会一并关闭；按一次 `Ctrl+C` 可以完整停止三个进程。

默认访问地址：

- 网页：`http://127.0.0.1:5173/tasks`
- 健康接口：`http://127.0.0.1:8000/api/v1/health`

自定义监听地址和端口：

```bash
uv run python scripts/start_terminal_web.py \
  --host 127.0.0.1 \
  --api-port 8000 \
  --web-port 5173
```

Conda 路线激活环境后，将上述命令中的 `uv run python` 替换为 `python`。

### 3.4 Docker 一键部署与更新

Docker 部署不需要手动编辑完整 `.env`。在项目根目录执行：

```bash
./deploy.sh
```

首次运行时，向导会询问数据库名称、用户名和两次隐藏密码，并为设备、端口、输入尺寸和单任务图片数提供默认值。配置完成后，它会检查 Docker、三个权重和 GPU，生成权限为 `600` 的 `.env`，再构建镜像、初始化 PostgreSQL、执行 Alembic 迁移并启动 API、Worker 和前端。

当 `.env` 和 PostgreSQL 数据卷已经存在时，同一命令会自动进入更新模式：复用数据库凭据，可选择拉取当前 Git 上游，先备份数据库，再构建、迁移和健康检查。查看计划但不产生写入：

```bash
./deploy.sh --dry-run
```

CPU 部署默认使用 `compose.yaml`；只要检测或分类设备设置为 GPU 编号，向导就会自动叠加 `compose.gpu.yaml`。Docker 部署细节见 [部署文档](docs/web-deployment.md)。

## 4. 网页功能

- 单次上传 1–100 张 JPG、PNG 或 BMP 图片；
- 填写必填操作员，以及可选任务名称和备注；
- 查看目标检测、异常分类、结果生成等实时阶段；
- 对比检测前原图和检测后完整结果图；
- 按进行中、全部、成功、失败筛选任务；
- 查看历史任务、逐图预览和失败图片重试；
- PostgreSQL 保存任务元数据，文件系统保存原图和结果图，正常重启不会丢失历史记录。

当前版本使用单个推理 Worker，不支持多个 Worker 共享同一块 GPU。

## 5. 项目结构

```text
terminal_web/             FastAPI、数据库、任务队列、Worker 和推理编排
web_frontend/             React + Vite 网页前端
scripts/                  一键启动、训练、预测、评测和数据处理入口
weights/                  当前网页使用的检测与分类权重
migrations/               Alembic PostgreSQL 数据库迁移
deploy/                   API/前端容器、Nginx 和 systemd 配置
datasets/                 OBB 检测及 label3/label5 分类数据集
yolo11_obb/               YOLO OBB 数据、评测与分类兼容实现
obb_detection/            RHINO、Oriented R-CNN 等模型集成
tests/                    Python 单元测试与端到端测试
docs/                     架构、部署、训练日志和历史实验归档
outputs/                  数据分析和标注汇总结果
runs/                     训练、评测和推理产物
var/                      本地网页检测原图、结果图和就绪状态
```

核心运行关系：

```text
React/Vite -> FastAPI -> PostgreSQL
                         ^       |
                         |       v
                    单 Worker -> YOLO11l + ResNet18
                         |
                         v
                  INSPECTION_STORAGE_ROOT
```

## 6. 数据与持久化

- 网页检测模型使用 `datasets/obb_thin_thick` 对应的 7 类定义。
- YOLO 主数据集包含 198 张训练图片和 53 张测试图片。
- PostgreSQL 保存任务、图片状态和结构化检测结果。
- 数据库保存带时区的绝对时间，网页和 Docker PostgreSQL 统一按北京时间显示。
- `INSPECTION_STORAGE_ROOT` 保存上传原图、裁剪图、标注结果图和 Worker 就绪信息。
- 服务器备份必须同时覆盖 PostgreSQL 和整个 artifact 目录。

## 7. 测试与构建

运行 Python 测试：

```bash
uv run python -m unittest discover -s tests -v
```

运行前端测试和生产构建：

```bash
cd web_frontend
npm test -- --run
npm run build
```

## 8. 部署与详细文档

- [项目文档索引](docs/README.md)
- [代码架构和模型脚本组织](docs/architecture.md)
- [Docker Compose、systemd、Nginx、备份和回滚](docs/web-deployment.md)
- [完整模型选型、实验结果和旧版网页说明](docs/archive/06_model_selection_and_web_snapshot_20260921.md)

生产环境推荐使用交互式部署向导：

```bash
./deploy.sh
curl -fsS http://127.0.0.1:8080/api/v1/health
```

向导会自动区分首次部署和更新，更新前备份 PostgreSQL，并等待 API、Worker 模型和前端全部就绪。不要执行 `docker compose down -v`，该命令会删除 PostgreSQL 和检测文件的持久卷。
