# 端子智能检测系统部署指南

本 README 只说明 Docker 部署所需的前置准备、部署步骤和参数填写方法。部署脚本会自动区分首次部署与更新部署，并统一启动 PostgreSQL、API、推理 Worker 和网页前端。

## 1. 部署前准备

### 1.1 基础环境

部署主机需要安装：

- Git；
- Docker Engine 或 Docker Desktop；
- Docker Compose v2，能够执行 `docker compose version`；
- 至少 20 GB 可用磁盘空间，生产环境应按检测图片留存量额外预留空间。

CPU 部署不需要安装 CUDA。NVIDIA GPU 部署还需要：

- 可正常执行 `nvidia-smi`；
- 已安装 NVIDIA 驱动；
- 已安装 NVIDIA Container Toolkit；
- Docker 容器能够访问准备使用的 GPU。

macOS 的 Docker 容器不能使用 Apple MPS，本项目在 Mac Docker Desktop 中应选择 `cpu`。

### 1.2 模型权重

部署前必须准备以下三个文件，文件名和路径不能改变：

```text
weights/detector/yolo11l_obb_best.pt
weights/classifiers/label3/resnet18_best.pt
weights/classifiers/label5/resnet18_best.pt
```

检查权重：

```bash
ls -lh \
  weights/detector/yolo11l_obb_best.pt \
  weights/classifiers/label3/resnet18_best.pt \
  weights/classifiers/label5/resnet18_best.pt
```

部署脚本只检查权重是否存在且不为空，不会自动下载模型。

### 1.3 端口和目录

默认使用：

| 用途 | 默认值 | 说明 |
| --- | --- | --- |
| 网页端口 | `8080` | 浏览器访问端口 |
| PostgreSQL 端口 | `5432` | 仅绑定本机 `127.0.0.1` |
| 单任务图片上限 | `100` | 允许填写 1–100 |

确认端口没有被其他程序占用，并在项目根目录赋予部署脚本执行权限：

```bash
chmod +x deploy.sh
```

## 2. 首次部署

在项目根目录执行：

```bash
./deploy.sh
```

脚本会依次完成：

1. 检查 Docker、Docker Compose 和 Docker Engine；
2. 询问数据库、推理设备、端口和网页登录用户；
3. 检查三个模型权重；
4. 根据 CPU/GPU 配置选择 Worker 依赖和 Compose 配置；
5. 构建 API、Worker 和前端镜像；
6. 创建并启动 PostgreSQL；
7. 执行 Alembic 数据库迁移；
8. 创建首个网页登录用户；
9. 启动所有服务并等待健康检查通过。

如需先查看执行计划而不写入配置、不构建镜像：

```bash
./deploy.sh --dry-run
```

部署完成后访问：

```text
http://127.0.0.1:8080/tasks
```

局域网或服务器部署时，将 `127.0.0.1` 换成部署主机的实际 IP 地址。

## 3. 部署参数如何填写

### 3.1 数据库配置

| 提示 | 推荐填写 | 规则 |
| --- | --- | --- |
| 数据库名称 | 直接回车，使用 `terminal_inspection` | 只能包含字母、数字和下划线，不能以数字开头 |
| 数据库用户 | 直接回车，使用 `terminal` | 只能包含字母、数字和下划线，不能以数字开头 |
| 数据库密码 | 自行设置 | 至少 6 位，只能使用字母、数字及 `.`、`_`、`~`、`-` |
| 确认数据库密码 | 再输入一次 | 两次必须完全一致 |
| 数据库监听地址 | 直接回车，使用 `127.0.0.1` | 仅允许 `127.0.0.1` 或 `localhost` |
| 数据库端口 | 直接回车，使用 `5432` | 1–65535，且不能与现有服务冲突 |

密码输入不会回显。数据库凭据会写入权限为 `600` 的 `.env`，不要提交该文件，也不要删除更新部署仍在使用的 `.env`。

### 3.2 推理设备

部署脚本只接受 `cpu` 或非负 GPU 编号：

| 场景 | 检测模型设备 | 分类模型设备 |
| --- | --- | --- |
| 纯 CPU | `cpu` | `cpu` |
| 使用第一块 NVIDIA GPU | `0` | `0` |
| 使用第七块 NVIDIA GPU | `6` | `6` |
| 检测用 GPU、分类用 CPU | `0` | `cpu` |

首次部署时，如果 `nvidia-smi` 正常，默认设备为 `0`；否则默认是 `cpu`。设备编号必须与 `nvidia-smi --query-gpu=index --format=csv,noheader` 输出一致。

设备选择会同时决定 Worker 镜像依赖：

- 检测和分类都为 `cpu`：安装 CPU-only PyTorch，不包含 CUDA/NVIDIA 运行库；
- 任意一个设备为 GPU 编号：安装 CUDA PyTorch，并自动叠加 `compose.gpu.yaml`；
- API 镜像始终使用精简依赖，不安装 PyTorch、Ultralytics 或 CUDA 运行库。

部署脚本会在启动旧服务之前验证宿主机 GPU，并在 Worker 容器内再次验证 `torch.cuda.is_available()` 和设备编号。

### 3.3 网页和任务配置

| 提示 | 推荐填写 | 规则 |
| --- | --- | --- |
| 网页端口 | 直接回车，使用 `8080` | 1–65535 |
| 单任务最大图片数 | 直接回车，使用 `100` | 1–100 |

### 3.4 首个网页登录用户

首次部署必须创建一个网页登录用户：

- 用户名区分大小写；
- 用户名不能为空，不能包含空白或控制字符；
- 密码至少 6 位；
- 密码需要输入两次；
- 密码只通过标准输入传递，不写入 `.env`、部署日志或命令参数。

网页不提供自行注册功能。后续新增用户可在更新部署时按提示创建，也可以执行：

```bash
docker compose exec api python scripts/manage_users.py add USERNAME
```

重置密码：

```bash
docker compose exec api python scripts/manage_users.py reset-password USERNAME
```

## 4. 更新部署

已有 `.env` 和 PostgreSQL 数据卷时，再次执行同一个命令：

```bash
./deploy.sh
```

脚本会自动进入更新模式，并询问：

1. 是否从当前 Git 上游执行 `git pull --ff-only`；
2. 是否修改设备、端口或任务上限；
3. 是否新增网页登录用户；
4. 是否确认开始更新。

更新过程中会先备份 PostgreSQL，再构建新镜像、执行迁移和健康检查。数据库备份保存在：

```text
backups/
```

如果工作区存在未提交的已跟踪文件，脚本会拒绝自动拉取，避免覆盖本地修改。

## 5. 常用运维命令

查看服务状态：

```bash
docker compose ps
```

查看实时日志：

```bash
docker compose logs -f api worker frontend
```

停止服务但保留数据库和检测历史：

```bash
docker compose down
```

如果容器只是被暂停或停止，可按原部署配置重新启动：

```bash
docker compose start
```

如果已经执行过 `docker compose down`，请重新运行 `./deploy.sh`。部署脚本会恢复正确的 CPU/GPU Compose 组合；不要直接执行不带 GPU 覆盖文件的 `docker compose up -d`。

查看镜像、容器、数据卷和构建缓存占用：

```bash
docker system df -v
```

不要执行下面的命令：

```bash
docker compose down -v
```

`-v` 会删除 PostgreSQL 和检测结果使用的持久卷，可能造成不可恢复的数据丢失。

## 6. 镜像下载与网络问题

部署脚本获取基础镜像时按以下顺序尝试：

1. Docker Hub；
2. DaoCloud 国内镜像；
3. 本地已有基础镜像缓存。

只有镜像仓库连接超时、鉴权超时或连接被重置时才会自动切换；Dockerfile、依赖版本或程序错误会直接停止并打印日志路径。

自定义国内 Docker 镜像前缀：

```bash
DOCKER_MIRROR_PREFIX=mirror.example.com/docker.io ./deploy.sh
```

CPU-only PyTorch 默认从官方 CPU wheel 仓库安装。如部署网络需要使用兼容镜像，可临时覆盖：

```bash
PYTORCH_CPU_INDEX_URL=https://mirror.example.com/pytorch/cpu ./deploy.sh
```

该地址必须提供与 `requirements.txt` 中固定版本匹配的 `torch` 和 `torchvision` wheel。

## 7. 配置和数据保留

- `.env`：Docker 部署配置及数据库凭据，权限为 `600`；
- `backups/env/`：更新配置前自动生成的 `.env` 回滚副本；
- `backups/`：更新部署前生成的 PostgreSQL 备份；
- `terminal-inspection_postgres-data`：PostgreSQL 数据卷；
- `terminal-inspection_inspection-data`：上传原图、检测结果图和 Worker 就绪状态。

服务器迁移或备份时，必须同时保留 PostgreSQL 数据和检测文件卷。只备份其中一项无法完整恢复检测历史。
