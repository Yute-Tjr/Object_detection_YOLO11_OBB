# 端子检测网页部署

本系统由 PostgreSQL、FastAPI、一个推理 Worker 和 Nginx/React 前端组成。Worker 数量固定为 1；当前版本不支持多个 Worker 共享同一块 GPU。原图、结果图和 Worker 就绪文件位于 `INSPECTION_STORAGE_ROOT`，任务元数据位于 PostgreSQL，因此正常重启不会丢失检测历史。

## 1. 部署前检查

仓库根目录必须存在以下权重，容器和进程只读使用它们：

```text
weights/detector/yolo11l_obb_best.pt
weights/classifiers/label3/resnet18_best.pt
weights/classifiers/label5/resnet18_best.pt
```

检查文件而不读取模型内容：

```bash
test -s weights/detector/yolo11l_obb_best.pt
test -s weights/classifiers/label3/resnet18_best.pt
test -s weights/classifiers/label5/resnet18_best.pt
```

网页单任务限制为 100 张图片（`MAX_IMAGES_PER_TASK=100`）。Nginx 请求体上限为 2 GiB，用于覆盖 100 张生产图；若现场单批文件可能超过 2 GiB，需要同时调整 `deploy/nginx.conf`，而不是只改前端提示。

## 2. Docker Compose 一键部署与更新

推荐从仓库根目录运行交互式向导：

```bash
./deploy.sh
```

首次部署时，向导会：

1. 检查 Docker、Docker Compose、权重和 GPU；
2. 询问数据库名称、用户名，并隐藏输入和二次确认密码；
3. 为设备、端口、输入尺寸和单任务图片数提供默认值；
4. 生成权限为 `600` 的 `.env`；
5. 构建镜像、初始化 PostgreSQL、执行迁移并启动服务；
6. 等待 API、Worker 模型和前端健康检查通过。

新建数据库密码至少 12 位，并限定为 URL 安全的字母、数字、点、下划线、波浪线和连字符；交互模式必须输入两次。脚本拒绝把 `.env.example` 的占位密码用于首次部署，也不会在摘要或日志中显示密码。若已有数据库仍使用旧的短密码，更新模式会保留原凭据并给出轮换提醒，因为只修改 `.env` 并不能修改数据库内部密码。查看执行计划但不写配置、不启动容器：

```bash
./deploy.sh --dry-run
```

当 `.env` 和 `terminal-inspection_postgres-data` 数据卷同时存在时，脚本自动进入更新模式。更新默认复用数据库凭据，可选择使用 `git pull --ff-only` 拉取当前上游；拉取成功后会重新载入新版部署脚本。随后依次启动数据库、备份 PostgreSQL、构建镜像、停止旧 API/Worker、迁移、清除旧 Worker 就绪状态并重启服务。若存在未提交的已跟踪文件，脚本拒绝自动拉取。

脚本只更新自己管理的 Docker 字段；若 `.env` 原本含有供 uv/Conda 原生启动使用的 `DATABASE_URL`，会将其同步到本次 PostgreSQL 用户、密码和端口，同时保留权重路径、存储路径及其他自定义配置。真实部署会拒绝符号链接或非当前用户所有的 `.env`，并将权限收紧为 `600`；更新前还会在 `backups/env/` 中保留一份权限为 `600` 的配置备份。

### 2.1 Docker Hub 网络故障兜底

基础镜像默认仍从 Docker Hub 获取。如果日志明确出现 Docker Hub 超时或连接重置，部署脚本先不拉取新版本，改用本地基础镜像缓存构建。本地缓存仍不可用时，交互模式会询问是否切换到 `m.daocloud.io/docker.io`，`--yes` 非交互模式则自动接受。该切换同时覆盖 Python、Node、Nginx 和 PostgreSQL 基础镜像，仅对当前部署进程生效，不会修改 Docker Desktop 全局配置。

自定义兼容镜像前缀时，使用 registry/路径形式，不要包含协议或末尾斜杠：

```bash
DOCKER_MIRROR_PREFIX=mirror.example.com/docker.io ./deploy.sh
```

国内公共镜像仍可能限流或临时不可用。如果兜底构建也失败，脚本会保留失败状态并输出详细日志，不会将普通 Dockerfile 或依赖安装错误误判为成功。

CPU 是基础 Compose 配置；当 `DETECTION_DEVICE` 或 `CLASSIFICATION_DEVICE` 为 GPU 编号时，脚本自动叠加 `compose.gpu.yaml`。GPU 模式会先检查宿主机 GPU 编号，再在构建完成后、停止旧服务前验证 Worker 容器中的 CUDA，因此服务器必须安装 NVIDIA 驱动和 NVIDIA Container Toolkit。

高级用途仍可手动执行 Compose。CPU 模式：

```bash
docker compose --env-file .env -f compose.yaml up -d --build
```

GPU 模式：

```bash
docker compose --env-file .env -f compose.yaml -f compose.gpu.yaml up -d --build
```

本机打开 `http://127.0.0.1:8080/tasks`。如果部署在远程服务器，将 `127.0.0.1` 替换为服务器 IP。健康检查：

```bash
curl -fsS http://127.0.0.1:8080/api/v1/health
docker compose ps
docker compose logs --tail=200 api worker frontend
```

Compose 使用 `postgres-data` 保存数据库，使用 `inspection-data` 保存原图和结果图。`./weights` 以只读方式挂载。不要执行 `docker compose down -v`，该命令会删除两个持久卷。

## 3. 学校服务器原生部署（systemd）

以下示例将代码放在 `/opt/terminal-inspection/current`，数据放在 `/srv/terminal-inspection`。可以把 `TERMINAL_PYTHON` 指向已验证可同时导入 Ultralytics、Torchvision、FastAPI 和 psycopg 的 base/RHINO Conda Python；不要未经验证直接假定任一现有环境完整。

```bash
sudo useradd --system --home /srv/terminal-inspection --shell /usr/sbin/nologin terminal-inspection
sudo install -d -o terminal-inspection -g terminal-inspection /srv/terminal-inspection
sudo install -d -m 0750 /etc/terminal-inspection
```

创建 `/etc/terminal-inspection/terminal-web.env`，权限设为 `0640`。这里的密码仅写在服务器，不提交 Git：

```dotenv
TERMINAL_PYTHON=/home/tjr/miniconda3/bin/python
DATABASE_URL=postgresql+psycopg://terminal:替换密码@127.0.0.1:5432/terminal_inspection
INSPECTION_STORAGE_ROOT=/srv/terminal-inspection/artifacts
DETECTOR_WEIGHTS=/opt/terminal-inspection/current/weights/detector/yolo11l_obb_best.pt
LABEL3_CLASSIFIER_WEIGHTS=/opt/terminal-inspection/current/weights/classifiers/label3/resnet18_best.pt
LABEL5_CLASSIFIER_WEIGHTS=/opt/terminal-inspection/current/weights/classifiers/label5/resnet18_best.pt
DETECTION_DEVICE=0
CLASSIFICATION_DEVICE=0
MAX_IMAGES_PER_TASK=100
```

验证 Python 环境和模型加载依赖：

```bash
/home/tjr/miniconda3/bin/python -c 'import fastapi, psycopg, sqlalchemy, torch, torchvision, ultralytics; print("runtime ok")'
/home/tjr/miniconda3/bin/python -m pip install -r requirements.txt
sudo systemctl start terminal-api.service
```

`terminal-api.service` 的 `ExecStartPre` 会读取同一个环境文件并执行 `alembic upgrade head`，无需把数据库密码展开到命令行参数或 shell 历史中。

PostgreSQL 数据库首次创建：

```bash
sudo -u postgres createuser --pwprompt terminal
sudo -u postgres createdb --owner terminal terminal_inspection
```

构建前端并把静态文件交给主机 Nginx：

```bash
cd web_frontend
npm ci
npm run build
sudo rsync -a --delete dist/client/ /var/www/terminal-inspection/
```

主机 Nginx 可复用 `deploy/nginx.conf` 的 `server` 内容，但将 `upstream terminal_api` 指向 `127.0.0.1:8000`。安装服务：

```bash
sudo cp deploy/terminal-api.service deploy/terminal-worker.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now terminal-api.service terminal-worker.service
sudo nginx -t
sudo systemctl reload nginx
```

检查服务与日志：

```bash
systemctl status terminal-api terminal-worker --no-pager
journalctl -u terminal-api -u terminal-worker -n 200 --no-pager
curl -fsS http://127.0.0.1:8000/api/v1/health
nvidia-smi
```

## 4. 备份、升级与回滚

每日备份至少包含 PostgreSQL 和整个 artifact 目录：

```bash
pg_dump -Fc terminal_inspection > /srv/terminal-inspection/backups/terminal_inspection_$(date +%F).dump
rsync -a /srv/terminal-inspection/artifacts/ /srv/terminal-inspection/backups/artifacts/
```

升级前创建 Git 标签或记录提交 SHA，然后执行迁移、构建前端并依次重启 API、Worker。模型权重的 SHA256 应另行记录：

```bash
sha256sum weights/detector/yolo11l_obb_best.pt weights/classifiers/label3/resnet18_best.pt weights/classifiers/label5/resnet18_best.pt
sudo systemctl restart terminal-api
sudo systemctl restart terminal-worker
```

代码回滚使用上一已验证提交重新部署前端与两个服务。数据库迁移默认只向前；若新版本包含不可逆迁移，先从 `pg_dump` 恢复到独立数据库验证，再切换 `DATABASE_URL`。artifact 目录不能随代码版本删除。

## 5. 故障定位

- `/api/v1/health` 中 `modelsReady=false`：检查三个权重路径、权限和文件是否非空。
- `workerReady=false`：检查 Worker 日志、GPU 设备号和 `INSPECTION_STORAGE_ROOT/worker-readiness.json` 的更新时间。
- 页面任务一直等待：确认只有一个 Worker 正在运行，并检查 PostgreSQL 连接。
- 原图可见但结果图缺失：查看对应图片的脱敏错误信息和 Worker 日志；不要直接暴露服务器绝对路径。
