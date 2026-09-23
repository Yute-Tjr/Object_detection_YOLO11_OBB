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

系统使用服务端会话 Cookie，默认有效期 12 小时。局域网 HTTP 部署使用 `SESSION_COOKIE_SECURE=false`；生产 HTTPS 部署必须设置 `SESSION_COOKIE_SECURE=true`。该设置只控制 Cookie 是否要求安全连接，不替代 TLS、访问控制和数据库备份。

## 2. Docker Compose 一键部署与更新

推荐从仓库根目录运行交互式向导：

```bash
./deploy.sh
```

首次部署时，向导会：

1. 检查 Docker、Docker Compose、权重和 GPU；
2. 询问数据库名称、用户名，并隐藏输入和二次确认密码；
3. 询问首个网页登录用户名，并隐藏输入和二次确认密码；
4. 为设备、端口和单任务图片数提供默认值；
5. 生成权限为 `600` 的 `.env`；
6. 构建镜像、初始化 PostgreSQL、执行迁移并启动服务；
7. 自动创建首个登录用户，等待 API、Worker 模型和前端健康检查通过。

向导不会生成弱默认账户，而是使用本次输入的凭据创建首个用户。密码在终端中隐藏输入并要求二次确认，不会写入 `.env`、部署日志或 Docker 命令参数。用户名区分大小写，`Admin` 与 `admin` 是两个不同账户。更新部署默认不新增账户，交互模式可选择新增；已有用户改密仍使用管理命令：

```bash
docker compose exec api python scripts/manage_users.py list
docker compose exec api python scripts/manage_users.py reset-password USERNAME
docker compose exec api python scripts/manage_users.py disable USERNAME
docker compose exec api python scripts/manage_users.py enable USERNAME
```

重置密码或禁用用户会立即删除其现有会话。系统不提供网页注册；新增、启用、禁用和改密均由管理员执行。

数据库密码和网页登录密码最低均为 6 位，交互模式必须输入两次。数据库密码另外限定为 URL 安全的字母、数字、点、下划线、波浪线和连字符。脚本拒绝把 `.env.example` 的占位密码用于首次部署，也不会在摘要或日志中显示密码。查看执行计划但不写配置、不启动容器：

```bash
./deploy.sh --dry-run
```

当 `.env` 和 `terminal-inspection_postgres-data` 数据卷同时存在时，脚本自动进入更新模式。更新默认复用数据库凭据，可选择新增登录用户并使用 `git pull --ff-only` 拉取当前上游；拉取成功后会重新载入新版部署脚本。随后依次启动数据库、备份 PostgreSQL、构建镜像、停止旧 API/Worker、迁移、清除旧 Worker 就绪状态并重启服务。若存在未提交的已跟踪文件，脚本拒绝自动拉取。

非交互首次部署必须在进程环境中同时提供 `POSTGRES_PASSWORD`、`INITIAL_APP_USERNAME` 和 `INITIAL_APP_PASSWORD`；更新部署只有同时提供后两个变量时才新增用户。`INITIAL_APP_PASSWORD` 会在 Docker 命令执行前从进程环境移除，也不会写入 `.env`：

```bash
POSTGRES_PASSWORD=DbPass6 \
INITIAL_APP_USERNAME=Admin \
INITIAL_APP_PASSWORD=WebPass6 \
./deploy.sh --yes --no-pull
```

脚本只更新自己管理的 Docker 字段；若 `.env` 原本含有供 uv/Conda 原生启动使用的 `DATABASE_URL`，会将其同步到本次 PostgreSQL 用户、密码和端口，同时保留权重路径、存储路径及其他自定义配置。真实部署会拒绝符号链接或非当前用户所有的 `.env`，并将权限收紧为 `600`；更新前还会在 `backups/env/` 中保留一份权限为 `600` 的配置备份。

### 2.1 Docker Hub 网络故障兜底

基础镜像默认从 Docker Hub 获取。如果日志明确出现镜像仓库超时或连接重置，部署脚本按 Docker Hub → `m.daocloud.io/docker.io` → 本地默认镜像缓存的顺序自动回退。本地阶段恢复 `python:3.11-slim`、`node:22-alpine`、`nginx:1.27-alpine` 和 `postgres:16-alpine` 等默认镜像名，并禁止 PostgreSQL 再次拉取。镜像源切换只对当前部署进程生效，不会修改 Docker Desktop 全局配置。

自定义兼容镜像前缀时，使用 registry/路径形式，不要包含协议或末尾斜杠：

```bash
DOCKER_MIRROR_PREFIX=mirror.example.com/docker.io ./deploy.sh
```

国内公共镜像和本地缓存仍可能不可用。如果三级尝试全部失败，脚本会保留失败状态并输出详细日志；普通 Dockerfile、依赖安装或容器启动错误不会触发镜像源回退。

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
SESSION_TTL_HOURS=12
SESSION_COOKIE_SECURE=true
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

迁移并创建首个原生部署用户：

```bash
cd /opt/terminal-inspection/current
/home/tjr/miniconda3/bin/python -m alembic upgrade head
/home/tjr/miniconda3/bin/python scripts/manage_users.py add USERNAME
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

每日备份至少包含 PostgreSQL 和整个 artifact 目录。PostgreSQL 现在还保存用户、服务端会话和逐图片人工反馈：

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

认证与反馈迁移的降级会删除用户、会话和反馈表，因此不能把 `alembic downgrade` 当作普通代码回滚。回滚前必须同时保留数据库 dump 与 artifact 目录，并在独立数据库中验证恢复。含人工反馈的检测任务会被 API 拒绝删除，这是为了保留后续模型优化所需的图片和纠正记录；如确需清理，应先导出反馈数据并制定显式的数据迁移流程，而不是直接绕过外键或删除持久卷。

## 5. 故障定位

- `/api/v1/health` 中 `modelsReady=false`：检查三个权重路径、权限和文件是否非空。
- `workerReady=false`：检查 Worker 日志、GPU 设备号和 `INSPECTION_STORAGE_ROOT/worker-readiness.json` 的更新时间。
- 页面任务一直等待：确认只有一个 Worker 正在运行，并检查 PostgreSQL 连接。
- 原图可见但结果图缺失：查看对应图片的脱敏错误信息和 Worker 日志；不要直接暴露服务器绝对路径。
- 登录后立即返回登录页：检查账户是否被禁用、会话是否过期，以及 HTTPS 部署是否已设置 `SESSION_COOKIE_SECURE=true`。
- 任务删除按钮不可用并提示“包含人工反馈”：这是数据保留策略，不是页面故障。
