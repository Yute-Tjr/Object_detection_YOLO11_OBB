# 端子分区域检测与异常分类 Web 系统设计

## 1. 目标

构建一个部署在工厂推理服务器上的桌面端 Web 系统，把现有模型能力串成可操作、可追溯的完整闭环：

```text
上传 1-100 张端子图像
  -> YOLO11l-OBB 分区域目标检测
  -> 裁剪 label3 / label5 区域
  -> ResNet18 OK/NG 异常分类
  -> 预留颜色分类接口
  -> 生成可视化结果图
  -> 持久化任务、结果和原始文件
  -> 网页查看当前进度与历史结果
```

系统面向生产线操作人员。第一版优先保证流程清晰、运行稳定、结果可追溯，不追求复杂动画、实时协作或大规模多用户并发。

## 2. 已确认约束

| 项目 | 第一版决定 |
| --- | --- |
| 单次上传 | 1-100 张图片 |
| 用户规模 | 暂不考虑多用户并发 |
| 检测模型 | 当前最强基线 YOLO11l-OBB `best.pt` |
| 异常分类 | label3、label5 各使用一个 ResNet18 OK/NG 权重 |
| 颜色分类 | 当前无模型，只预留后端字段、接口和前端显示位置 |
| 数据库 | PostgreSQL |
| 文件保存 | 服务器持久化磁盘，PostgreSQL 记录相对路径与元数据 |
| 历史记录 | 服务器重启后仍可查询，并可重新打开原图和结果图 |
| 推理执行 | 独立 GPU Worker，不在 Web 请求进程中直接运行 |
| 前端 | React + TypeScript + Vite |
| 后端 | FastAPI |
| 进度更新 | 第一版使用短轮询，不引入 WebSocket |
| 任务队列 | PostgreSQL 持久化队列，不额外引入 Redis/Celery |

## 3. 当前能力与未知项

### 3.1 已验证的仓库能力

仓库已有 `yolo11_obb.pipeline_predict`，能够：

1. 加载 YOLO OBB 权重并生成旋转框。
2. 选取 label3、label5 检测框。
3. 对旋转框做透视矫正裁剪。
4. 加载两个 ResNet18 checkpoint 并输出 `OK/NG` 与置信度。
5. 生成检测明细、汇总 CSV 和结果可视化图片。

Web 系统应复用并拆分这些能力，不再复制一套推理实现。

### 3.2 当前未知但不阻塞前端开发的内容

- 三个最终权重尚未下载到本地，准确路径和文件校验值未知。
- 颜色分类模型尚不存在，适用区域、类别集合和权重格式未知。
- 生产服务器的 GPU 型号、显存、域名和反向代理配置尚未确认。

以上内容必须通过环境变量或模型注册配置注入，不能硬编码到页面或业务代码。

## 4. 系统架构

```text
浏览器
  |
  | HTTP/JSON + 图片资源
  v
Nginx（生产部署）
  |------------------------------|
  v                              v
React 静态文件                FastAPI API
                                 |
               |-----------------|------------------|
               v                                    v
          PostgreSQL                         持久化文件目录
      任务、图片、框、分类结果          原图、裁剪图、结果图
               |
               v
       独立 inference-worker
       启动时加载三个权重
               |
       YOLO -> ResNet18 -> 渲染
```

部署时 API 与 Worker 使用同一 Python 包，但作为两个独立进程启动。这样 Web 请求不会被 GPU 推理阻塞，Worker 也可以一次性加载模型并复用显存。

第一版只运行一个 Worker。它串行领取任务，同一批次内按图片逐张处理并更新进度，避免多个推理任务同时争抢 GPU 显存。

## 5. 前端信息架构

### 5.1 全局导航

左侧仅保留两个入口：

- `检测任务`：上传、当前推理、结果预览及任务列表。
- `检测历史`：筛选、查询和重新打开已完成任务。

不在第一版加入仪表盘、统计图表、用户中心或模型管理页面。

### 5.2 检测任务页

页面采用已经确认的融合视觉稿，自上而下分成四层。

#### A. 上传区

- 支持拖拽或文件选择。
- 接受 JPG、JPEG、PNG、BMP。
- 显示已选数量与上限 100。
- 主按钮为 `开始检测`。
- 开始前允许移除单张图片或清空当前选择。
- 上传完成后生成一个批次任务，浏览器刷新不影响后台处理。

#### B. 当前推理状态

显示：

- 当前完成张数，例如 `正在检测 7 / 12`。
- 当前检测模型 `YOLO11l-OBB`。
- 总体百分比进度。
- 当前处理文件名。
- 两阶段状态：`分区域目标检测`、`异常分类`。
- 后端返回的模型就绪状态。

阶段状态不是装饰。Worker 在每张图进入对应阶段时更新数据库，网页轮询后展示真实状态。

#### C. 检测前后对比

左侧显示原始图像，右侧显示带旋转框的结果图。批量任务通过上一张、下一张或缩略列表切换当前图片。

框颜色语义固定：

| 情况 | 框颜色 | 标签示例 |
| --- | --- | --- |
| label3/label5 分类为 OK | 绿色 | `label3 · OK` |
| label3/label5 分类为 NG | 红色 | `label5 · NG` |
| 已检测但不支持异常分类 | 中性灰 | `label2 · 暂不支持分类` |
| label3/label5 分类失败 | 灰色虚线或错误标识 | `label3 · 分类失败` |

如果未来颜色模型返回结果，标签扩展为：

```text
label3 · OK · 蓝色
```

颜色接口未接入时不显示虚构颜色，只在详情区显示 `颜色：接口预留` 或不展示颜色行。

#### D. 任务管理表

状态筛选为：

- `进行中`
- `全部`
- `成功`
- `失败`

表格字段：任务 ID、图像数量、完成进度、模型、当前阶段、状态、创建时间。任务行可以展开，显示任务名称、文件存储标识、创建人占位和备注。

`失败`筛选同时包含完全失败和部分失败的任务。部分失败任务必须明确显示成功/失败图片数量，不能被标成完全成功。

### 5.3 检测历史页

历史页复用任务表，但默认只显示终态任务，并增加：

- 按任务 ID 或原始文件名搜索。
- 按状态筛选。
- 按创建时间倒序排列。
- 打开任务详情。
- 打开原始图和结果图。
- 对单张失败图片执行重新检测。

第一版不提供批量删除和自动清理，避免误删生产记录。后续若需要保留策略，应单独设计归档和审计规则。

## 6. 任务与图片状态机

### 6.1 任务状态

```text
queued
  -> running
      -> succeeded
      -> partial_failed
      -> failed
```

- `queued`：文件和数据库记录已创建，等待 Worker。
- `running`：Worker 已领取，至少一张图片开始处理。
- `succeeded`：所有图片成功生成结果。
- `partial_failed`：至少一张成功且至少一张失败。
- `failed`：任务无法启动，或所有图片均失败。

API/Worker 重启时，长时间停留在 `running` 且 Worker 心跳失效的任务转回 `queued`，并增加 `attempt_count`。达到最大尝试次数后标为 `failed`，防止无限重试。

### 6.2 图片状态和阶段

图片状态：

```text
queued -> running -> succeeded
                  -> failed
```

当前阶段：

```text
pending
object_detection
anomaly_classification
rendering
complete
```

单张图片失败不会终止整个批次。Worker 记录 `error_code` 和面向用户的简洁 `error_message`，然后继续处理下一张。

## 7. 推理流程

Worker 启动时执行模型自检：

1. 检查三个权重文件存在且可读。
2. 加载 YOLO11l-OBB。
3. 加载 label3 ResNet18。
4. 加载 label5 ResNet18。
5. 验证分类 checkpoint 的类别包含 `OK` 和 `NG`。
6. 注册模型名称、版本和权重校验值。
7. 将 Worker 标记为 ready。

每张图片执行：

```text
读取原图
  -> YOLO11l-OBB 推理
  -> 保存所有旋转框、类别和置信度
  -> 对 label3 选择检测置信度最高的框
  -> 对 label5 选择检测置信度最高的框
  -> 旋转矫正裁剪
  -> 对应 ResNet18 OK/NG 推理
  -> 调用可选颜色分类接口
  -> 聚合整图结论
  -> 生成结果图
  -> 提交数据库事务并更新进度
```

整图结论规则沿用当前管线：

- 任一受支持区域为 `NG`，整图为 `NG`。
- label3、label5 均存在且均为 `OK`，整图为 `OK`。
- 必需区域缺失、分类失败或结果不完整，整图为 `UNKNOWN`，不能误判为 `OK`。

## 8. 颜色分类扩展接口

颜色分类器使用可选协议，不耦合到现有 ResNet18：

```python
class ColorClassifier(Protocol):
    def predict(self, crop, region_label: str) -> ColorPrediction | None:
        ...
```

标准返回结构：

```json
{
  "color": "蓝色",
  "confidence": 0.961,
  "model_name": "terminal-color-classifier",
  "model_version": "v1"
}
```

没有配置颜色模型时返回 `null`，并且不影响 OK/NG 推理、整图结论或任务成功状态。

## 9. PostgreSQL 数据设计

### 9.1 `inspection_tasks`

| 字段 | 说明 |
| --- | --- |
| `id` | UUID 主键 |
| `display_id` | 面向用户的任务编号，如 `T20260919-001` |
| `name` | 可选任务名称 |
| `status` | 任务状态枚举 |
| `current_stage` | 当前整体阶段 |
| `total_images` | 图片总数 |
| `completed_images` | 已结束图片数，包括成功和失败 |
| `succeeded_images` | 成功数 |
| `failed_images` | 失败数 |
| `attempt_count` | 任务领取/恢复次数 |
| `worker_id` | 当前 Worker 标识 |
| `heartbeat_at` | Worker 最后心跳 |
| `detector_model_id` | 本次任务使用的检测模型记录 |
| `created_at` / `started_at` / `finished_at` | 时间戳 |
| `note` | 备注 |

### 9.2 `inspection_images`

| 字段 | 说明 |
| --- | --- |
| `id` | UUID 主键 |
| `task_id` | 所属任务 |
| `sequence_no` | 批次内顺序 |
| `original_filename` | 上传时文件名，仅用于显示 |
| `stored_filename` | 服务端生成的安全文件名 |
| `original_path` | 原图相对路径 |
| `result_path` | 结果图相对路径 |
| `status` / `stage` | 图片状态和阶段 |
| `overall_result` | `OK/NG/UNKNOWN` |
| `width` / `height` / `size_bytes` | 文件元数据 |
| `error_code` / `error_message` | 失败原因 |
| `created_at` / `started_at` / `finished_at` | 时间戳 |

### 9.3 `detections`

每个旋转框保存：图片 ID、区域标签、检测置信度、四个点坐标、是否被选为分类框、裁剪图相对路径。

四点坐标使用 PostgreSQL `jsonb` 保存，API 统一返回：

```json
[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
```

### 9.4 `classification_results`

| 字段 | 说明 |
| --- | --- |
| `detection_id` | 对应检测框 |
| `classifier_type` | `anomaly` 或 `color` |
| `predicted_label` | `OK/NG` 或颜色名称 |
| `confidence` | 预测置信度 |
| `probabilities` | 可选 `jsonb` 概率明细 |
| `model_id` | 使用的模型版本 |

### 9.5 `model_registry`

保存模型类型、名称、版本、权重相对路径、SHA-256 校验值、启用状态和加载配置。任务创建时绑定当前启用模型，历史结果因此能追溯到具体权重。

## 10. 文件存储设计

持久化根目录由 `INSPECTION_STORAGE_ROOT` 配置，例如：

```text
var/terminal-inspection/
  tasks/
    2026/09/19/<task-uuid>/
      originals/
      crops/
      results/
```

安全要求：

- 服务端使用 UUID 生成存储文件名，不信任用户文件名。
- 只允许经过白名单验证的图片扩展名和实际可解码图片。
- 所有解析后的路径必须位于存储根目录内，防止路径穿越。
- 数据库只存相对路径。
- 文件写入先进入临时名，成功后原子重命名。
- 数据库提交失败时清理本次未登记的临时文件。

## 11. API 设计

第一版 API 前缀为 `/api/v1`。

| 方法与路径 | 用途 |
| --- | --- |
| `GET /health` | API、数据库、Worker 和模型就绪状态 |
| `POST /tasks` | 创建任务并上传 1-100 张图片 |
| `GET /tasks` | 按状态、时间和关键词分页查询任务 |
| `GET /tasks/{task_id}` | 查询任务详情与聚合进度 |
| `GET /tasks/{task_id}/images` | 分页查询批次图片及状态 |
| `GET /images/{image_id}` | 查询单张图片、检测框和分类结果 |
| `GET /images/{image_id}/original` | 返回原图 |
| `GET /images/{image_id}/result` | 返回结果图 |
| `POST /images/{image_id}/retry` | 仅重新处理失败图片 |

上传接口使用 multipart/form-data。创建任务成功后立即返回 `202 Accepted` 和任务 ID，不等待推理结束。

前端在任务运行期间每 1 秒查询任务详情；任务进入终态后停止轮询。页面不可见时降低轮询频率，避免无意义请求。

## 12. PostgreSQL 队列领取

Worker 使用数据库事务领取最早的 `queued` 任务：

```sql
SELECT id
FROM inspection_tasks
WHERE status = 'queued'
ORDER BY created_at
FOR UPDATE SKIP LOCKED
LIMIT 1;
```

领取后在同一事务中将任务改为 `running` 并记录 Worker。即使未来增加多个 Worker，也不会重复领取同一任务。

数据库队列适合当前单用户和单 Worker 场景。如果未来出现大量并发任务、优先级队列或跨服务器 Worker，再评估迁移到 Redis/Celery；第一版不提前引入。

## 13. 错误处理与恢复

| 场景 | 行为 |
| --- | --- |
| 权重缺失或加载失败 | `/health` 返回模型未就绪，禁止开始检测并显示明确原因 |
| PostgreSQL 不可用 | API 返回服务不可用，不接受新任务 |
| 单张图片损坏 | 标记该图片失败，继续处理批次其他图片 |
| label3/label5 未检测到 | 图片结果为 `UNKNOWN` 并记录缺失区域警告 |
| 分类推理异常 | 对应区域显示分类失败，图片结果为 `UNKNOWN` |
| 结果图写入失败 | 单张图片失败，不写入伪成功记录 |
| API 重启 | 已上传文件和任务仍保留；Worker 继续运行或恢复任务 |
| Worker 重启 | 通过心跳和重试次数恢复未完成任务 |

用户界面展示可操作的错误信息，不直接暴露 Python traceback、服务器绝对路径或数据库异常细节。

## 14. 配置与部署

关键配置通过环境变量提供：

```text
DATABASE_URL
INSPECTION_STORAGE_ROOT
DETECTOR_WEIGHTS
LABEL3_CLASSIFIER_WEIGHTS
LABEL5_CLASSIFIER_WEIGHTS
DETECTION_DEVICE
CLASSIFICATION_DEVICE
DETECTION_IMGSZ=1280
CLASSIFICATION_IMGSZ=224
DETECTION_CONFIDENCE=0.25
MAX_IMAGES_PER_TASK=100
```

生产部署进程：

```text
nginx
frontend static build
api service
inference-worker service
postgresql
```

API 与 Worker 通过 systemd 或 Docker Compose 设置自动重启。权重和任务文件目录必须挂载到持久化磁盘，并纳入服务器备份策略。

## 15. 测试与验收

### 15.1 后端单元测试

- 状态机合法与非法转换。
- 任务进度聚合。
- 文件名与路径安全。
- 1张和100张上传边界。
- OK/NG/UNKNOWN 聚合规则。
- 颜色分类器为 `null` 时的兼容行为。
- Worker 任务领取与过期任务恢复。

### 15.2 API 集成测试

- 创建任务、分页查询、详情查询。
- 上传非法文件、超过100张、损坏图片。
- 原图和结果图访问。
- 部分失败任务统计。
- 失败图片重新检测。

### 15.3 Worker 测试

- 使用伪模型验证 detection -> classification -> render 顺序和进度更新。
- 使用小型真实样本与真实权重做 smoke test。
- Worker 中断后重启，确认任务可恢复且不会生成重复结果。

### 15.4 前端测试

- 文件选择、拖拽、移除与100张上限。
- 任务状态筛选与展开。
- 两阶段进度显示。
- 批量图片切换和原图/结果图对比。
- OK 绿色、NG 红色、未支持分类灰色。
- 页面刷新后恢复当前任务。
- API、模型或单张图片失败时的错误状态。

### 15.5 第一版验收标准

1. 可一次上传 1-100 张支持格式的图片。
2. 点击开始后立即获得任务编号并看到真实进度。
3. 每张图片依次完成 YOLO 检测及 label3/label5 分类。
4. 结果图颜色和文字符合已确认规则。
5. 单张失败不影响批次其余图片。
6. 服务器服务重启后历史任务、原图和结果图仍可打开。
7. 缺少权重时系统明确显示未就绪，而不是接受后静默失败。
8. 当前未配置颜色模型时，系统仍完整运行且不显示虚构颜色。

## 16. 第一版不做

- 用户登录、权限管理和多租户。
- 多 GPU 调度与多 Worker 并发。
- WebSocket 实时推送。
- 在线训练、模型上传或模型切换后台。
- 自动删除历史任务。
- 统计分析大屏和趋势图。
- 人工在线修改旋转框或异常标签。
- 未经模型支持的颜色预测。

这些能力不会被当前数据结构阻断，但只有出现明确生产需求后再设计实现。
