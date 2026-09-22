# 检测任务界面简化与模型身份隐藏 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除任务元数据及公开模型身份，让检测任务页只承担上传和当前结果展示，让全部历史浏览集中到检测历史页。

**Architecture:** 通过一条破坏性 Alembic 迁移永久删除任务元数据列，并同步收紧 ORM、仓储、API schema 和 presenter。前端以新的最小公开契约为准，检测任务页移除历史列表和元数据状态，检测历史页移除模型列及展开详情，服务内部模型注册与推理关联保持不变。

**Tech Stack:** Python 3.11、FastAPI、Pydantic 2、SQLAlchemy 2、Alembic、PostgreSQL、React 19、TypeScript、Vitest、Testing Library、Vite。

**Spec:** `docs/superpowers/specs/2026-09-22-simplify-task-ui-and-hide-model-identity-design.md`

## Global Constraints

- `inspection_tasks.operator`、`inspection_tasks.name`、`inspection_tasks.note` 及历史值必须永久删除。
- 任务 ID、图片、检测结果、状态、时间和文件持久化路径必须保持可用。
- `model_registry`、`detector_model_id`、内部 readiness 文件及 Worker 模型日志必须保留。
- 网页 HTML 和公开 JSON 响应均不得包含模型名称、版本或哈希。
- 检测任务页不得查询或展示历史任务列表。
- 历史任务只在检测历史页展示，仍支持筛选、搜索、预览和删除。
- API、Worker 和前端必须作为同一版本部署，部署继续通过 `alembic upgrade head` 执行迁移。
- 不新增运行时依赖，不修改模型推理和渲染算法。

## Review Focus

- 旧客户端仍提交 `operator`、`name`、`note` 时，新 API 应忽略这些多余表单字段且不持久化。
- 迁移已有任务时，仅删除三列，任务、图片与检测结果行必须保留。
- 使用原始图片文件名搜索时，后端命中结果不得再被前端二次过滤掉。
- 任一服务未就绪时，开始按钮必须禁用且只能显示通用错误，不能暴露失败模型名称。
- 所有任务、健康检查、分类响应和删除卡片都不得残留模型身份或已删除任务元数据。

---

### Task 1: 永久删除任务元数据并收紧公开 API

**Files:**
- Create: `migrations/versions/_20260922_03_remove_task_metadata.py`
- Create: `tests/web/test_task_metadata_migration.py`
- Modify: `terminal_web/models.py`
- Modify: `terminal_web/repositories.py`
- Modify: `terminal_web/schemas.py`
- Modify: `terminal_web/api/tasks.py`
- Modify: `terminal_web/api/presenters.py`
- Modify: `terminal_web/api/health.py`
- Modify: `terminal_web/readiness.py`
- Modify: `tests/web/test_api.py`
- Modify: `tests/web/test_repositories.py`
- Modify: `tests/web/test_end_to_end.py`

**Interfaces:**
- Consumes: `TaskRepository.create_task(task_id, display_id, images)`；内部 `ModelHealth` 仍供模型注册和 readiness 文件使用。
- Produces: 不含 `name`、`operator`、`note`、`detectorModel` 的 `TaskSummary`；不含 `models` 的 `HealthResponse`；不含 `modelName`、`modelVersion` 的 `ClassificationResponse`。

- [ ] **Step 1: 写迁移和 ORM 的失败测试**

创建真实 SQLite 临时表并通过 Alembic `Operations` 执行迁移函数，验证升级只删除三列且保留任务行，降级只恢复三个空的可空列：

```python
class TaskMetadataMigrationTest(unittest.TestCase):
    def test_upgrade_drops_metadata_without_deleting_tasks(self):
        with self.engine.begin() as connection:
            connection.execute(text(
                "CREATE TABLE inspection_tasks ("
                "id VARCHAR PRIMARY KEY, display_id VARCHAR NOT NULL, "
                "name VARCHAR, operator VARCHAR, note TEXT)"
            ))
            connection.execute(text(
                "INSERT INTO inspection_tasks VALUES "
                "('1', 'T-1', '早班', '张三', '首件')"
            ))
            operations = Operations(MigrationContext.configure(connection))
            with patch.object(migration, "op", operations):
                migration.upgrade()
            columns = {item[1] for item in connection.execute(text(
                "PRAGMA table_info(inspection_tasks)"
            ))}
            self.assertEqual(columns, {"id", "display_id"})
            self.assertEqual(
                connection.scalar(text("SELECT count(*) FROM inspection_tasks")),
                1,
            )
```

将仓储测试改为无元数据创建：

```python
def test_create_task_persists_required_task_fields(self):
    task = self.repo.create_task(uuid.uuid4(), "T-create", [])
    self.assertEqual(task.display_id, "T-create")
    self.assertEqual(task.total_images, 0)
    self.assertNotIn("operator", InspectionTask.__table__.columns)
    self.assertNotIn("name", InspectionTask.__table__.columns)
    self.assertNotIn("note", InspectionTask.__table__.columns)
```

- [ ] **Step 2: 运行迁移和仓储测试，确认 RED**

Run:

```bash
.venv/bin/python -m unittest tests.web.test_task_metadata_migration tests.web.test_repositories
```

Expected: FAIL，原因是迁移文件不存在、旧 ORM 仍含三列且 `create_task` 仍要求 `operator`。

- [ ] **Step 3: 实现迁移、ORM 和仓储最小修改**

迁移必须使用当前 head `20260920_02`：

```python
revision = "20260922_03"
down_revision = "20260920_02"

def upgrade() -> None:
    op.drop_column("inspection_tasks", "operator")
    op.drop_column("inspection_tasks", "name")
    op.drop_column("inspection_tasks", "note")

def downgrade() -> None:
    op.add_column("inspection_tasks", sa.Column("note", sa.Text(), nullable=True))
    op.add_column("inspection_tasks", sa.Column("name", sa.String(length=255), nullable=True))
    op.add_column("inspection_tasks", sa.Column("operator", sa.String(length=128), nullable=True))
```

从 `InspectionTask` 删除三个 mapped columns，并把仓储签名收紧为：

```python
def create_task(
    self,
    task_id: uuid.UUID,
    display_id: str,
    images: Sequence[InspectionImage],
) -> InspectionTask:
```

任务搜索只保留 `display_id` 与关联图片 `original_filename`。

- [ ] **Step 4: 运行迁移和仓储测试，确认 GREEN**

Run:

```bash
.venv/bin/python -m unittest tests.web.test_task_metadata_migration tests.web.test_repositories
```

Expected: PASS。

- [ ] **Step 5: 写公开 API 契约的失败测试**

更新 `tests/web/test_api.py`，让任务创建无需 metadata，并明确断言敏感字段缺失：

```python
def test_create_task_requires_only_images_and_hides_internal_identity(self):
    response = self.client.post("/api/v1/tasks", files=self.valid_files(1))
    self.assertEqual(response.status_code, 202, response.text)
    payload = response.json()
    self.assertEqual(payload["totalImages"], 1)
    for key in ("operator", "name", "note", "detectorModel"):
        self.assertNotIn(key, payload)
```

旧客户端多余字段不得重新进入响应：

```python
def test_legacy_metadata_form_fields_are_ignored(self):
    response = self.client.post(
        "/api/v1/tasks",
        files=self.valid_files(1),
        data={"operator": "张三", "name": "早班", "note": "首件"},
    )
    self.assertEqual(response.status_code, 202, response.text)
    self.assertTrue({"operator", "name", "note"}.isdisjoint(response.json()))
```

新增健康检查和分类结果断言：

```python
def test_health_response_exposes_only_readiness_booleans(self):
    payload = self.client.get("/api/v1/health").json()
    self.assertEqual(
        set(payload),
        {"apiReady", "databaseReady", "workerReady", "modelsReady"},
    )
```

在 image detail presenter/API 测试中断言分类结果不存在 `modelName` 和 `modelVersion`。

- [ ] **Step 6: 运行 API 与端到端测试，确认 RED**

Run:

```bash
.venv/bin/python -m unittest tests.web.test_api tests.web.test_end_to_end
```

Expected: FAIL，旧接口仍要求操作员并返回任务及模型身份字段。

- [ ] **Step 7: 实现新的后端公开契约**

`create_task` 仅保留文件、依赖和 settings 参数：

```python
@router.post("", response_model=TaskDetail, status_code=status.HTTP_202_ACCEPTED)
async def create_task(
    files: Annotated[list[UploadFile], File()],
    repository: Annotated[TaskRepository, Depends(get_repository)],
    storage: Annotated[ArtifactStorage, Depends(get_storage)],
    readiness: Annotated[ReadinessProvider, Depends(get_readiness)],
    settings=Depends(get_app_settings),
) -> TaskDetail:
```

任务 presenter 只构造公开业务字段：

```python
return TaskSummary(
    id=task.id,
    display_id=task.display_id,
    status=task.status,
    current_stage=task.current_stage,
    total_images=task.total_images,
    completed_images=task.completed_images,
    succeeded_images=task.succeeded_images,
    failed_images=task.failed_images,
    created_at=task.created_at,
    started_at=task.started_at,
    finished_at=task.finished_at,
)
```

`HealthResponse` 删除 `models` 字段；`ReadinessStore.snapshot()` 仍解析内部 readiness 文件以计算状态，但不把模型列表放入返回对象。`ClassificationResponse` 与 `detection_response()` 删除模型名称和版本。删除仓储查询中不再需要的 `detector_model` eager loading。

- [ ] **Step 8: 更新所有 Python fixtures 并运行后端全量测试**

Run:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Expected: 全部通过；只允许已有 PostgreSQL 环境测试因未设置 `TEST_DATABASE_URL` 被跳过。

- [ ] **Step 9: 提交后端和迁移**

```bash
git add migrations/versions/_20260922_03_remove_task_metadata.py tests/web/test_task_metadata_migration.py terminal_web tests/web
git commit -m "功能：删除任务元数据并隐藏模型身份"
```

---

### Task 2: 简化检测任务创建与当前结果页面

**Files:**
- Modify: `web_frontend/src/api/types.ts`
- Modify: `web_frontend/src/api/client.ts`
- Modify: `web_frontend/src/api/client.test.ts`
- Modify: `web_frontend/src/pages/TasksPage.tsx`
- Modify: `web_frontend/src/pages/TasksPage.test.tsx`
- Modify: `web_frontend/src/components/UploadPanel.tsx`
- Modify: `web_frontend/src/components/ProgressPanel.tsx`
- Modify: `web_frontend/src/components/ImageComparison.tsx`
- Modify: `web_frontend/src/hooks/useActiveTask.test.tsx`
- Modify: `web_frontend/src/styles.css`

**Interfaces:**
- Consumes: Task 1 的最小 `HealthResponse`、`TaskSummary` 与无 metadata 创建任务接口。
- Produces: `apiClient.createTask(files, signal?)`；只包含上传、当前进度和当前结果的 `TasksPage`。

- [ ] **Step 1: 写检测任务页简化的失败测试**

更新 fixtures，使 `HealthResponse` 只有四个 readiness 布尔值、任务对象不含 metadata 和模型字段。新增/替换断言：

```tsx
it("creates a task from images without metadata or model identity", async () => {
  const api = fakeClient();
  const user = userEvent.setup();
  render(<TasksPage client={api} />);

  expect(await screen.findByText("系统已就绪")).toBeInTheDocument();
  expect(screen.queryByLabelText("操作员")).not.toBeInTheDocument();
  expect(screen.queryByLabelText("任务名称")).not.toBeInTheDocument();
  expect(screen.queryByLabelText("备注")).not.toBeInTheDocument();
  expect(screen.queryByText("YOLO11l-OBB")).not.toBeInTheDocument();

  fireEvent.change(screen.getByLabelText("选择图片"), {
    target: { files: files(1) },
  });
  await user.click(screen.getByRole("button", { name: "开始检测" }));
  expect(api.createTask).toHaveBeenCalledWith(expect.any(Array));
});
```

新增断言确保任务页没有历史表：

```tsx
expect(screen.queryByRole("region", { name: "检测任务列表" })).not.toBeInTheDocument();
expect(api.listTasks).not.toHaveBeenCalled();
```

未就绪测试只能看到通用文案：

```tsx
expect(await screen.findByText("系统未就绪")).toBeInTheDocument();
expect(screen.queryByText(/ResNet|YOLO/)).not.toBeInTheDocument();
expect(screen.getByRole("button", { name: "开始检测" })).toBeDisabled();
```

- [ ] **Step 2: 运行相关前端测试，确认 RED**

Run:

```bash
npm test -- --run src/api/client.test.ts src/pages/TasksPage.test.tsx
```

Expected: FAIL，旧页面仍要求 metadata、加载历史任务并显示模型名称。

- [ ] **Step 3: 收紧前端类型和 API 客户端**

删除 `ModelHealth`、`CreateTaskMetadata` 和公开模型字段。将创建方法改为：

```ts
createTask(files: File[], signal?: AbortSignal): Promise<TaskDetail> {
  const body = new FormData();
  files.forEach((file) => body.append("files", file));
  return request("/tasks", { method: "POST", body, signal });
}
```

同步更新所有 `ApiClient` test doubles，使签名与真实客户端一致。

- [ ] **Step 4: 实现任务页最小流程**

从 `TasksPage` 删除 `metadata`、`recentTasks`、历史加载 effect 和 `selectTask`。上传面板 props 收紧为：

```ts
interface UploadPanelProps {
  files: File[];
  onFilesChange: (files: File[]) => void;
  onStart: () => void;
  health: HealthResponse | null;
  loadingHealth: boolean;
  submitting: boolean;
}
```

按钮禁用条件改为：

```ts
const disabled = files.length === 0 || tooMany || !ready || submitting;
```

`TasksPage.start()` 直接调用 `client.createTask(files)`。页头只显示通用系统状态。`ProgressPanel` 删除模型引用。`ImageComparison` 删除 `colorModelAvailable` prop，并保留固定的“颜色分类：接口预留”。删除 metadata 和 model-reference 对应的无用 CSS。

- [ ] **Step 5: 运行检测任务页测试，确认 GREEN**

Run:

```bash
npm test -- --run src/api/client.test.ts src/pages/TasksPage.test.tsx src/hooks/useActiveTask.test.tsx
```

Expected: PASS。

- [ ] **Step 6: 提交检测任务页修改**

```bash
git add web_frontend/src/api web_frontend/src/pages/TasksPage.tsx web_frontend/src/pages/TasksPage.test.tsx web_frontend/src/components/UploadPanel.tsx web_frontend/src/components/ProgressPanel.tsx web_frontend/src/components/ImageComparison.tsx web_frontend/src/hooks/useActiveTask.test.tsx web_frontend/src/styles.css
git commit -m "界面：简化检测任务创建流程"
```

---

### Task 3: 将历史浏览集中到检测历史页面

**Files:**
- Delete: `web_frontend/src/components/TaskDetails.tsx`
- Modify: `web_frontend/src/components/TaskTable.tsx`
- Modify: `web_frontend/src/components/TaskTable.test.tsx`
- Modify: `web_frontend/src/components/DeleteTaskDialog.tsx`
- Modify: `web_frontend/src/pages/HistoryPage.tsx`
- Modify: `web_frontend/src/pages/HistoryPage.test.tsx`
- Modify: `web_frontend/src/styles.css`

**Interfaces:**
- Consumes: Task 1 的最小 `TaskSummary`，Task 2 的前端类型。
- Produces: 无展开详情和模型列的历史表；只含任务 ID 与图片数量的删除卡片。

- [ ] **Step 1: 写历史页面最小信息展示的失败测试**

删除 fixtures 中的 metadata 与 `detectorModel`，并新增断言：

```tsx
it("shows history actions without metadata or model identity", async () => {
  render(<HistoryPage client={client()} />);
  expect(await screen.findByText("T20260920-0100")).toBeInTheDocument();
  expect(screen.queryByRole("columnheader", { name: "模型" })).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /展开任务/ })).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: "预览 T20260920-0100" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "删除 T20260920-0100" })).toBeInTheDocument();
});
```

删除卡片断言只允许 ID 和图片数量：

```tsx
const dialog = screen.getByRole("dialog", { name: "确认删除任务" });
expect(within(dialog).getByText("T20260920-0100")).toBeInTheDocument();
expect(within(dialog).getByText("2 张")).toBeInTheDocument();
expect(within(dialog).queryByText("任务名称")).not.toBeInTheDocument();
```

为图片文件名搜索增加 API 行为测试，确保结果不被本地二次过滤：

```tsx
await user.type(screen.getByLabelText("搜索历史任务"), "terminal-ok.png");
await user.click(screen.getByRole("button", { name: "搜索" }));
await waitFor(() => expect(api.listTasks).toHaveBeenLastCalledWith(
  expect.objectContaining({ query: "terminal-ok.png" }),
));
expect(screen.getByText("T20260920-0100")).toBeInTheDocument();
```

- [ ] **Step 2: 运行历史组件测试，确认 RED**

Run:

```bash
npm test -- --run src/components/TaskTable.test.tsx src/pages/HistoryPage.test.tsx
```

Expected: FAIL，旧表仍有模型列、展开入口和任务名称。

- [ ] **Step 3: 实现历史表与删除卡片简化**

从 `TaskTable` 删除 `Fragment`、展开状态、展开按钮列、模型列和 `TaskDetails`。最终列顺序为：

```tsx
<tr>
  <th>#</th>
  <th>任务ID</th>
  <th>图像数量</th>
  <th>完成进度</th>
  <th>当前阶段</th>
  <th>状态</th>
  <th>创建时间</th>
  <th>操作</th>
</tr>
```

删除 `TaskDetails.tsx`。删除确认卡片 summary 只保留：

```tsx
<dl className="delete-dialog__summary">
  <div><dt>任务 ID</dt><dd>{task.displayId}</dd></div>
  <div><dt>图像数量</dt><dd>{task.totalImages} 张</dd></div>
</dl>
```

`HistoryPage` 删除 `useMemo` 和本地 metadata 过滤，直接把 API 返回的 `tasks` 交给 `TaskTable`；搜索 placeholder 改为“搜索任务 ID 或图片文件名”。清理展开详情对应 CSS。

- [ ] **Step 4: 运行历史组件测试，确认 GREEN**

Run:

```bash
npm test -- --run src/components/TaskTable.test.tsx src/pages/HistoryPage.test.tsx
```

Expected: PASS。

- [ ] **Step 5: 提交历史页面修改**

```bash
git add web_frontend/src/components/TaskTable.tsx web_frontend/src/components/TaskTable.test.tsx web_frontend/src/components/DeleteTaskDialog.tsx web_frontend/src/pages/HistoryPage.tsx web_frontend/src/pages/HistoryPage.test.tsx web_frontend/src/styles.css
git rm web_frontend/src/components/TaskDetails.tsx
git commit -m "界面：集中管理检测历史信息"
```

---

### Task 4: 契约泄露扫描与全量验证

**Files:**
- Modify only if verification reveals a concrete stale fixture or unused style; do not add new behavior.

**Interfaces:**
- Consumes: Tasks 1–3 的完整后端和前端契约。
- Produces: 可由现有 Docker 部署流程原子升级的最终版本。

- [ ] **Step 1: 扫描公开代码中的残留身份字段**

Run:

```bash
rg -n "operator|任务名称|操作员|备注|detectorModel|当前模型|modelName|modelVersion" terminal_web web_frontend/src tests/web
```

Expected: 仅允许内部模型注册、迁移降级列定义、测试旧客户端兼容输入和与本需求无关的 Python 类型运算符文本；网页生产组件和公开 schema 不得命中这些身份字段。

- [ ] **Step 2: 验证迁移链和 ORM head 一致**

Run:

```bash
DATABASE_URL=postgresql+psycopg://terminal:terminal@127.0.0.1:5432/terminal_inspection .venv/bin/alembic heads
DATABASE_URL=postgresql+psycopg://terminal:terminal@127.0.0.1:5432/terminal_inspection .venv/bin/alembic check
```

Expected: 唯一 head 为 `20260922_03`，`alembic check` 输出无新的升级操作。若本机测试数据库不可连接，只记录该环境限制，并通过 Task 1 的真实 Alembic Operations 测试验证迁移行为；不得对未知生产数据库执行降级。

- [ ] **Step 3: 运行 Python 全量测试**

Run:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

Expected: 全部通过，只有需要显式 PostgreSQL 测试地址的既有测试可以跳过。

- [ ] **Step 4: 运行前端全量测试、类型检查和生产构建**

Run:

```bash
npm test -- --run
npx tsc --noEmit
npm run build
```

Expected: Vitest 全部通过，TypeScript 无错误，Vite 生产构建完成。

- [ ] **Step 5: 检查补丁完整性**

Run:

```bash
git diff --check
git status --short
```

Expected: 无空白错误；不得追踪 `datasets/obb_thin_thick/.DS_Store`；只保留本计划产生的预期文件变更。

- [ ] **Step 6: 提交验证阶段发现的机械修复**

仅当 Step 1–5 发现并修复残留 fixture、类型或样式时执行：

```bash
git add terminal_web tests/web web_frontend/src
git commit -m "修复：清理旧任务字段残留"
```

如果没有额外修改，则不创建空提交。
