# 端子检测 Web 界面与运行稳定性改进实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保留 React + FastAPI + PostgreSQL + 单 Worker 架构的前提下，修复数据库连接耗尽和历史页面503，补充必填操作员，并改进任务预览、完整图片展示、标注位置、响应式布局和Worker日志。

**Architecture:** 新任务通过 multipart 表单写入可空的数据库 `operator` 字段，但 API 对新建请求强制非空；请求级 SQLAlchemy 会话由异步依赖可靠回滚和关闭。前端保留最后一次成功数据并对轮询失败退避，现有结果图渲染器调整标签锚点，Worker 使用标准 logging 输出精简阶段日志。

**Tech Stack:** Python 3.11、FastAPI、SQLAlchemy 2.x、Alembic、PostgreSQL、Pillow/OpenCV、React 19、TypeScript 5、Vite 7、Vitest、Testing Library。

**Spec:** `docs/superpowers/specs/2026-09-20-terminal-inspection-web-hardening-design.md`

## Global Constraints

- 操作员对所有新任务必填，但不得在浏览器中记住上一次输入。
- 现有历史任务允许 `operator = NULL`，页面显示 `-`。
- API 或数据库暂时失败时不得清空已经显示的任务、图片或结果。
- 不扩大连接池来掩盖会话泄漏；每个请求成功、失败和取消后都必须归还连接。
- 纵向图片必须完整显示，不改变检测坐标、模型输入或预测结果。
- 标签优先位于框右侧，空间不足时回退到左侧，并限制在图片边界内。
- Worker只记录模型加载、任务开始、每图阶段、每图结果和任务汇总，不逐框打印。
- 保留短轮询、单 Worker、PostgreSQL 队列，不引入 WebSocket、Redis 或 Celery。
- Git 提交信息使用中文。
- 不改动或提交 `datasets/obb_thin_thick/.DS_Store`。

## Review Focus

- 仅含空白字符的操作员必须在写文件和建任务前返回 422，不能留下暂存目录。
- 旧任务没有操作员时仍可序列化和打开，不能因数据库迁移产生500。
- 同时请求列表和任务详情时，请求完成后连接池 checked-out 数量必须回落，不能出现长期 `idle in transaction`。
- 刷新历史列表或当前详情失败时，用户仍应看到上一次成功内容，且不会每秒无限重试。
- 极窄检测框靠近图片右边缘时，长标签必须回退并保持在画布内，而不是遮挡框主体或越界。

---

### Task 1: 为任务增加必填操作员并兼容旧数据

**Files:**
- Create: `migrations/versions/_20260920_02_add_task_operator.py`
- Modify: `terminal_web/models.py`
- Modify: `terminal_web/schemas.py`
- Modify: `terminal_web/repositories.py`
- Modify: `terminal_web/api/tasks.py`
- Modify: `terminal_web/api/presenters.py`
- Test: `tests/web/test_api.py`
- Test: `tests/web/test_repositories.py`

**Interfaces:**
- Consumes: multipart `operator`, `name`, `note` 和现有 `TaskRepository.create_task(...)`。
- Produces: `InspectionTask.operator: str | None`、`TaskSummary.operator: str | None`、`TaskRepository.create_task(..., operator: str)`。

- [ ] **Step 1: 编写操作员 API 失败与兼容性测试**

在 `tests/web/test_api.py` 中把现有成功创建请求补充 `data={"operator": "张三"}`，并新增：

```python
def test_create_task_requires_non_blank_operator(self):
    for data in ({}, {"operator": "   "}):
        response = self.client.post(
            "/api/v1/tasks", files=self.valid_files(1), data=data
        )
        self.assertEqual(response.status_code, 422, response.text)
    self.assertEqual(self.count_tasks(), 0)

def test_create_task_persists_operator_name_and_note(self):
    response = self.client.post(
        "/api/v1/tasks",
        files=self.valid_files(1),
        data={"operator": " 张三 ", "name": "早班", "note": "首件"},
    )
    self.assertEqual(response.status_code, 202, response.text)
    payload = response.json()
    self.assertEqual(payload["operator"], "张三")
    self.assertEqual(payload["name"], "早班")
    self.assertEqual(payload["note"], "首件")

def test_legacy_task_without_operator_is_readable(self):
    with self.session_factory() as session:
        task = InspectionTask(display_id="T-legacy", operator=None)
        session.add(task)
        session.commit()
        task_id = task.id
    response = self.client.get(f"/api/v1/tasks/{task_id}")
    self.assertEqual(response.status_code, 200, response.text)
    self.assertIsNone(response.json()["operator"])
```

- [ ] **Step 2: 运行操作员测试确认 RED**

Run: `.venv/bin/python -m unittest tests.web.test_api.ApiTest.test_create_task_requires_non_blank_operator tests.web.test_api.ApiTest.test_create_task_persists_operator_name_and_note tests.web.test_api.ApiTest.test_legacy_task_without_operator_is_readable -v`

Expected: FAIL，因为接口尚未要求 `operator` 且响应中没有该字段。

- [ ] **Step 3: 增加数据库字段、迁移和 API 校验**

迁移使用：

```python
revision = "20260920_02"
down_revision = "20260920_01"

def upgrade() -> None:
    op.add_column("inspection_tasks", sa.Column("operator", sa.String(128), nullable=True))

def downgrade() -> None:
    op.drop_column("inspection_tasks", "operator")
```

模型与 schema 增加 `operator: str | None`；创建接口使用：

```python
operator: Annotated[str, Form(min_length=1, max_length=128)],
```

在任何文件写入前执行：

```python
operator = operator.strip()
if not operator:
    raise HTTPException(status_code=422, detail="操作员不能为空")
```

仓库创建方法接收并保存 `operator`，presenter 将其返回。

- [ ] **Step 4: 运行操作员测试确认 GREEN**

Run: `.venv/bin/python -m unittest tests.web.test_api tests.web.test_repositories -v`

Expected: PASS，所有新建任务测试显式传入操作员，旧任务空值可读取。

- [ ] **Step 5: 验证迁移可升级和降级**

Run: `.venv/bin/alembic upgrade head`

Expected: 升级到 `20260920_02`。

Run: `.venv/bin/alembic downgrade 20260920_01 && .venv/bin/alembic upgrade head`

Expected: 两条命令均退出0，最终恢复到 `20260920_02`。

- [ ] **Step 6: 提交任务元数据变更**

```bash
git add migrations/versions/_20260920_02_add_task_operator.py terminal_web/models.py terminal_web/schemas.py terminal_web/repositories.py terminal_web/api/tasks.py terminal_web/api/presenters.py tests/web/test_api.py tests/web/test_repositories.py
git commit -m "功能：为检测任务增加必填操作员"
```

### Task 2: 根治请求数据库会话泄漏和503

**Files:**
- Modify: `terminal_web/api/dependencies.py`
- Modify: `terminal_web/database.py`
- Modify: `terminal_web/api/app.py`
- Create: `tests/web/test_dependencies.py`
- Test: `tests/web/test_api.py`
- Test: `tests/web/test_config.py`

**Interfaces:**
- Consumes: `request.app.state.session_factory()` 和同步 SQLAlchemy `Session`。
- Produces: `async def get_session(...) -> AsyncGenerator[Session, None]`，保证 rollback/close；`build_session_factory(..., pool_timeout_seconds=...)`。

- [ ] **Step 1: 写会话成功与异常释放测试**

在 `tests/web/test_dependencies.py` 使用记录调用次数的 FakeSession 和 Request stub，按期望的异步生成器接口测试：

```python
class FakeSession:
    def __init__(self):
        self.rollbacks = 0
        self.closes = 0
        self.active = True
    def rollback(self):
        self.rollbacks += 1
        self.active = False
    def close(self):
        self.closes += 1
    def in_transaction(self):
        return self.active

async def consume_dependency(generator, *, fail=False):
    session = await anext(generator)
    if fail:
        try:
            await generator.athrow(RuntimeError("boom"))
        except RuntimeError:
            pass
    else:
        await generator.aclose()
    return session

def test_session_dependency_rolls_back_and_closes_after_response(self):
    fake = FakeSession()
    request = SimpleNamespace(app=SimpleNamespace(
        state=SimpleNamespace(session_factory=lambda: fake)
    ))
    session = asyncio.run(consume_dependency(get_session(request)))
    self.assertIs(session, fake)
    self.assertEqual(fake.rollbacks, 1)
    self.assertEqual(fake.closes, 1)

def test_session_dependency_rolls_back_and_closes_after_exception(self):
    fake = FakeSession()
    request = SimpleNamespace(app=SimpleNamespace(
        state=SimpleNamespace(session_factory=lambda: fake)
    ))
    asyncio.run(consume_dependency(get_session(request), fail=True))
    self.assertEqual(fake.rollbacks, 1)
    self.assertEqual(fake.closes, 1)
```

在 `tests/web/test_api.py` 使用带 `QueuePool(pool_size=2, max_overflow=0)` 的临时 SQLite 文件创建独立 app，连续请求30次列表和详情后断言 `engine.pool.checkedout() == 0`。

- [ ] **Step 2: 运行连接释放测试确认 RED**

Run: `.venv/bin/python -m unittest tests.web.test_dependencies tests.web.test_api -v`

Expected: `test_dependencies` FAIL，因为当前 `get_session` 是同步生成器，不能按异步依赖生命周期消费。

- [ ] **Step 3: 实现可靠的异步依赖清理**

```python
async def get_session(request: Request) -> AsyncGenerator[Session, None]:
    session = request.app.state.session_factory()
    try:
        yield session
    except BaseException:
        session.rollback()
        raise
    finally:
        if session.in_transaction():
            session.rollback()
        session.close()
```

保持 endpoint 和 repository 为同步 SQLAlchemy 调用；依赖清理在事件循环侧执行，避免线程池等待连接时阻塞同步生成器 finalizer。

数据库 engine 增加有限 `pool_timeout`，但不扩大默认池：

```python
engine = create_engine(database_url, pool_pre_ping=True, pool_timeout=10)
```

SQLite/StaticPool 测试工厂仍由测试直接注入，不经过生产 engine 配置。

- [ ] **Step 4: 为数据库异常增加服务端日志**

在 `terminal_web/api/app.py` 定义模块 logger，异常处理器调用：

```python
logger.exception("database request failed path=%s", request.url.path, exc_info=exc)
```

响应继续只返回 `{"detail": "database is unavailable"}`，不暴露堆栈。

- [ ] **Step 5: 运行后端接口和配置测试确认 GREEN**

Run: `.venv/bin/python -m unittest tests.web.test_dependencies tests.web.test_api tests.web.test_config tests.web.test_postgres_integration -v`

Expected: PASS；如本机未启用 PostgreSQL，集成测试按既有条件明确 SKIP，而非失败。

- [ ] **Step 6: 提交连接管理修复**

```bash
git add terminal_web/api/dependencies.py terminal_web/database.py terminal_web/api/app.py tests/web/test_dependencies.py tests/web/test_api.py tests/web/test_config.py
git commit -m "修复：可靠释放API数据库连接"
```

### Task 3: 增加前端任务字段和明确预览入口

**Files:**
- Modify: `web_frontend/src/api/types.ts`
- Modify: `web_frontend/src/api/client.ts`
- Modify: `web_frontend/src/api/client.test.ts`
- Modify: `web_frontend/src/components/UploadPanel.tsx`
- Modify: `web_frontend/src/components/TaskTable.tsx`
- Modify: `web_frontend/src/components/TaskDetails.tsx`
- Modify: `web_frontend/src/pages/TasksPage.tsx`
- Test: `web_frontend/src/pages/TasksPage.test.tsx`
- Test: `web_frontend/src/components/TaskTable.test.tsx`

**Interfaces:**
- Consumes: API `TaskSummary.operator` 与 multipart `operator/name/note`。
- Produces: `CreateTaskMetadata = { operator: string; name?: string; note?: string }`；任务表每行独立 `预览 <displayId>` 按钮。

- [ ] **Step 1: 写 multipart 和任务表单 RED 测试**

在 `client.test.ts` 断言 FormData：

```typescript
await apiClient.createTask([file], { operator: "张三", name: "早班", note: "首件" });
const body = fetchMock.mock.calls[0][1].body as FormData;
expect(body.get("operator")).toBe("张三");
expect(body.get("name")).toBe("早班");
expect(body.get("note")).toBe("首件");
```

在 `TasksPage.test.tsx` 新增：

```typescript
it("requires operator and clears metadata only after creation succeeds", async () => {
  const api = fakeClient();
  render(<TasksPage client={api} />);
  fireEvent.change(screen.getByLabelText("选择图片"), { target: { files: files(1) } });
  expect(screen.getByRole("button", { name: "开始检测" })).toBeDisabled();
  await userEvent.type(screen.getByLabelText("操作员"), "张三");
  await userEvent.type(screen.getByLabelText("任务名称"), "早班");
  await userEvent.type(screen.getByLabelText("备注"), "首件");
  await userEvent.click(screen.getByRole("button", { name: "开始检测" }));
  expect(api.createTask).toHaveBeenCalledWith(expect.any(Array), {
    operator: "张三", name: "早班", note: "首件",
  });
  expect(screen.getByLabelText("操作员")).toHaveValue("");
});
```

另建一次全新的 render，断言操作员输入为空，证明没有 localStorage 记忆。

- [ ] **Step 2: 写任务预览和详情 RED 测试**

更新 fixture 增加 `operator: "张三"`，然后断言：

```typescript
expect(screen.getByText("T20260920-0001").tagName).not.toBe("BUTTON");
await user.click(screen.getByRole("button", { name: "预览 T20260920-0001" }));
expect(onPreview).toHaveBeenCalledWith(tasks[0]);
await user.click(screen.getByRole("button", { name: "展开任务 T20260920-0001" }));
expect(screen.getByText("张三")).toBeInTheDocument();
expect(screen.queryByText("存储标识")).not.toBeInTheDocument();
expect(screen.queryByText("running-id")).not.toBeInTheDocument();
```

- [ ] **Step 3: 运行前端定向测试确认 RED**

Run: `npm test -- --run src/api/client.test.ts src/pages/TasksPage.test.tsx src/components/TaskTable.test.tsx`

Workdir: `web_frontend`

Expected: FAIL，因为字段、预览按钮和真实操作员尚不存在。

- [ ] **Step 4: 实现任务字段和表单状态**

- `TaskSummary` 增加 `operator?: string | null`。
- `CreateTaskMetadata.operator` 设为必填。
- `apiClient.createTask` 始终 append 经 trim 的 operator。
- `TasksPage` 持有 `operator/name/note` 状态，将其传入 `UploadPanel`。
- `UploadPanel` 渲染有 label 的输入控件，并把 `operator.trim() === ""` 纳入开始按钮禁用条件。
- 创建成功后清空文件和全部元数据；失败时保留。
- 不读写 `localStorage` 或 `sessionStorage`。

- [ ] **Step 5: 实现预览列和详情字段**

- `TaskTable` 表头增加“操作”，ID 使用 `<span>`。
- 每行增加 `Eye` 图标按钮，accessible name 为 `预览 ${task.displayId}`。
- 展开行 `colSpan` 从9更新为10。
- `TaskDetails` 删除“存储标识”，将“创建人”改为“操作员”，值为 `task.operator || "-"`。

- [ ] **Step 6: 运行前端定向测试确认 GREEN**

Run: `npm test -- --run src/api/client.test.ts src/pages/TasksPage.test.tsx src/components/TaskTable.test.tsx`

Workdir: `web_frontend`

Expected: PASS。

- [ ] **Step 7: 提交前端任务信息和预览交互**

```bash
git add web_frontend/src/api/types.ts web_frontend/src/api/client.ts web_frontend/src/api/client.test.ts web_frontend/src/components/UploadPanel.tsx web_frontend/src/components/TaskTable.tsx web_frontend/src/components/TaskDetails.tsx web_frontend/src/pages/TasksPage.tsx web_frontend/src/pages/TasksPage.test.tsx web_frontend/src/components/TaskTable.test.tsx
git commit -m "功能：补充任务信息与预览入口"
```

### Task 4: 保留历史数据并为轮询失败退避

**Files:**
- Modify: `web_frontend/src/pages/HistoryPage.tsx`
- Modify: `web_frontend/src/pages/HistoryPage.test.tsx`
- Modify: `web_frontend/src/hooks/useActiveTask.ts`
- Create: `web_frontend/src/hooks/useActiveTask.test.tsx`

**Interfaces:**
- Consumes: `ApiClient.listTasks/getTask` 抛出的 Error 或 AbortError。
- Produces: 失败时保留既有 state；轮询延迟 `pollingDelay(hidden, consecutiveFailures)`，成功后失败次数归零。

- [ ] **Step 1: 写历史详情失败仍保留结果的 RED 测试**

```typescript
it("keeps the selected comparison when reopening fails", async () => {
  const api = client({
    getTask: vi.fn()
      .mockResolvedValueOnce(detail)
      .mockRejectedValueOnce(new Error("服务暂不可用")),
  });
  render(<HistoryPage client={api} />);
  await userEvent.click(await screen.findByRole("button", { name: "预览 T20260920-0100" }));
  expect(await screen.findByAltText("检测前原图")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "预览 T20260920-0100" }));
  expect(await screen.findByText("服务暂不可用")).toBeInTheDocument();
  expect(screen.getByAltText("检测前原图")).toBeInTheDocument();
});
```

- [ ] **Step 2: 写轮询指数退避 RED 测试**

```typescript
expect(pollingDelay(false, 0)).toBe(1_000);
expect(pollingDelay(false, 1)).toBe(2_000);
expect(pollingDelay(false, 4)).toBe(10_000);
expect(pollingDelay(true, 4)).toBe(10_000);
```

使用 fake timers 断言失败后2秒才再次请求，成功后恢复1秒。

- [ ] **Step 3: 运行历史与轮询测试确认 RED**

Run: `npm test -- --run src/pages/HistoryPage.test.tsx src/hooks/useActiveTask.test.tsx`

Workdir: `web_frontend`

Expected: FAIL，因为预览测试仍依赖任务ID按钮，轮询没有失败计数。

- [ ] **Step 4: 实现非破坏性错误状态和退避**

- `HistoryPage.loadTasks` 只在成功时替换 `tasks`。
- `openTask` 只在成功时替换 `selectedTask/selectedIndex`。
- 所有失败只设置错误提示，不清空列表和对比图。
- 使用新的显式预览按钮调用 `openTask`。
- `useActiveTask` 局部维护 `consecutiveFailures`，延迟为 `min(1000 * 2 ** failures, 10000)`；成功后置零；隐藏页面固定至少10秒。
- 终态任务继续完全停止轮询。

- [ ] **Step 5: 运行历史与轮询测试确认 GREEN**

Run: `npm test -- --run src/pages/HistoryPage.test.tsx src/hooks/useActiveTask.test.tsx`

Workdir: `web_frontend`

Expected: PASS。

- [ ] **Step 6: 提交历史容错**

```bash
git add web_frontend/src/pages/HistoryPage.tsx web_frontend/src/pages/HistoryPage.test.tsx web_frontend/src/hooks/useActiveTask.ts web_frontend/src/hooks/useActiveTask.test.tsx
git commit -m "修复：保留历史数据并降低失败轮询频率"
```

### Task 5: 完整展示图片、移动标签并改进响应式样式

**Files:**
- Modify: `terminal_web/inference/rendering.py`
- Test: `tests/web/test_rendering.py`
- Modify: `web_frontend/src/components/ImageComparison.tsx`
- Modify: `web_frontend/src/styles/tokens.css`
- Modify: `web_frontend/src/styles.css`
- Test: `web_frontend/src/components/TaskTable.test.tsx`

**Interfaces:**
- Consumes: `RegionPrediction.points`、Pillow `textbbox`、图片原始宽高、现有 CSS 类。
- Produces: `_label_origin(...) -> tuple[int, int]`；全宽主内容、完整图片画布、按钮 hover/active/focus 状态和窄屏换行。

- [ ] **Step 1: 写标签右侧定位和边界 RED 测试**

从渲染器导出或直接测试纯函数：

```python
def test_label_origin_prefers_right_side(self):
    self.assertEqual(
        label_origin(((10, 20), (60, 20), (60, 80), (10, 80)), 40, 18, 200, 120),
        (68, 41),
    )

def test_label_origin_falls_back_left_and_clamps_to_canvas(self):
    x, y = label_origin(
        ((150, 0), (195, 0), (195, 40), (150, 40)), 100, 24, 200, 80
    )
    self.assertGreaterEqual(x, 0)
    self.assertLessEqual(x + 100, 200)
    self.assertGreaterEqual(y, 0)
    self.assertLessEqual(y + 24, 80)
    self.assertLess(x, 150)
```

尺寸包含标签底色 padding，期望值以实现中统一的 `gap=8` 为准。

- [ ] **Step 2: 运行渲染测试确认 RED**

Run: `.venv/bin/python -m unittest tests.web.test_rendering -v`

Expected: FAIL，因为纯定位函数不存在。

- [ ] **Step 3: 实现标签定位并用于结果渲染**

实现纯函数：

```python
def label_origin(points, label_width, label_height, image_width, image_height, gap=8):
    left = max(0, int(min(x for x, _ in points)))
    right = min(image_width, int(max(x for x, _ in points)))
    top = max(0, int(min(y for _, y in points)))
    bottom = min(image_height, int(max(y for _, y in points)))
    x = right + gap
    if x + label_width > image_width:
        x = left - gap - label_width
    x = min(max(x, 0), max(image_width - label_width, 0))
    y = int((top + bottom - label_height) / 2)
    y = min(max(y, 0), max(image_height - label_height, 0))
    return x, y
```

`render_prediction` 使用该坐标绘制标签，框线和颜色逻辑不变。

- [ ] **Step 4: 运行渲染测试确认 GREEN**

Run: `.venv/bin/python -m unittest tests.web.test_rendering -v`

Expected: PASS，原有绿色/红色/灰色像素断言继续通过。

- [ ] **Step 5: 增加样式结构断言并确认 RED**

为 `ImageComparison` 增加宽高属性断言，确保浏览器知道原始比例：

```typescript
expect(screen.getByAltText("检测前原图")).toHaveAttribute("width", "1440");
expect(screen.getByAltText("检测前原图")).toHaveAttribute("height", "3072");
```

Run: `npm test -- --run src/pages/TasksPage.test.tsx`

Workdir: `web_frontend`

Expected: FAIL，因为 img 尚未输出 width/height。

- [ ] **Step 6: 实现完整图片、全宽布局和按钮交互**

- 图片元素输出原始 `width`、`height`。
- 删除 `--content-max` 对 `.app-shell__content` 的限制，内容宽度设为100%。
- `.comparison-card__canvas` 不使用固定高度和 `overflow:hidden`；图片使用 `width:auto; height:auto; max-width:100%; max-height:min(72vh,900px); object-fit:contain`。
- `@media (max-width: 980px)` 将对比区改为单列并移除第二列左边框。
- `@media (max-width: 760px)` 调整侧栏/内容、上传表单和任务区间距。
- 为 `.button`、`.icon-button`、`.preview-button`、分页按钮添加120ms左右 transition、hover、active、focus-visible；`:disabled` 禁止位移和阴影。
- 上传任务字段使用 `.task-metadata-form` 网格，在窄屏改为单列。

- [ ] **Step 7: 运行前端测试和构建确认 GREEN**

Run: `npm test -- --run`

Workdir: `web_frontend`

Expected: 全部 Vitest 测试 PASS。

Run: `npm run build`

Workdir: `web_frontend`

Expected: TypeScript 和 Vite 构建退出0。

- [ ] **Step 8: 提交渲染和响应式改进**

```bash
git add terminal_web/inference/rendering.py tests/web/test_rendering.py web_frontend/src/components/ImageComparison.tsx web_frontend/src/styles/tokens.css web_frontend/src/styles.css web_frontend/src/pages/TasksPage.test.tsx
git commit -m "改进：完整展示检测图片与标签"
```

### Task 6: 增加精简 Worker 运行日志

**Files:**
- Modify: `terminal_web/worker.py`
- Modify: `scripts/run_terminal_worker.py`
- Test: `tests/web/test_worker.py`
- Test: `tests/web/test_entrypoints.py`

**Interfaces:**
- Consumes: 任务ID、图片序号、文件名、阶段 callback、最终任务统计。
- Produces: `terminal_web.worker` 标准 logger 输出，不改变 Worker 返回值和数据库状态机。

- [ ] **Step 1: 写正常任务与失败任务日志 RED 测试**

```python
def test_worker_logs_compact_task_progress(self):
    task_id = self.make_task(1)
    with self.assertLogs("terminal_web.worker", level="INFO") as captured:
        self.worker(FakePipeline()).run_once()
    text = "\n".join(captured.output)
    self.assertIn("模型加载完成", text)
    self.assertIn(f"开始任务 task_id={task_id}", text)
    self.assertIn("[1/1] 目标检测 filename=image-0.png", text)
    self.assertIn("[1/1] 异常分类 filename=image-0.png", text)
    self.assertIn("任务完成", text)
    self.assertNotIn(str(self.storage.root), text)

def test_worker_logs_image_failure_without_stopping_batch(self):
    self.make_task(2)
    with self.assertLogs("terminal_web.worker", level="INFO") as captured:
        self.worker(FakePipeline(fail_calls={0})).run_once()
    text = "\n".join(captured.output)
    self.assertIn("[1/2] 失败", text)
    self.assertIn("[2/2] 完成", text)
```

- [ ] **Step 2: 运行 Worker 日志测试确认 RED**

Run: `.venv/bin/python -m unittest tests.web.test_worker -v`

Expected: FAIL，因为 Worker 尚未发出日志。

- [ ] **Step 3: 实现精简结构化日志**

- 在模块顶部定义 `logger = logging.getLogger(__name__)`。
- 模型成功加载记录一次 INFO；加载失败使用 `logger.exception`。
- 领取任务后记录任务ID和总图数。
- `_process_image` 接收当前序号和总数；初始进入目标检测时记录一行，stage callback 只在阶段实际变化时记录异常分类，避免重复打印目标检测。
- 单图成功记录结果，失败记录 `logger.exception`，消息中只含文件名和异常类型。
- 重新计算任务状态后记录成功数、失败数和总耗时。

- [ ] **Step 4: 配置 Worker 入口日志格式**

在 `main()` 开始处调用：

```python
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
```

补充入口测试，patch `logging.basicConfig` 后断言被调用一次。

- [ ] **Step 5: 运行 Worker 和入口测试确认 GREEN**

Run: `.venv/bin/python -m unittest tests.web.test_worker tests.web.test_entrypoints -v`

Expected: PASS。

- [ ] **Step 6: 提交 Worker 日志**

```bash
git add terminal_web/worker.py scripts/run_terminal_worker.py tests/web/test_worker.py tests/web/test_entrypoints.py
git commit -m "改进：增加Worker任务进度日志"
```

### Task 7: 全量验证与真实浏览器验收

**Files:**
- Modify only if verification reveals a tested defect.

**Interfaces:**
- Consumes: Task 1-6 的 API、UI、渲染和日志行为。
- Produces: 可复现的全量测试、构建、迁移和浏览器验收证据。

- [ ] **Step 1: 运行完整 Python 测试套件**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Expected: 所有可运行测试 PASS；依赖外部 PostgreSQL/GPU 的测试只允许按既有条件明确 SKIP。

- [ ] **Step 2: 运行完整前端测试和构建**

Run: `npm test -- --run && npm run build && npm run test:sites`

Workdir: `web_frontend`

Expected: 测试、TypeScript/Vite 构建和 Sites worker 测试全部退出0。

- [ ] **Step 3: 检查迁移状态和代码差异**

Run: `.venv/bin/alembic current && .venv/bin/alembic heads && git diff --check`

Expected: current 与 heads 都为 `20260920_02`，`git diff --check` 无输出。

- [ ] **Step 4: 启动本地 API、Worker 和前端并进行浏览器验收**

使用仓库现有启动脚本运行服务，并在浏览器依次验证：

1. 2164px 宽屏下内容填满侧边栏右侧。
2. 1440px 常规桌面下任务信息、表格和对比图无重叠。
3. 760px 窄窗口下表单和对比图单列，表格可横向滚动。
4. 未填写操作员不能创建任务；填写后任务响应和详情正确。
5. 纵向端子原图和结果图完整显示。
6. 每行通过“预览”打开结果，任务ID不可点击，详情无存储标识。
7. 模拟 API 短暂失败时历史列表和当前结果保持可见。
8. Worker终端日志符合精简格式。

- [ ] **Step 5: 记录完成状态**

确认 `git status --short` 仅允许存在用户原有的 `datasets/obb_thin_thick/.DS_Store`，不得把它纳入任何提交。
