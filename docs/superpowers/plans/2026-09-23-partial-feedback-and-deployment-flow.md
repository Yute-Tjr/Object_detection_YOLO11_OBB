# 部分区域反馈与部署流程改进 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 固定反馈区域顺序，允许仅纠正部分区域并保存带来源的完整结果快照，同时把 Docker 镜像回退、密码规则和部署时用户创建调整为可用的生产流程。

**Architecture:** 前端只提交用户明确选择的人工项，FastAPI 根据持久化检测及分类记录补齐 `manual/model/unreviewed` 完整快照。Alembic 为反馈项增加来源并放宽判定为空；部署脚本把镜像获取封装为 Hub、国内源、本地缓存三阶段，并在迁移后通过标准输入调用现有用户管理命令。

**Tech Stack:** Python 3.11、FastAPI、SQLAlchemy 2、Alembic、PostgreSQL、Bash、Docker Compose、React 19、TypeScript、Vitest。

**Spec:** `docs/superpowers/specs/2026-09-23-partial-feedback-and-deployment-flow-design.md`

## Global Constraints

- 逻辑区域固定按 `label1`、`label2`、`label3`、`label4`、`label5`、`label6` 排序。
- 用户只提交人工处理项；服务端补齐完整快照并决定 `manual/model/unreviewed`。
- `manual` 必须有 `OK/NG`，人工 `label1` 必须有 `B/G/R/W`；`model/unreviewed` 的判定允许为空。
- 首次部署必须创建用户；更新部署仅在用户选择或同时提供非交互凭据时新增用户。
- 网页和数据库密码最低 6 位；网页登录密码不得写入 `.env`、日志或命令参数。
- 镜像回退顺序必须是 Docker Hub → 国内镜像 → 本地缓存，非仓库网络错误不得回退。
- 不处理既有未跟踪文件 `datasets/obb_thin_thick/.DS_Store`。

## Review Focus

- 模型输出未知判定或未知颜色时应降级为 `unreviewed` 或空字段，不能伪造人工值；Task 2 覆盖。
- 多个检测框映射到同一逻辑区域时排序必须稳定且检测 ID 仍逐项唯一；Task 2、Task 3 覆盖。
- 用户取消已有人工反馈后，服务端应恢复当前模型快照而不是保留旧人工值；Task 2、Task 3 覆盖。
- 国内镜像因网络失败后，本地缓存必须恢复默认基础镜像名，不能查找国内标签缓存；Task 6 覆盖。
- 用户创建失败不能泄露密码，且必须给出可恢复命令；Task 5 覆盖。

---

### Task 1: 反馈来源数据库迁移

**Files:**
- Create: `migrations/versions/_20260923_05_add_feedback_item_source.py`
- Modify: `terminal_web/models.py`
- Modify: `tests/web/test_auth_feedback_migration.py`

**Interfaces:**
- Produces: `ImageFeedbackItem.source: str`，允许为空的 `ImageFeedbackItem.verdict`。
- Produces: 数据库检查约束 `source IN ('manual','model','unreviewed')`。

- [ ] **Step 1: 编写失败的迁移和模型测试**

新增升级后字段及历史数据断言：

```python
columns = {item["name"]: item for item in inspector.get_columns("image_feedback_items")}
self.assertFalse(columns["source"]["nullable"])
self.assertTrue(columns["verdict"]["nullable"])
self.assertIn("source", {column.name for column in ImageFeedbackItem.__table__.columns})
```

- [ ] **Step 2: 验证测试因缺少迁移而失败**

Run: `.venv/bin/python -m unittest tests.web.test_auth_feedback_migration -v`

Expected: FAIL，缺少 revision `20260923_05` 或 `source` 字段。

- [ ] **Step 3: 实现迁移和模型**

迁移顺序：新增可空 `source`、把旧行更新为 `manual`、设为非空、放宽 `verdict`、增加检查约束。降级时先为所有空判定填入兼容值不可接受，因此若存在 `verdict IS NULL` 应明确拒绝 downgrade；正常升级不修改旧反馈含义。

模型使用：

```python
source: Mapped[str] = mapped_column(String(16), nullable=False)
verdict: Mapped[str | None] = mapped_column(String(2))
```

- [ ] **Step 4: 运行迁移测试**

Run: `.venv/bin/python -m unittest tests.web.test_auth_feedback_migration -v`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add migrations/versions/_20260923_05_add_feedback_item_source.py terminal_web/models.py tests/web/test_auth_feedback_migration.py
git commit -m "反馈：增加结果来源与可空判定"
```

---

### Task 2: 后端部分反馈与完整快照

**Files:**
- Modify: `terminal_web/feedback.py`
- Modify: `terminal_web/feedback_repository.py`
- Modify: `terminal_web/api/feedback.py`
- Modify: `terminal_web/schemas.py`
- Modify: `tests/web/test_feedback_api.py`

**Interfaces:**
- Produces: `FeedbackValue(detection_id, verdict, color, source)`。
- Produces: `feedback_region_order(region_label: str) -> tuple[int, str]`。
- API 请求仍为部分 `items` + `missedRegions`；响应项增加 `source` 且 `verdict` 可空。

- [ ] **Step 1: 编写失败测试**

覆盖：

```python
partial = {"items": [{"detectionId": self.detection_ids["label3"], "verdict": "NG"}], "missedRegions": []}
response = self.put(self.image_id, partial)
self.assertEqual(response.status_code, 200)
self.assertEqual([item["logicalRegion"] for item in response.json()["feedback"]["items"]], ["label1", "label3", "label5"])
self.assertEqual({item["logicalRegion"]: item["source"] for item in response.json()["feedback"]["items"]}, {"label1": "unreviewed", "label3": "manual", "label5": "model"})
```

另测空请求拒绝、重复/外部检测 ID、未知模型输出、取消人工项后恢复模型来源、区域稳定排序。

- [ ] **Step 2: 验证失败**

Run: `.venv/bin/python -m unittest tests.web.test_feedback_api -v`

Expected: FAIL，当前要求检测 ID 集合完全相等且无 `source`。

- [ ] **Step 3: 实现领域规则**

`validate_feedback` 改为要求提交 ID 是当前检测 ID 的无重复子集；完全没有人工项且没有漏检时拒绝。新增从 `Detection.classifications` 提取模型快照的函数，只接受 `OK/NG` 和 `B/G/R/W`。

```python
if detection.id in submitted_by_id:
    return FeedbackValue(detection.id, manual.verdict, manual.color, "manual")
if model_verdict is not None or model_color is not None:
    return FeedbackValue(detection.id, model_verdict, model_color, "model")
return FeedbackValue(detection.id, None, None, "unreviewed")
```

- [ ] **Step 4: 持久化和响应来源**

Repository 写入 `source`；API 对检测、快照项和漏检候选统一排序。Schema 将响应 `verdict` 改为可空并增加：

```python
source: Literal["manual", "model", "unreviewed"]
```

- [ ] **Step 5: 运行测试**

Run: `.venv/bin/python -m unittest tests.web.test_feedback_api tests.web.test_repositories -v`

Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add terminal_web/feedback.py terminal_web/feedback_repository.py terminal_web/api/feedback.py terminal_web/schemas.py tests/web/test_feedback_api.py
git commit -m "反馈：支持部分人工纠正和完整结果快照"
```

---

### Task 3: 前端固定排序与部分选择

**Files:**
- Modify: `web_frontend/src/api/types.ts`
- Modify: `web_frontend/src/components/FeedbackDialog.tsx`
- Modify: `web_frontend/src/components/FeedbackDialog.test.tsx`
- Modify: `web_frontend/src/styles.css`

**Interfaces:**
- Consumes: 响应项 `source` 以及可空 `verdict`。
- Produces: 仅包含用户启用为人工反馈的检测项的 `FeedbackUpdateRequest.items`。

- [ ] **Step 1: 编写失败的交互测试**

构造乱序 `label3,label1,label6,label2`，断言 DOM 顺序为 `label1,label2,label3,label6`。只启用 `label3` 并选择 `NG`，提交体必须是：

```typescript
{
  items: [{ detectionId: "d3", verdict: "NG" }],
  missedRegions: [],
}
```

另测 `manual` 回填、取消后不提交、`model/unreviewed` 来源提示、人工 `label1` 必须同时选择判定和颜色、空反馈被前端阻止。

- [ ] **Step 2: 验证失败**

Run: `PATH=/opt/homebrew/opt/node@24/bin:$PATH npm test -- --run src/components/FeedbackDialog.test.tsx`

Workdir: `web_frontend`

Expected: FAIL，当前顺序透传且强制所有区域填写。

- [ ] **Step 3: 实现草稿和排序**

草稿项加入 `manual: boolean`。只有历史来源为 `manual` 才默认启用。统一顺序：

```typescript
const LOGICAL_REGION_ORDER: LogicalRegion[] = ["label1", "label2", "label3", "label4", "label5", "label6"];
const orderedDetections = [...view.detections].sort((a, b) => LOGICAL_REGION_ORDER.indexOf(a.logicalRegion) - LOGICAL_REGION_ORDER.indexOf(b.logicalRegion));
```

每行提供“人工反馈”开关；关闭时清除草稿人工值并展示“沿用模型”或“暂无分类结果”。提交只映射 `manual=true` 的检测项。

- [ ] **Step 4: 调整样式和文案**

保持现有弹窗尺寸与工厂端简洁风格，人工项突出，来源使用短标签，不增加动画。

- [ ] **Step 5: 运行前端测试与类型检查**

Run: `PATH=/opt/homebrew/opt/node@24/bin:$PATH npm test -- --run src/components/FeedbackDialog.test.tsx`

Run: `PATH=/opt/homebrew/opt/node@24/bin:$PATH npx tsc --noEmit`

Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add web_frontend/src/api/types.ts web_frontend/src/components/FeedbackDialog.tsx web_frontend/src/components/FeedbackDialog.test.tsx web_frontend/src/styles.css
git commit -m "前端：按区域排序并支持部分反馈"
```

---

### Task 4: 密码最低长度调整为 6 位

**Files:**
- Modify: `terminal_web/auth.py`
- Modify: `tests/web/test_auth.py`
- Modify: `tests/web/test_manage_users.py`
- Modify: `deploy.sh`
- Modify: `tests/test_docker_deploy.py`

**Interfaces:**
- Produces: `PASSWORD_MIN_LENGTH = 6`。
- Produces: `validate_db_password` 接受 6 位以上 URL 安全密码。

- [ ] **Step 1: 修改测试为 5 位失败、6 位成功**

```python
self.assertRaises(ValueError, hash_password, "12345")
self.assertTrue(hash_password("123456").startswith("$argon2id$"))
```

部署测试断言 `Db_123` 通过，`Db_12` 失败，占位密码仍失败。

- [ ] **Step 2: 验证测试失败**

Run: `.venv/bin/python -m unittest tests.web.test_auth tests.web.test_manage_users tests.test_docker_deploy.DockerDeployScriptTest.test_database_password_requires_six_url_safe_characters -v`

Expected: FAIL，当前下限为 12。

- [ ] **Step 3: 修改最小长度和终端文案**

把认证常量和部署验证统一为 6，删除更新模式下针对少于 12 位的旧警告；继续拒绝占位符和 URL 非安全字符。

- [ ] **Step 4: 运行测试**

Run: `.venv/bin/python -m unittest tests.web.test_auth tests.web.test_manage_users tests.test_docker_deploy -v`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add terminal_web/auth.py deploy.sh tests/web/test_auth.py tests/web/test_manage_users.py tests/test_docker_deploy.py
git commit -m "部署：密码最低长度调整为六位"
```

---

### Task 5: 部署向导创建登录用户

**Files:**
- Modify: `deploy.sh`
- Modify: `tests/test_docker_deploy.py`

**Interfaces:**
- Consumes: `INITIAL_APP_USERNAME`、`INITIAL_APP_PASSWORD`（只读进程环境，不写 `.env`）。
- Produces: `prompt_initial_app_user`、`create_configured_app_user`。

- [ ] **Step 1: 编写失败的部署脚本测试**

扩展 fake docker 捕获标准输入，覆盖：首次交互必须收集用户；更新选择否不创建；更新选择是调用 `manage_users.py add`；`--yes` 首次缺少凭据失败；密码不出现在 `.env`、stdout、日志和 docker 命令行。

- [ ] **Step 2: 验证失败**

Run: `.venv/bin/python -m unittest tests.test_docker_deploy -v`

Expected: FAIL，当前仅打印示例命令。

- [ ] **Step 3: 实现安全收集和创建**

用户名在部署前收集，密码隐藏输入两次。首次部署强制；更新部署默认不新增。服务启动后执行：

```bash
printf '%s\n%s\n' "$INITIAL_APP_PASSWORD" "$INITIAL_APP_PASSWORD" \
  | compose exec -T api python scripts/manage_users.py add "$INITIAL_APP_USERNAME"
```

该调用不经 `run_logged_step` 写入输入；只记录成功或脱敏失败消息，并在函数返回后 `unset INITIAL_APP_PASSWORD`。

- [ ] **Step 4: 更新完成提示**

删除把 `USERNAME` 当成字面量复制的主提示；更新场景的已有用户名失败提示给出 `reset-password` 命令。

- [ ] **Step 5: 运行测试**

Run: `.venv/bin/python -m unittest tests.test_docker_deploy tests.web.test_manage_users -v`

Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add deploy.sh tests/test_docker_deploy.py
git commit -m "部署：首次创建用户并支持更新时新增"
```

---

### Task 6: 镜像回退顺序改为 Hub、国内源、本地缓存

**Files:**
- Modify: `deploy.sh`
- Modify: `tests/test_docker_deploy.py`

**Interfaces:**
- Produces: `reset_default_image_source`，`enable_domestic_image_source` 不再假设本地缓存已失败。
- Produces: `run_logged_step` 的构建网络错误三阶段回退。

- [ ] **Step 1: 重写失败测试的期望顺序**

fake docker 记录基础镜像环境和 build 命令，断言：

```text
build --pull api (Docker Hub)
build --pull api (m.daocloud.io)
build api (default local tag)
```

国内源成功时不得执行无 `--pull` 构建；非网络构建错误仍只尝试一次。

- [ ] **Step 2: 验证失败**

Run: `.venv/bin/python -m unittest tests.test_docker_deploy -v`

Expected: FAIL，当前顺序为 Hub、本地、国内。

- [ ] **Step 3: 实现三阶段状态机**

保存默认基础镜像名。Hub 网络失败后直接启用国内源；只有国内源日志也匹配仓库网络错误时才恢复默认镜像名并调用 `build_service_images cached`。本地失败后统一输出日志并终止。

PostgreSQL `up -d postgres` 使用同一阶段顺序；不要对数据库认证、Compose 配置或容器启动错误执行回退。

- [ ] **Step 4: 运行部署脚本测试和语法检查**

Run: `.venv/bin/python -m unittest tests.test_docker_deploy -v`

Run: `bash -n deploy.sh`

Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add deploy.sh tests/test_docker_deploy.py
git commit -m "部署：调整镜像源回退顺序"
```

---

### Task 7: 文档、迁移和端到端验证

**Files:**
- Modify: `README.md`
- Modify: `docs/web-deployment.md`

**Interfaces:**
- Documents: 部分反馈来源、6 位密码、首次/更新用户流程和镜像回退顺序。

- [ ] **Step 1: 更新文档**

删除 12 位密码和部署后手动创建首个用户的旧说明；明确初始用户在向导内创建、更新可选新增、密码重置仍使用 `manage_users.py reset-password`。

- [ ] **Step 2: 执行迁移验证**

Run: `.venv/bin/alembic upgrade head`

Run: `.venv/bin/alembic current`

Expected: 当前 revision 为 `20260923_05`，现有反馈项来源为 `manual`。

- [ ] **Step 3: 运行完整自动化验证**

Run: `.venv/bin/python -m unittest discover -s tests -v`

Run: `PATH=/opt/homebrew/opt/node@24/bin:$PATH npm test -- --run`（workdir `web_frontend`）

Run: `PATH=/opt/homebrew/opt/node@24/bin:$PATH npx tsc --noEmit`（workdir `web_frontend`）

Run: `PATH=/opt/homebrew/opt/node@24/bin:$PATH npm run build`（workdir `web_frontend`）

Run: `git diff --check`

Expected: 全部退出码为 0；仅允许缺少 `TEST_DATABASE_URL` 的 PostgreSQL 隔离测试跳过。

- [ ] **Step 4: Docker 重新部署和接口验收**

使用 `./deploy.sh` 更新镜像，确认四个容器健康；登录后提交单个区域反馈并读取响应，确认返回全部检测项和正确来源。不得在日志中出现用户密码。

- [ ] **Step 5: 提交文档**

```bash
git add README.md docs/web-deployment.md
git commit -m "文档：更新反馈和部署使用说明"
```

