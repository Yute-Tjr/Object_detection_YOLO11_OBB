# 用户认证与图片反馈 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为端子检测系统增加大小写敏感的账户登录、12 小时可撤销服务端会话，以及按用户保存的逐图片区域反馈和漏检反馈。

**Architecture:** FastAPI 通过 PostgreSQL 中的 `users` 与 `user_sessions` 实现 Cookie 会话，业务路由统一依赖当前用户；反馈由主表、检测框反馈表和漏检表原子保存。React 在根组件维护认证状态，共用的 `ImageComparison` 打开反馈弹窗，因此任务页和历史页得到完全一致的反馈能力。

**Tech Stack:** Python 3.11、FastAPI、SQLAlchemy 2、Alembic、PostgreSQL、Argon2id、Pydantic 2、React 19、TypeScript、Vite 6、Vitest、Testing Library、Docker Compose。

**Spec:** `docs/superpowers/specs/2026-09-23-authentication-and-image-feedback-design.md`

## Global Constraints

- 用户名清理首尾空白后长度为 1–64，禁止空白/控制字符，登录严格区分大小写。
- 密码至少 12 个字符，只保存 Argon2id 哈希；CLI 不接受明文密码参数。
- 会话令牌只通过名为 `terminal_session` 的 `HttpOnly`、`SameSite=Strict`、`Path=/` Cookie 传输；数据库只保存 SHA-256 摘要。
- 会话固定 12 小时，不滑动续期；`SESSION_COOKIE_SECURE` 默认 `false`，HTTPS 部署必须改为 `true`。
- 健康检查、登录、幂等退出公开；任务、图片、文件和反馈接口均要求有效会话。
- 每个 `(user_id, image_id)` 只有一份可更新反馈；不同用户的反馈互不覆盖。
- `label1_thin` 与 `label1_thick` 都映射为逻辑 `label1`，实际检测框仍保存原始标签快照。
- 每个实际检测框必须提交 `OK/NG`；label1 还必须提交 `B/G/R/W`，其他区域禁止颜色。
- 漏检只能选择当前未检测到的 `label1`–`label6`；漏检项不具有 `OK/NG` 或颜色。
- 任何用户反馈存在时，所属任务必须拒绝永久删除并返回 HTTP 409。
- 不修改目标检测、异常分类、颜色分类和结果图渲染流程。
- 保留 `datasets/obb_thin_thick/.DS_Store` 为未跟踪文件，所有提交必须显式列出文件，不能使用 `git add .`。

## Review Focus

1. **反向代理后的 Origin 比较：** 浏览器的 `Origin` 应与 Nginx 转发的 `Host`/`X-Forwarded-Proto` 匹配，合法 Docker 请求不能被误判为 CSRF；Task 3 添加代理头测试。
2. **并发更新同一反馈：** 唯一约束和事务应保证同一用户不会产生重复主记录；Task 4 添加两次 PUT 与唯一记录断言。
3. **检测集合在弹窗打开后变化：** 提交旧 detection ID 集合必须返回 409/422，而不是保存部分反馈；Task 4 添加集合不一致测试。
4. **有反馈任务的级联删除：** API 必须在隔离查询中先检查反馈，存储文件不能先被永久清除；Task 5 添加数据库与文件均保留测试。
5. **401 期间的前端状态：** 会话过期后只跳转登录，不能把历史列表或反馈草稿误当作业务错误清空；Task 6、Task 7 分别添加认证跳转和草稿保留测试。

---

## File Structure

### Backend files

- `terminal_web/models.py`：增加五张表和 ORM 关系。
- `terminal_web/auth.py`：密码哈希、会话令牌生成/摘要、用户名校验；不访问 HTTP。
- `terminal_web/auth_repository.py`：用户和会话数据库操作。
- `terminal_web/feedback.py`：逻辑区域映射和反馈业务校验。
- `terminal_web/feedback_repository.py`：反馈查询、原子替换和任务反馈存在性查询。
- `terminal_web/api/auth.py`：登录、退出、当前用户接口。
- `terminal_web/api/feedback.py`：读取和更新当前用户的图片反馈。
- `terminal_web/api/dependencies.py`：当前用户依赖和同源请求校验。
- `terminal_web/api/app.py`：注册认证、反馈路由与 CSRF 中间件。
- `terminal_web/api/tasks.py`、`presenters.py`、`schemas.py`：反馈保护和响应字段。
- `scripts/manage_users.py`：无网页入口的管理员用户工具。
- `migrations/versions/_20260923_04_add_auth_and_feedback.py`：五张新表及约束。

### Frontend files

- `web_frontend/src/auth/AuthContext.tsx`：认证启动、登录、退出和 401 失效状态。
- `web_frontend/src/pages/LoginPage.tsx`：简约登录页。
- `web_frontend/src/components/FeedbackDialog.tsx`：逐检测框人工判定与漏检多选。
- `web_frontend/src/components/ImageComparison.tsx`：反馈按钮及弹窗入口。
- `web_frontend/src/api/client.ts`、`types.ts`：认证与反馈接口契约。
- `web_frontend/src/App.tsx`、`Sidebar.tsx`：路由保护、用户名和退出。
- `web_frontend/src/components/TaskTable.tsx`：反馈任务删除禁用状态。
- `web_frontend/src/styles.css`：登录页、反馈按钮、对话框和响应式样式。

### Tests and deployment

- `tests/web/test_auth.py`、`test_auth_api.py`、`test_feedback_api.py`、`test_auth_feedback_migration.py`、`test_manage_users.py`：后端与 CLI 测试。
- `tests/web/auth_helpers.py`：现有 API 测试可复用的登录辅助函数。
- `web_frontend/src/auth/AuthContext.test.tsx`、`pages/LoginPage.test.tsx`、`components/FeedbackDialog.test.tsx`、`components/ImageComparison.test.tsx`：前端测试。
- `requirements.txt`、`pyproject.toml`、`uv.lock`：Argon2id 依赖。
- `.env.example`、`compose.yaml`、`deploy.sh`、`README.md`、`docs/web-deployment.md`：会话配置和首个用户说明。

---

### Task 1: 数据模型、迁移与配置

**Files:**
- Modify: `terminal_web/models.py`
- Modify: `terminal_web/config.py`
- Create: `migrations/versions/_20260923_04_add_auth_and_feedback.py`
- Create: `tests/web/test_auth_feedback_migration.py`
- Modify: `tests/web/test_config.py`
- Modify: `requirements.txt`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Produces: `User`, `UserSession`, `ImageFeedback`, `ImageFeedbackItem`, `ImageFeedbackMiss` ORM classes。
- Produces: `Settings.session_ttl_hours: int` and `Settings.session_cookie_secure: bool`。
- Consumes: existing `Base`, `TimestampMixin`, `InspectionImage`, `Detection`。

- [ ] **Step 1: 写迁移和配置失败测试**

```python
def test_upgrade_creates_auth_and_feedback_tables(self):
    migration = importlib.import_module(
        "migrations.versions._20260923_04_add_auth_and_feedback"
    )
    # 在最小旧结构上执行 upgrade，断言 users、user_sessions、
    # image_feedbacks、image_feedback_items、image_feedback_misses 全部存在。

def test_session_defaults_are_fixed():
    settings = Settings(_env_file=None, **required_settings())
    assert settings.session_ttl_hours == 12
    assert settings.session_cookie_secure is False
```

- [ ] **Step 2: 运行测试确认因迁移/字段不存在而失败**

Run: `uv run python -m unittest tests.web.test_auth_feedback_migration tests.web.test_config -v`

Expected: FAIL，提示迁移模块或会话配置字段不存在。

- [ ] **Step 3: 增加 ORM 与数据库约束**

```python
class User(TimestampMixin, Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

class ImageFeedback(TimestampMixin, Base):
    __tablename__ = "image_feedbacks"
    __table_args__ = (
        UniqueConstraint("user_id", "image_id", name="uq_feedback_user_image"),
    )
```

迁移必须显式设置：用户删除对反馈为 `RESTRICT`，图片删除对反馈为 `CASCADE`，会话随用户级联删除，反馈子项随主反馈级联删除；为所有外键查询列和 `token_hash` 创建索引/唯一约束。

- [ ] **Step 4: 加入 Argon2id 依赖与会话配置**

Run: `uv add 'argon2-cffi==25.1.0'`

随后确认 `requirements.txt` 与 `pyproject.toml` 顺序和内容完全一致，`uv.lock` 已更新；在 `Settings` 增加：

```python
session_ttl_hours: int = 12
session_cookie_secure: bool = False
```

为 TTL 添加 `ge=1, le=168` 约束。

- [ ] **Step 5: 运行模型、迁移和依赖清单测试**

Run: `uv run python -m unittest tests.web.test_auth_feedback_migration tests.web.test_config tests.test_dependency_manifest -v`

Expected: PASS。

- [ ] **Step 6: 提交**

```bash
git add terminal_web/models.py terminal_web/config.py migrations/versions/_20260923_04_add_auth_and_feedback.py tests/web/test_auth_feedback_migration.py tests/web/test_config.py requirements.txt pyproject.toml uv.lock
git commit -m "认证：新增账户会话与反馈数据表"
```

### Task 2: 密码、会话和用户管理命令

**Files:**
- Create: `terminal_web/auth.py`
- Create: `terminal_web/auth_repository.py`
- Create: `scripts/manage_users.py`
- Create: `tests/web/test_auth.py`
- Create: `tests/web/test_manage_users.py`

**Interfaces:**
- Produces: `normalize_username(value: str) -> str`。
- Produces: `hash_password(password: str) -> str`, `verify_password(hash: str, password: str) -> bool`。
- Produces: `create_session_token() -> tuple[str, str]`，返回 `(raw_token, sha256_hex)`；`session_token_digest(raw_token: str) -> str` 用于请求校验。
- Produces: `AuthRepository.get_user_by_username`, `create_user`, `create_session`, `get_user_for_session`, `delete_session`, `delete_sessions_for_user`。
- Consumes: Task 1 的 `User`、`UserSession`。

- [ ] **Step 1: 写认证原语失败测试**

```python
def test_username_is_case_sensitive_and_rejects_whitespace():
    assert normalize_username(" Admin ") == "Admin"
    with pytest.raises(ValueError):
        normalize_username("admin user")

def test_password_hash_is_argon2id_and_verifies():
    encoded = hash_password("correct horse battery staple")
    assert encoded.startswith("$argon2id$")
    assert verify_password(encoded, "correct horse battery staple")
    assert not verify_password(encoded, "wrong password")

def test_session_database_value_is_not_raw_token():
    raw, digest = create_session_token()
    assert raw != digest
    assert len(digest) == 64
```

- [ ] **Step 2: 运行测试确认模块不存在**

Run: `uv run python -m unittest tests.web.test_auth -v`

Expected: FAIL，提示 `terminal_web.auth` 不存在。

- [ ] **Step 3: 实现认证原语和仓库**

```python
PASSWORD_MIN_LENGTH = 12

def create_session_token() -> tuple[str, str]:
    raw = secrets.token_urlsafe(48)
    return raw, hashlib.sha256(raw.encode("utf-8")).hexdigest()

def normalize_username(value: str) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > 64:
        raise ValueError("username must contain 1-64 characters")
    if any(char.isspace() or unicodedata.category(char) == "Cc" for char in normalized):
        raise ValueError("username cannot contain whitespace or control characters")
    return normalized
```

使用 `argon2.PasswordHasher`；为不存在用户准备固定的 dummy Argon2id hash，登录校验始终执行一次 `verify`。

- [ ] **Step 4: 写用户管理 CLI 失败测试**

测试 `add Admin` 和 `add admin` 可共存、重复精确用户名失败、密码少于 12 位失败、两次密码不一致失败、禁用删除会话、启用恢复状态、列表不输出密码哈希。

Run: `uv run python -m unittest tests.web.test_manage_users -v`

Expected: FAIL，提示脚本或命令处理函数不存在。

- [ ] **Step 5: 实现 CLI**

```python
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="管理端子检测系统用户")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("add", "reset-password", "disable", "enable"):
        command = subcommands.add_parser(name)
        command.add_argument("username")
    subcommands.add_parser("list")
    return parser
```

`add`/`reset-password` 使用 `getpass.getpass` 两次读取密码；所有写操作单事务提交，错误回滚并返回非零退出码。

- [ ] **Step 6: 运行测试并提交**

Run: `uv run python -m unittest tests.web.test_auth tests.web.test_manage_users -v`

Expected: PASS。

```bash
git add terminal_web/auth.py terminal_web/auth_repository.py scripts/manage_users.py tests/web/test_auth.py tests/web/test_manage_users.py
git commit -m "认证：增加密码会话与用户管理命令"
```

### Task 3: 登录接口、当前用户依赖和业务路由保护

**Files:**
- Create: `terminal_web/api/auth.py`
- Modify: `terminal_web/api/dependencies.py`
- Modify: `terminal_web/api/app.py`
- Modify: `terminal_web/schemas.py`
- Create: `tests/web/auth_helpers.py`
- Create: `tests/web/test_auth_api.py`
- Modify: `tests/web/test_api.py`
- Modify: `tests/web/test_end_to_end.py`

**Interfaces:**
- Produces: `CurrentUser = Annotated[User, Depends(get_current_user)]`。
- Produces: `POST /api/v1/auth/login`, `POST /api/v1/auth/logout`, `GET /api/v1/auth/me`。
- Produces: `AuthenticationRequired` 前端可识别的稳定 HTTP 401。
- Consumes: `AuthRepository`、`Settings.session_ttl_hours`、`Settings.session_cookie_secure`。

- [ ] **Step 1: 写登录和权限失败测试**

```python
def test_health_is_public_but_tasks_require_login(self):
    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/tasks").status_code == 401

def test_login_is_case_sensitive_and_sets_secure_cookie_fields(self):
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "Admin", "password": VALID_PASSWORD},
        headers={"Origin": "http://testserver"},
    )
    assert response.status_code == 200
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=strict" in response.headers["set-cookie"]
```

同时覆盖不存在用户、错误密码、禁用用户、登出幂等、12 小时过期、禁用后已有 Cookie 立即 401。

- [ ] **Step 2: 写同源保护与代理头失败测试**

```python
def test_unsafe_request_rejects_foreign_origin(self):
    response = client.post(
        "/api/v1/auth/login",
        json=credentials,
        headers={"Origin": "https://evil.example"},
    )
    assert response.status_code == 403

def test_forwarded_origin_matches_public_nginx_host(self):
    response = client.post(
        "/api/v1/auth/login",
        json=credentials,
        headers={
            "Origin": "http://factory.local:8080",
            "Host": "factory.local:8080",
            "X-Forwarded-Proto": "http",
        },
    )
    assert response.status_code == 200
```

- [ ] **Step 3: 运行测试确认接口不存在**

Run: `uv run python -m unittest tests.web.test_auth_api -v`

Expected: FAIL，登录路由 404、任务接口仍公开。

- [ ] **Step 4: 实现认证 API 与依赖**

```python
@router.post("/login", response_model=CurrentUserResponse)
def login(
    payload: LoginRequest,
    response: Response,
    session: Annotated[Session, Depends(get_session)],
    settings=Depends(get_app_settings),
) -> CurrentUserResponse:
    repository = AuthRepository(session)
    username = normalize_username(payload.username)
    user = repository.get_user_by_username(username)
    encoded = user.password_hash if user is not None else DUMMY_PASSWORD_HASH
    password_valid = verify_password(encoded, payload.password)
    if user is None or not user.is_active or not password_valid:
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    raw_token, token_hash = create_session_token()
    expires_at = datetime.now(UTC) + timedelta(hours=settings.session_ttl_hours)
    repository.create_session(user.id, token_hash, expires_at)
    session.commit()
    response.set_cookie(
        SESSION_COOKIE_NAME,
        raw_token,
        httponly=True,
        secure=settings.session_cookie_secure,
        samesite="strict",
        path="/",
        max_age=settings.session_ttl_hours * 3600,
    )
    return CurrentUserResponse(id=user.id, username=user.username)

def get_current_user(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
) -> User:
    raw = request.cookies.get(SESSION_COOKIE_NAME)
    if not raw:
        raise HTTPException(status_code=401, detail="authentication required")
    repository = AuthRepository(session)
    user = repository.get_user_for_session(
        session_token_digest(raw),
        now=datetime.now(UTC),
    )
    if user is None or not user.is_active:
        repository.delete_expired_sessions(datetime.now(UTC))
        session.commit()
        raise HTTPException(status_code=401, detail="authentication required")
    return user
```

在 `tasks.router` 与 `images.router` 级别增加 `dependencies=[Depends(get_current_user)]`；`health` 和 `auth` 路由保持公开。应用中间件只对 `POST/PUT/PATCH/DELETE` 比较 `Origin` 与 `X-Forwarded-Proto + Host`，缺失或不匹配返回 403。

- [ ] **Step 5: 改造现有 API 测试登录辅助**

```python
def create_and_login(client, session_factory, username="tester"):
    password = "Test-password-2026"
    with session_factory() as session:
        session.add(User(username=username, password_hash=hash_password(password)))
        session.commit()
    client.headers["Origin"] = "http://testserver"
    response = client.post(
        "/api/v1/auth/login",
        json={"username": username, "password": password},
    )
    assert response.status_code == 200
    return response.json()
```

`test_api.py` 和 `test_end_to_end.py` 的 `setUp` 调用该辅助；明确验证公开边界的测试使用未登录新客户端。

- [ ] **Step 6: 运行后端 API 测试并提交**

Run: `uv run python -m unittest tests.web.test_auth_api tests.web.test_api tests.web.test_end_to_end -v`

Expected: PASS。

```bash
git add terminal_web/api/auth.py terminal_web/api/dependencies.py terminal_web/api/app.py terminal_web/api/tasks.py terminal_web/api/images.py terminal_web/schemas.py tests/web/auth_helpers.py tests/web/test_auth_api.py tests/web/test_api.py tests/web/test_end_to_end.py
git commit -m "认证：保护业务接口并增加登录会话"
```

### Task 4: 图片反馈领域逻辑与 API

**Files:**
- Create: `terminal_web/feedback.py`
- Create: `terminal_web/feedback_repository.py`
- Create: `terminal_web/api/feedback.py`
- Modify: `terminal_web/api/app.py`
- Modify: `terminal_web/schemas.py`
- Create: `tests/web/test_feedback_api.py`

**Interfaces:**
- Produces: `logical_region(region_label: str) -> str`。
- Produces: `FeedbackRepository.get_for_user_image`, `upsert_feedback`, `task_has_feedback`。
- Produces: `GET /api/v1/images/{image_id}/feedback` and `PUT /api/v1/images/{image_id}/feedback`。
- Consumes: `CurrentUser`、`TaskRepository.get_image`、Task 1 反馈 ORM。

- [ ] **Step 1: 写反馈读取失败测试**

```python
def test_feedback_view_maps_label1_and_lists_only_missing_regions(self):
    payload = logged_in_client.get(f"/api/v1/images/{image_id}/feedback").json()
    assert payload["detections"][0]["logicalRegion"] == "label1"
    assert "label1" not in payload["missedRegionCandidates"]
    assert "label6" in payload["missedRegionCandidates"]
    assert payload["feedback"] is None
```

断言未完成/失败图片不能提交、其他用户反馈不会在当前用户响应中泄露。

- [ ] **Step 2: 写反馈更新与边界失败测试**

覆盖：全部实际 detection 必填、重复 ID、跨图片 ID、陈旧 ID 集合、非法 verdict、label1 缺颜色、label1 非法颜色、非 label1 带颜色、已检测区域标记漏检、无检测且无漏检、同用户二次 PUT 只保留一个主记录、不同用户各自一条记录。

并添加并发/唯一约束断言：

```python
def test_repeated_put_replaces_children_without_duplicate_parent(self):
    first = client.put(url, json=first_payload)
    second = client.put(url, json=updated_payload)
    assert first.status_code == second.status_code == 200
    assert count_feedbacks(user_id, image_id) == 1
    assert stored_verdict(detection_id) == "NG"
```

- [ ] **Step 3: 运行测试确认反馈接口 404**

Run: `uv run python -m unittest tests.web.test_feedback_api -v`

Expected: FAIL，反馈路由或领域函数不存在。

- [ ] **Step 4: 实现领域校验和仓库事务**

```python
LOGICAL_REGIONS = frozenset({"label1", "label2", "label3", "label4", "label5", "label6"})
LABEL1_VARIANTS = frozenset({"label1_thin", "label1_thick"})
ALLOWED_COLORS = frozenset({"B", "G", "R", "W"})

def logical_region(region_label: str) -> str:
    return "label1" if region_label in LABEL1_VARIANTS else region_label
```

`upsert_feedback` 在一个事务内锁定 `(user_id, image_id)` 主记录，验证数据库当前 detections 的精确集合，删除旧子项、插入本次子项并 flush；捕获唯一约束竞争后重试一次或返回稳定 409，不产生重复反馈。

- [ ] **Step 5: 实现响应模型和路由**

```python
class FeedbackItemInput(ApiModel):
    detection_id: uuid.UUID
    verdict: Literal["OK", "NG"]
    color: Literal["B", "G", "R", "W"] | None = None

class FeedbackUpdateRequest(ApiModel):
    items: list[FeedbackItemInput]
    missed_regions: list[Literal["label1", "label2", "label3", "label4", "label5", "label6"]]
```

GET 返回 detection 的只读模型异常/颜色结果、候选漏检和当前用户已有反馈；PUT 成功返回同一完整视图，便于前端直接回填。

- [ ] **Step 6: 运行测试并提交**

Run: `uv run python -m unittest tests.web.test_feedback_api -v`

Expected: PASS。

```bash
git add terminal_web/feedback.py terminal_web/feedback_repository.py terminal_web/api/feedback.py terminal_web/api/app.py terminal_web/schemas.py tests/web/test_feedback_api.py
git commit -m "反馈：增加逐图片区域与漏检反馈接口"
```

### Task 5: 任务反馈状态和删除保护

**Files:**
- Modify: `terminal_web/repositories.py`
- Modify: `terminal_web/api/presenters.py`
- Modify: `terminal_web/api/tasks.py`
- Modify: `terminal_web/schemas.py`
- Modify: `tests/web/test_api.py`
- Modify: `tests/web/test_repositories.py`

**Interfaces:**
- Produces: `TaskSummary.has_feedback: bool` / JSON `hasFeedback`。
- Produces: `TaskRepository.task_has_feedback(task_id: UUID) -> bool`。
- Consumes: Task 4 的 `ImageFeedback`。

- [ ] **Step 1: 写反馈摘要和删除保护失败测试**

```python
def test_task_summary_marks_feedback_and_delete_preserves_artifacts(self):
    create_feedback(task_id, image_id, user_id)
    page = client.get("/api/v1/tasks?status=all").json()
    assert page["items"][0]["hasFeedback"] is True
    response = client.delete(f"/api/v1/tasks/{task_id}")
    assert response.status_code == 409
    assert task_exists(task_id)
    assert artifact_directory(task_id).exists()
```

再断言无反馈已完成任务仍可删除，进行中任务仍使用原有 409 原因。

- [ ] **Step 2: 运行测试确认 `hasFeedback` 缺失**

Run: `uv run python -m unittest tests.web.test_api tests.web.test_repositories -v`

Expected: FAIL，响应缺少字段且反馈任务仍可删除。

- [ ] **Step 3: 使用 EXISTS 查询实现反馈状态**

```python
def task_has_feedback(self, task_id: uuid.UUID) -> bool:
    statement = select(
        select(ImageFeedback.id)
        .join(InspectionImage, ImageFeedback.image_id == InspectionImage.id)
        .where(InspectionImage.task_id == task_id)
        .exists()
    )
    return bool(self.session.scalar(statement))
```

列表查询批量加载有反馈任务 ID，禁止每行执行一次查询；`task_summary` 接收显式 `has_feedback` 值。

- [ ] **Step 4: 删除前检查反馈再隔离文件**

在 `storage.quarantine_task` 之前调用反馈存在性查询；存在时返回 `409` 和稳定文案“任务包含人工反馈，不能删除”。数据库异常仍走 503，不移动文件。

- [ ] **Step 5: 运行测试并提交**

Run: `uv run python -m unittest tests.web.test_api tests.web.test_repositories -v`

Expected: PASS。

```bash
git add terminal_web/repositories.py terminal_web/api/presenters.py terminal_web/api/tasks.py terminal_web/schemas.py tests/web/test_api.py tests/web/test_repositories.py
git commit -m "反馈：保护含人工反馈的历史任务"
```

### Task 6: 前端认证状态、登录页和会话失效处理

**Files:**
- Modify: `web_frontend/src/api/types.ts`
- Modify: `web_frontend/src/api/client.ts`
- Create: `web_frontend/src/auth/AuthContext.tsx`
- Create: `web_frontend/src/auth/AuthContext.test.tsx`
- Create: `web_frontend/src/pages/LoginPage.tsx`
- Create: `web_frontend/src/pages/LoginPage.test.tsx`
- Modify: `web_frontend/src/App.tsx`
- Modify: `web_frontend/src/components/Sidebar.tsx`
- Modify: `web_frontend/src/api/client.test.ts`
- Modify: `web_frontend/src/styles.css`

**Interfaces:**
- Produces: `AuthUser { id: string; username: string }`。
- Produces: `AuthProvider` and `useAuth(): { status, user, login, logout }`。
- Produces: global API unauthorized subscription without erasing page-local business data。
- Consumes: `/auth/login`, `/auth/logout`, `/auth/me`。

- [ ] **Step 1: 写 API client Cookie 与 401 失败测试**

```typescript
it("sends same-origin credentials and notifies once on 401", async () => {
  const listener = vi.fn();
  const unsubscribe = subscribeUnauthorized(listener);
  await expect(apiClient.getTask("task-1")).rejects.toMatchObject({ status: 401 });
  expect(fetchMock).toHaveBeenCalledWith(
    "/api/v1/tasks/task-1",
    expect.objectContaining({ credentials: "same-origin" }),
  );
  expect(listener).toHaveBeenCalledTimes(1);
  unsubscribe();
});
```

- [ ] **Step 2: 写登录与路由失败测试**

覆盖：启动 `/auth/me` 加载态、未登录显示登录页、成功后恢复原 `/history/<taskId>?image=<imageId>` 路由、错误密码就地提示、用户名大小写原样提交、退出回到登录页、业务请求 401 进入登录但不触发历史页“加载失败”清空流程。

Run: `cd web_frontend && npm test -- --run src/api/client.test.ts src/auth/AuthContext.test.tsx src/pages/LoginPage.test.tsx`

Expected: FAIL，认证模块和登录页不存在。

- [ ] **Step 3: 实现类型、client 和认证上下文**

```typescript
export class ApiError extends Error {
  constructor(message: string, readonly status: number) { super(message); }
}

const response = await fetch(`${API_ROOT}${path}`, {
  credentials: "same-origin",
  ...init,
});
if (response.status === 401) unauthorizedListeners.forEach((listener) => listener());
```

`AuthProvider` 首次只调用一次 `getCurrentUser`；401 将认证状态改为匿名，但不主动重置页面组件内部的任务/反馈数据。

- [ ] **Step 4: 实现登录页、App 守卫和侧边栏**

```tsx
if (status === "loading") return <div className="auth-loading">正在验证登录状态…</div>;
if (status === "anonymous") return <LoginPage onLogin={login} />;
return <AuthenticatedApplication user={user} onLogout={logout} />;
```

登录表单只有用户名、密码、登录按钮和错误；侧边栏底部显示用户原始大小写用户名与退出按钮。不要加入注册、找回密码或模型名称。

- [ ] **Step 5: 运行测试和生产构建并提交**

Run: `cd web_frontend && npm test -- --run src/api/client.test.ts src/auth/AuthContext.test.tsx src/pages/LoginPage.test.tsx`

Run: `cd web_frontend && npm run build`

Expected: PASS。

```bash
git add web_frontend/src/api/types.ts web_frontend/src/api/client.ts web_frontend/src/auth/AuthContext.tsx web_frontend/src/auth/AuthContext.test.tsx web_frontend/src/pages/LoginPage.tsx web_frontend/src/pages/LoginPage.test.tsx web_frontend/src/App.tsx web_frontend/src/components/Sidebar.tsx web_frontend/src/api/client.test.ts web_frontend/src/styles.css
git commit -m "前端：增加登录页与会话状态管理"
```

### Task 7: 反馈弹窗与任务页/历史页入口

**Files:**
- Modify: `web_frontend/src/api/types.ts`
- Modify: `web_frontend/src/api/client.ts`
- Create: `web_frontend/src/components/FeedbackDialog.tsx`
- Create: `web_frontend/src/components/FeedbackDialog.test.tsx`
- Modify: `web_frontend/src/components/ImageComparison.tsx`
- Create: `web_frontend/src/components/ImageComparison.test.tsx`
- Modify: `web_frontend/src/pages/TasksPage.tsx`
- Modify: `web_frontend/src/pages/HistoryPage.tsx`
- Modify: `web_frontend/src/styles.css`

**Interfaces:**
- Produces: `ImageFeedbackView`, `FeedbackUpdateRequest` TypeScript types。
- Produces: `apiClient.getImageFeedback`, `apiClient.updateImageFeedback`。
- Produces: `<FeedbackDialog imageId filename onClose onSaved />`。
- Consumes: `ImageComparison` current selected `ImageSummary`。

- [ ] **Step 1: 写 API 合同和反馈按钮失败测试**

```typescript
it("loads and updates the selected image feedback", async () => {
  await apiClient.getImageFeedback("image/1");
  await apiClient.updateImageFeedback("image/1", {
    items: [{ detectionId: "d1", verdict: "NG", color: "R" }],
    missedRegions: ["label6"],
  });
  expect(fetchMock.mock.calls[1][1]).toMatchObject({ method: "PUT" });
});

it("shows feedback only for completed images", () => {
  render(
    <ImageComparison
      images={[completedImage]}
      selectedIndex={0}
      onSelectedIndexChange={vi.fn()}
    />,
  );
  expect(screen.getByRole("button", { name: "结果反馈" })).toBeEnabled();
});
```

- [ ] **Step 2: 写弹窗校验和错误保留测试**

覆盖：首次不预选模型值、每个 detection 独立选择、label1 显示统一逻辑名和实际变体、label1 颜色必填、其他区域没有颜色、只展示未检测的漏检候选、零检测必须选择漏检、已有反馈回填、加载失败可重试、提交失败保留全部草稿、防止重复提交、Escape/遮罩关闭前不丢失提交中状态。

Run: `cd web_frontend && npm test -- --run src/components/FeedbackDialog.test.tsx src/components/ImageComparison.test.tsx`

Expected: FAIL，组件和 client 方法不存在。

- [ ] **Step 3: 实现类型和 client**

```typescript
export type FeedbackVerdict = "OK" | "NG";
export type FeedbackColor = "B" | "G" | "R" | "W";
export type LogicalRegion = "label1" | "label2" | "label3" | "label4" | "label5" | "label6";

getImageFeedback(imageId: string): Promise<ImageFeedbackView>;
updateImageFeedback(imageId: string, payload: FeedbackUpdateRequest): Promise<ImageFeedbackView>;
```

- [ ] **Step 4: 实现弹窗状态机**

`loading → editing → submitting → success/error` 明确分支；草稿按 `detectionId` 保存：

```typescript
type FeedbackDraft = {
  items: Record<string, { verdict?: FeedbackVerdict; color?: FeedbackColor }>;
  missedRegions: Set<LogicalRegion>;
};
```

提交前前端校验用于即时提示，后端仍是最终约束。成功后显示“反馈已保存”并关闭；失败只更新错误字段，不重建 draft。

- [ ] **Step 5: 在共享结果头部接入按钮**

按钮放在图片序号左侧；`TasksPage` 与 `HistoryPage` 继续通过同一个 `ImageComparison` 获得功能。仅 `status === "succeeded" && stage === "complete"` 可用；失败/处理中显示禁用原因。

- [ ] **Step 6: 运行前端测试和构建并提交**

Run: `cd web_frontend && npm test -- --run src/components/FeedbackDialog.test.tsx src/components/ImageComparison.test.tsx src/pages/TasksPage.test.tsx src/pages/HistoryPage.test.tsx`

Run: `cd web_frontend && npm run build`

Expected: PASS。

```bash
git add web_frontend/src/api/types.ts web_frontend/src/api/client.ts web_frontend/src/components/FeedbackDialog.tsx web_frontend/src/components/FeedbackDialog.test.tsx web_frontend/src/components/ImageComparison.tsx web_frontend/src/components/ImageComparison.test.tsx web_frontend/src/pages/TasksPage.tsx web_frontend/src/pages/HistoryPage.tsx web_frontend/src/styles.css
git commit -m "前端：增加逐图片检测结果反馈"
```

### Task 8: 前端反馈任务删除保护

**Files:**
- Modify: `web_frontend/src/api/types.ts`
- Modify: `web_frontend/src/components/TaskTable.tsx`
- Modify: `web_frontend/src/components/TaskTable.test.tsx`
- Modify: `web_frontend/src/pages/HistoryPage.tsx`
- Modify: `web_frontend/src/pages/HistoryPage.test.tsx`

**Interfaces:**
- Consumes: backend JSON `TaskSummary.hasFeedback`。
- Produces: disabled delete UI with exact reason；后端 409 仍可显示为页面错误。

- [ ] **Step 1: 写删除禁用失败测试**

```typescript
it("disables deletion for tasks that contain feedback", async () => {
  render(
    <TaskTable
      tasks={[{ ...task, hasFeedback: true }]}
      onSelectTask={vi.fn()}
      onDeleteTask={vi.fn()}
      initialFilter="all"
    />,
  );
  const button = screen.getByRole("button", { name: /删除/ });
  expect(button).toBeDisabled();
  expect(button).toHaveAttribute(
    "title",
    "包含人工反馈，已作为模型优化数据保留",
  );
});
```

同时验证无反馈已完成任务仍可打开自定义删除卡片，反馈任务不会调用 `onDeleteTask`。

- [ ] **Step 2: 运行测试确认按钮仍可用**

Run: `cd web_frontend && npm test -- --run src/components/TaskTable.test.tsx src/pages/HistoryPage.test.tsx`

Expected: FAIL。

- [ ] **Step 3: 实现 `hasFeedback` 类型和按钮原因优先级**

优先级：正在删除 > 进行中不可删 > 有反馈不可删 > 可删。HistoryPage 的 `requestDeleteTask` 再检查一次 `hasFeedback`，避免通过非按钮调用打开删除弹窗。

- [ ] **Step 4: 运行测试并提交**

Run: `cd web_frontend && npm test -- --run src/components/TaskTable.test.tsx src/pages/HistoryPage.test.tsx`

Expected: PASS。

```bash
git add web_frontend/src/api/types.ts web_frontend/src/components/TaskTable.tsx web_frontend/src/components/TaskTable.test.tsx web_frontend/src/pages/HistoryPage.tsx web_frontend/src/pages/HistoryPage.test.tsx
git commit -m "前端：禁止删除含人工反馈的任务"
```

### Task 9: 部署配置、说明文档与端到端验收

**Files:**
- Modify: `.env.example`
- Modify: `compose.yaml`
- Modify: `deploy.sh`
- Modify: `tests/test_docker_deploy.py`
- Modify: `README.md`
- Modify: `docs/web-deployment.md`
- Modify: `tests/web/test_postgres_integration.py`

**Interfaces:**
- Produces: Docker/native 均传递 `SESSION_TTL_HOURS=12`、`SESSION_COOKIE_SECURE=false`。
- Produces: 部署结束时可复制的首用户创建命令，但绝不自动创建默认账户。
- Consumes: `scripts/manage_users.py`。

- [ ] **Step 1: 写部署配置失败测试**

```python
def test_compose_passes_session_settings(self):
    environment = compose["services"]["api"]["environment"]
    self.assertEqual(environment["SESSION_TTL_HOURS"], "12")
    self.assertEqual(environment["SESSION_COOKIE_SECURE"], "false")

def test_deploy_prints_first_user_command_without_password(self):
    self.assertIn(
        "docker compose exec api python scripts/manage_users.py add USERNAME",
        result.stdout,
    )
    self.assertNotIn("DEFAULT_PASSWORD", written_env)
```

- [ ] **Step 2: 运行测试确认配置缺失**

Run: `uv run python -m unittest tests.test_docker_deploy -v`

Expected: FAIL，会话变量和首用户提示缺失。

- [ ] **Step 3: 更新环境、Compose 和部署输出**

`.env.example` 增加：

```dotenv
SESSION_TTL_HOURS=12
SESSION_COOKIE_SECURE=false
```

`compose.yaml` 将两项传入 API 环境；Worker 可复用但不读取。`deploy.sh` 保留已有交互，部署成功后输出：

```text
首次使用请创建登录用户：
docker compose exec api python scripts/manage_users.py add USERNAME
```

不把用户名或密码写进 `.env`，不自动创建弱默认用户。

- [ ] **Step 4: 更新 README 和部署文档**

说明原生命令 `uv run python scripts/manage_users.py add USERNAME`、Conda 等价命令、Docker 命令、12 小时会话、HTTP/HTTPS Cookie 配置、禁用用户会立即退出、反馈任务不能删除和数据库备份/降级风险。

- [ ] **Step 5: 添加 PostgreSQL 约束集成测试**

在 `TEST_DATABASE_URL` 可用时验证：用户名大小写两个值可共存、同一精确用户名唯一、同用户同图片反馈唯一、删除图片级联反馈、存在反馈时直接删除用户被 RESTRICT。测试独立 schema，结束后清理。

- [ ] **Step 6: 执行后端全量验证**

Run: `uv run python -m unittest discover -s tests -v`

Expected: PASS；仅未提供 `TEST_DATABASE_URL` 的 PostgreSQL 专用测试显示 SKIP。

- [ ] **Step 7: 执行前端全量验证**

Run: `cd web_frontend && npm test -- --run`

Run: `cd web_frontend && npm run build`

Run: `cd web_frontend && npm run test:sites`

Expected: 全部 PASS，Vite 生产构建生成 `dist/client/index.html`、`dist/server/index.js` 和 `dist/.openai/hosting.json`。

- [ ] **Step 8: 执行迁移与真实启动冒烟验证**

Run: `uv run alembic upgrade head`

Run: `uv run python scripts/manage_users.py add smoke-admin`

在本机 PostgreSQL 与模型权重可用时运行 `uv run python scripts/start_terminal_web.py`，验证：登录、上传一张图、处理完成后提交反馈、历史页回填、含反馈任务删除被拒绝、退出后图片 URL 返回 401。随后重启服务并确认会话与反馈仍存在。

- [ ] **Step 9: 浏览器视觉验收**

使用本地前端预览检查 1280px、1440px 和窄屏：登录卡片不过宽；反馈按钮不挤压图片翻页；反馈弹窗长列表可滚动；label1 颜色控件与 OK/NG 不重叠；侧边栏用户名溢出时省略且可查看完整 title；所有按钮具备现有 hover/focus 风格。

- [ ] **Step 10: 最终提交**

```bash
git add .env.example compose.yaml deploy.sh tests/test_docker_deploy.py README.md docs/web-deployment.md tests/web/test_postgres_integration.py
git commit -m "部署：完善认证与反馈功能说明"
```

最终检查：

```bash
git status --short
git log --oneline -10
```

Expected: 只允许预先存在的 `datasets/obb_thin_thick/.DS_Store` 保持未跟踪，其他实现文件均已提交。
