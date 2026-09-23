# 已提交反馈撤销 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 允许登录用户永久撤销自己对单张图片已提交的反馈，并在任务不再含任何用户反馈时立即恢复任务删除能力。

**Architecture:** FastAPI 增加用户隔离且幂等的反馈删除接口，由 Repository 删除当前用户与图片的唯一反馈主记录，数据库级联清理反馈项和漏检项。React 反馈弹窗仅在已有反馈时提供项目内二次确认卡片；删除成功后恢复空白草稿，并通知历史页重新读取任务详情与列表，由后端重新计算 `hasFeedback`。

**Tech Stack:** Python 3.11、FastAPI、SQLAlchemy 2、PostgreSQL、React 19、TypeScript、Vitest。

**Spec:** `docs/superpowers/specs/2026-09-23-partial-feedback-and-deployment-flow-design.md`

## Global Constraints

- 只能撤销当前登录用户对当前图片的反馈，不得删除或暴露其他用户的反馈。
- 删除反馈采用幂等语义；记录不存在时返回 `200` 和 `feedback=null`。
- 删除反馈不能修改检测任务、图片、检测框、分类结果或结果图。
- 任务存在任意图片、任意用户反馈时仍不可删除；全部反馈撤销后 `hasFeedback=false`。
- 撤销必须使用项目内确认卡片，不使用 `window.confirm`。
- 不处理既有未跟踪文件 `datasets/obb_thin_thick/.DS_Store`。

## Review Focus

- 两个用户反馈同一图片时，一个用户撤销不能删除另一个用户记录；Task 1 覆盖。
- 重复撤销或反馈已在另一窗口删除时必须幂等成功；Task 1 覆盖。
- 删除主反馈后其 items 和 misses 必须级联清理，检测数据必须保留；Task 1 覆盖。
- 撤销接口失败时确认卡片关闭但原反馈和草稿必须保留，并显示错误；Task 2 覆盖。
- 撤销任务中最后一条反馈后必须重新读取后端状态，而不是把 `hasFeedback` 盲目设为 `false`；Task 3 覆盖。

---

### Task 1: 用户隔离的反馈撤销接口

**Files:**
- Modify: `terminal_web/feedback_repository.py`
- Modify: `terminal_web/api/feedback.py`
- Modify: `tests/web/test_feedback_api.py`

**Interfaces:**
- Produces: `FeedbackRepository.delete_for_user_image(user_id: UUID, image_id: UUID) -> bool`。
- Produces: `DELETE /api/v1/images/{image_id}/feedback -> ImageFeedbackView`。

- [ ] **Step 1: 编写失败的 API 测试**

新增测试：保存带人工项和漏检项的反馈后执行 DELETE，断言响应为 `200`、`feedback is None`，数据库中当前用户的 `ImageFeedback`、`ImageFeedbackItem` 和 `ImageFeedbackMiss` 均为零，而 `InspectionImage` 与 `Detection` 仍存在。

再新增两个测试：连续 DELETE 两次都返回 `200`；第二个用户撤销自己的反馈后，第一个用户反馈仍可读取且任务仍报告 `hasFeedback=true`。

- [ ] **Step 2: 运行测试并确认失败**

Run: `.venv/bin/python -m unittest tests.web.test_feedback_api -v`

Expected: FAIL，DELETE 路由返回 `405 Method Not Allowed`。

- [ ] **Step 3: 实现 Repository 删除方法**

```python
def delete_for_user_image(self, user_id: uuid.UUID, image_id: uuid.UUID) -> bool:
    result = self.session.execute(
        delete(ImageFeedback).where(
            ImageFeedback.user_id == user_id,
            ImageFeedback.image_id == image_id,
        )
    )
    self.session.flush()
    return bool(result.rowcount)
```

- [ ] **Step 4: 实现幂等 DELETE 路由**

加载图片以保持不存在图片返回 `404`；按当前用户和图片删除反馈、提交事务，再返回 `_feedback_view(image, None)`。不要求图片仍处于可反馈状态，因为撤销已有记录不修改检测结果。

- [ ] **Step 5: 运行后端反馈与任务测试**

Run: `.venv/bin/python -m unittest tests.web.test_feedback_api tests.web.test_api tests.web.test_repositories -v`

Expected: PASS。

---

### Task 2: 反馈弹窗撤销确认与状态恢复

**Files:**
- Modify: `web_frontend/src/api/client.ts`
- Modify: `web_frontend/src/api/client.test.ts`
- Modify: `web_frontend/src/components/FeedbackDialog.tsx`
- Modify: `web_frontend/src/components/FeedbackDialog.test.tsx`
- Modify: `web_frontend/src/styles.css`

**Interfaces:**
- Produces: `ApiClient.deleteImageFeedback(imageId, signal?) -> Promise<ImageFeedbackView>`。
- Produces: `FeedbackDialog.onDeleted(feedback: ImageFeedbackView) -> void`。

- [ ] **Step 1: 编写失败的客户端和组件测试**

客户端测试断言请求为 `DELETE /api/v1/images/{encoded_id}/feedback`。组件测试断言：无历史反馈时不显示撤销按钮；有历史反馈时显示“撤销已提交反馈”；首次点击只打开确认卡片；取消不调用接口；确认才调用删除接口。

增加失败测试：接口拒绝时保留历史选择和撤销入口并显示错误；成功时把视图替换为 `feedback=null`、清空草稿、关闭确认卡片并调用 `onDeleted`。

- [ ] **Step 2: 运行测试并确认失败**

Run: `PATH=/opt/homebrew/opt/node@24/bin:$PATH npm test -- --run src/api/client.test.ts src/components/FeedbackDialog.test.tsx`

Workdir: `web_frontend`

Expected: FAIL，客户端没有删除方法且组件没有撤销入口。

- [ ] **Step 3: 实现客户端和弹窗状态**

将 `deleteImageFeedback` 加入 `FeedbackClient`。弹窗增加 `confirmingDelete` 与 `deleting` 状态；仅当 `view.feedback` 非空时显示撤销按钮。成功响应后执行：

```typescript
setView(updated);
setDraft(draftFromView(updated));
setConfirmingDelete(false);
onDeleted(updated);
```

失败时不改变 `view` 或 `draft`，只关闭确认卡片并显示 API 错误。

- [ ] **Step 4: 实现项目内确认卡片**

确认卡片文案为“撤销已提交反馈？”和“将永久删除你对这张图片提交的反馈，检测结果不会被删除。”，提供“取消”和“确认撤销”按钮；提交或撤销请求进行中时禁用会造成并发状态变化的操作。

- [ ] **Step 5: 运行组件测试、类型检查和构建**

Run: `PATH=/opt/homebrew/opt/node@24/bin:$PATH npm test -- --run src/api/client.test.ts src/components/FeedbackDialog.test.tsx`

Run: `PATH=/opt/homebrew/opt/node@24/bin:$PATH npx tsc --noEmit`

Run: `PATH=/opt/homebrew/opt/node@24/bin:$PATH npm run build`

Workdir: `web_frontend`

Expected: PASS。

---

### Task 3: 历史任务状态权威刷新

**Files:**
- Modify: `web_frontend/src/components/ImageComparison.tsx`
- Modify: `web_frontend/src/components/ImageComparison.test.tsx`
- Modify: `web_frontend/src/pages/HistoryPage.tsx`
- Modify: `web_frontend/src/pages/HistoryPage.test.tsx`

**Interfaces:**
- Consumes: `FeedbackDialog.onDeleted`。
- Produces: `ImageComparison.onFeedbackDeleted?: (imageId: string) => void`。

- [ ] **Step 1: 编写失败的状态刷新测试**

模拟任务初始 `hasFeedback=true`，在反馈弹窗确认撤销最后一条反馈后，让 `getTask` 和 `listTasks` 返回 `hasFeedback=false`；断言历史表删除按钮恢复可用。另测后端仍返回 `hasFeedback=true` 时删除按钮继续禁用，覆盖其他图片或其他用户仍有反馈的情况。

- [ ] **Step 2: 运行测试并确认失败**

Run: `PATH=/opt/homebrew/opt/node@24/bin:$PATH npm test -- --run src/components/ImageComparison.test.tsx src/pages/HistoryPage.test.tsx`

Workdir: `web_frontend`

Expected: FAIL，撤销事件尚未传播且历史页不会刷新任务状态。

- [ ] **Step 3: 传播撤销事件并重新获取状态**

`ImageComparison` 将弹窗 `onDeleted` 转换为当前 `image.id` 回调。`HistoryPage` 收到事件后并行重新获取任务列表和当前任务详情，使用响应中的 `hasFeedback` 更新表格及所选任务；不直接写死 `false`。检测任务页无需额外处理，因为它不显示历史任务删除按钮。

- [ ] **Step 4: 运行完整验证**

Run: `.venv/bin/python -m unittest tests.web.test_feedback_api tests.web.test_api tests.web.test_repositories -v`

Run: `PATH=/opt/homebrew/opt/node@24/bin:$PATH npm test -- --run`

Run: `PATH=/opt/homebrew/opt/node@24/bin:$PATH npx tsc --noEmit`

Run: `PATH=/opt/homebrew/opt/node@24/bin:$PATH npm run build`

Run: `git diff --check`

Expected: 全部退出码为 `0`。
