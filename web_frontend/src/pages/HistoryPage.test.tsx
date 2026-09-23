import "@testing-library/jest-dom/vitest";

import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { TaskDetail, TaskPage } from "../api/types";
import { HistoryPage } from "./HistoryPage";


const detail: TaskDetail = {
  id: "task-id",
  displayId: "T20260920-0100",
  status: "partial_failed",
  currentStage: "complete",
  totalImages: 2,
  completedImages: 2,
  succeededImages: 1,
  failedImages: 1,
  hasFeedback: false,
  createdAt: "2026-09-20T08:00:00Z",
  images: [
    {
      id: "ok-image",
      sequenceNo: 0,
      originalFilename: "terminal-ok.png",
      status: "succeeded",
      stage: "complete",
      overallResult: "OK",
      width: 1440,
      height: 3072,
      sizeBytes: 2300000,
      originalUrl: "/api/v1/images/ok-image/original",
      resultUrl: "/api/v1/images/ok-image/result",
    },
    {
      id: "failed-image",
      sequenceNo: 1,
      originalFilename: "terminal-failed.png",
      status: "failed",
      stage: "complete",
      overallResult: "UNKNOWN",
      width: 1440,
      height: 3072,
      sizeBytes: 2200000,
      originalUrl: "/api/v1/images/failed-image/original",
      errorMessage: "推理失败",
    },
  ],
};

const page: TaskPage = { items: [detail], total: 1, offset: 0, limit: 100 };

function client(overrides = {}) {
  return {
    getCurrentUser: vi.fn(),
    login: vi.fn(),
    logout: vi.fn(),
    getHealth: vi.fn(),
    createTask: vi.fn(),
    listTasks: vi.fn().mockResolvedValue(page),
    getTask: vi.fn().mockResolvedValue(detail),
    getTaskImages: vi.fn(),
    getImage: vi.fn(),
    retryImage: vi.fn().mockResolvedValue({ ...detail.images[1], status: "queued" }),
    deleteTask: vi.fn().mockResolvedValue(undefined),
    getImageFeedback: vi.fn(),
    updateImageFeedback: vi.fn(),
    ...overrides,
  };
}

afterEach(() => {
  cleanup();
  window.history.replaceState({}, "", "/");
});

describe("HistoryPage", () => {
  it("shows history actions without metadata or model identity", async () => {
    render(<HistoryPage client={client()} />);

    expect(await screen.findByText("T20260920-0100")).toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "模型" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /展开任务/ })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "预览 T20260920-0100" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "删除 T20260920-0100" })).toBeInTheDocument();
  });

  it("keeps server results when searching by original image filename", async () => {
    const api = client();
    const user = userEvent.setup();
    render(<HistoryPage client={api} />);
    await screen.findByText("T20260920-0100");

    await user.type(screen.getByLabelText("搜索历史任务"), "terminal-ok.png");
    await user.click(screen.getByRole("button", { name: "搜索" }));

    await waitFor(() => expect(api.listTasks).toHaveBeenLastCalledWith({
      status: "all",
      query: "terminal-ok.png",
      limit: 100,
    }));
    expect(screen.getByText("T20260920-0100")).toBeInTheDocument();
  });

  it("loads a historical task and opens its original/result comparison", async () => {
    const api = client();
    render(<HistoryPage client={api} />);

    await userEvent.click(
      await screen.findByRole("button", { name: "预览 T20260920-0100" }),
    );
    expect(api.getTask).toHaveBeenCalledWith("task-id", expect.anything());
    expect(await screen.findByAltText("检测前原图")).toHaveAttribute(
      "src",
      "/api/v1/images/ok-image/original",
    );
    expect(screen.getByAltText("检测后结果")).toBeInTheDocument();
  });

  it("offers retry only for a failed image and refreshes the task", async () => {
    const api = client();
    const user = userEvent.setup();
    render(<HistoryPage client={api} initialTaskId="task-id" initialImageId="failed-image" />);

    expect(await screen.findByText("terminal-failed.png")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "重试失败图片" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "重试失败图片" }));
    await waitFor(() => expect(api.retryImage).toHaveBeenCalledWith("failed-image"));
    await waitFor(() => expect(api.getTask).toHaveBeenCalledTimes(2));

    await user.click(screen.getByRole("button", { name: "上一张图片" }));
    expect(screen.queryByRole("button", { name: "重试失败图片" })).not.toBeInTheDocument();
  });

  it("keeps current tasks visible when refresh fails and allows retry", async () => {
    const listTasks = vi.fn()
      .mockResolvedValueOnce(page)
      .mockRejectedValueOnce(new Error("网络暂时不可用"))
      .mockResolvedValueOnce(page);
    const api = client({ listTasks });
    const user = userEvent.setup();
    render(<HistoryPage client={api} />);

    expect(await screen.findByText("T20260920-0100")).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "刷新历史记录" }));
    expect(await screen.findByText("网络暂时不可用")).toBeInTheDocument();
    expect(screen.getByText("T20260920-0100")).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "重试加载" }));
    await waitFor(() => expect(listTasks).toHaveBeenCalledTimes(3));
  });

  it("keeps the selected comparison when reopening fails", async () => {
    const api = client({
      getTask: vi.fn()
        .mockResolvedValueOnce(detail)
        .mockRejectedValueOnce(new Error("服务暂不可用")),
    });
    const user = userEvent.setup();
    render(<HistoryPage client={api} />);

    await user.click(
      await screen.findByRole("button", { name: "预览 T20260920-0100" }),
    );
    expect(await screen.findByAltText("检测前原图")).toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: "预览 T20260920-0100" }),
    );

    expect(await screen.findByText("服务暂不可用")).toBeInTheDocument();
    expect(screen.getByAltText("检测前原图")).toBeInTheDocument();
  });

  it("deletes a confirmed completed task and closes its preview", async () => {
    const api = client();
    const user = userEvent.setup();
    render(<HistoryPage client={api} initialTaskId="task-id" />);

    expect(await screen.findByAltText("检测前原图")).toBeInTheDocument();
    await user.click(
      screen.getByRole("button", { name: "删除 T20260920-0100" }),
    );

    const dialog = screen.getByRole("dialog", { name: "确认删除任务" });
    expect(within(dialog).getByText("T20260920-0100")).toBeInTheDocument();
    expect(within(dialog).getByText("2 张")).toBeInTheDocument();
    expect(within(dialog).queryByText("任务名称")).not.toBeInTheDocument();
    expect(within(dialog).getByText(/2 张原图、结果图和检测记录/)).toBeInTheDocument();
    expect(api.deleteTask).not.toHaveBeenCalled();
    await user.click(within(dialog).getByRole("button", { name: "确认删除" }));

    await waitFor(() => expect(api.deleteTask).toHaveBeenCalledWith("task-id"));
    await waitFor(() => {
      expect(screen.queryByText("T20260920-0100")).not.toBeInTheDocument();
    });
    expect(screen.queryByAltText("检测前原图")).not.toBeInTheDocument();
  });

  it("closes the delete card without deleting when cancelled or Escape is pressed", async () => {
    const api = client();
    const user = userEvent.setup();
    render(<HistoryPage client={api} />);

    const deleteButton = await screen.findByRole("button", {
      name: "删除 T20260920-0100",
    });
    await user.click(deleteButton);
    await user.click(screen.getByRole("button", { name: "取消删除" }));
    expect(screen.queryByRole("dialog", { name: "确认删除任务" })).not.toBeInTheDocument();

    await user.click(deleteButton);
    expect(screen.getByRole("dialog", { name: "确认删除任务" })).toBeInTheDocument();
    await user.keyboard("{Escape}");

    expect(screen.queryByRole("dialog", { name: "确认删除任务" })).not.toBeInTheDocument();
    expect(api.deleteTask).not.toHaveBeenCalled();
  });

  it("does not offer deletion for a task that contains feedback", async () => {
    const feedbackDetail = { ...detail, hasFeedback: true };
    const api = client({
      listTasks: vi.fn().mockResolvedValue({ ...page, items: [feedbackDetail] }),
      getTask: vi.fn().mockResolvedValue(feedbackDetail),
    });
    const user = userEvent.setup();
    render(<HistoryPage client={api} />);

    const button = await screen.findByRole("button", { name: "删除 T20260920-0100" });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute("title", "包含人工反馈，已作为模型优化数据保留");
    await user.click(button);

    expect(screen.queryByRole("dialog", { name: "确认删除任务" })).not.toBeInTheDocument();
    expect(api.deleteTask).not.toHaveBeenCalled();
  });

  it("disables task deletion immediately after feedback is saved", async () => {
    const feedbackView = {
      imageId: "ok-image",
      originalFilename: "terminal-ok.png",
      status: "succeeded" as const,
      detections: [],
      missedRegionCandidates: ["label1" as const],
      feedback: null,
    };
    const api = client({
      getImageFeedback: vi.fn().mockResolvedValue(feedbackView),
      updateImageFeedback: vi.fn().mockResolvedValue({
        ...feedbackView,
        feedback: {
          id: "feedback-1",
          items: [],
          missedRegions: ["label1"],
          createdAt: "2026-09-23T01:00:00Z",
          updatedAt: "2026-09-23T01:00:00Z",
        },
      }),
    });
    const user = userEvent.setup();
    render(<HistoryPage client={api} initialTaskId="task-id" initialImageId="ok-image" />);

    await user.click(await screen.findByRole("button", { name: "结果反馈" }));
    await user.click(await screen.findByRole("checkbox", { name: "label1 漏检" }));
    await user.click(screen.getByRole("button", { name: "提交反馈" }));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: "删除 T20260920-0100" })).toBeDisabled();
    });
    expect(screen.getByRole("button", { name: "删除 T20260920-0100" })).toHaveAttribute(
      "title",
      "包含人工反馈，已作为模型优化数据保留",
    );
  });
});
