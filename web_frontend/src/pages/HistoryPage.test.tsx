import "@testing-library/jest-dom/vitest";

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { TaskDetail, TaskPage } from "../api/types";
import { HistoryPage } from "./HistoryPage";


const detail: TaskDetail = {
  id: "task-id",
  displayId: "T20260920-0100",
  name: "端子_产线A_早班",
  note: "历史复检",
  status: "partial_failed",
  currentStage: "complete",
  totalImages: 2,
  completedImages: 2,
  succeededImages: 1,
  failedImages: 1,
  detectorModel: "YOLO11l-OBB",
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
    getHealth: vi.fn(),
    createTask: vi.fn(),
    listTasks: vi.fn().mockResolvedValue(page),
    getTask: vi.fn().mockResolvedValue(detail),
    getTaskImages: vi.fn(),
    getImage: vi.fn(),
    retryImage: vi.fn().mockResolvedValue({ ...detail.images[1], status: "queued" }),
    ...overrides,
  };
}

afterEach(() => {
  cleanup();
  window.history.replaceState({}, "", "/");
});

describe("HistoryPage", () => {
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
});
