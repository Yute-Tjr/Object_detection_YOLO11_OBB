import "@testing-library/jest-dom/vitest";

import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { HealthResponse, TaskDetail } from "../api/types";
import { pollingDelay, TasksPage } from "./TasksPage";


const health: HealthResponse = {
  apiReady: true,
  databaseReady: true,
  workerReady: true,
  modelsReady: true,
  models: [
    { modelType: "detector", name: "YOLO11l-OBB", version: "baseline", ready: true },
    { modelType: "anomaly", name: "ResNet18-label3", version: "best", ready: true },
    { modelType: "anomaly", name: "ResNet18-label5", version: "best", ready: true },
  ],
};

const runningTask: TaskDetail = {
  id: "task-1",
  displayId: "T20260920-0001",
  name: "端子_产线A_早班",
  operator: "张三",
  note: "",
  status: "running",
  currentStage: "object_detection",
  totalImages: 1,
  completedImages: 0,
  succeededImages: 0,
  failedImages: 0,
  detectorModel: "YOLO11l-OBB",
  createdAt: "2026-09-20T08:00:00Z",
  images: [
    {
      id: "image-1",
      sequenceNo: 0,
      originalFilename: "terminal.png",
      status: "running",
      stage: "object_detection",
      overallResult: "UNKNOWN",
      width: 1440,
      height: 3072,
      sizeBytes: 2300000,
      originalUrl: "/api/v1/images/image-1/original",
      resultUrl: "/api/v1/images/image-1/result",
    },
  ],
};

function fakeClient(overrides = {}) {
  return {
    getHealth: vi.fn().mockResolvedValue(health),
    createTask: vi.fn().mockResolvedValue(runningTask),
    getTask: vi.fn().mockResolvedValue(runningTask),
    listTasks: vi.fn().mockResolvedValue({ items: [], total: 0, offset: 0, limit: 100 }),
    getTaskImages: vi.fn(),
    getImage: vi.fn(),
    retryImage: vi.fn(),
    ...overrides,
  };
}

function files(count: number) {
  return Array.from(
    { length: count },
    (_, index) => new File(["image"], `terminal-${index}.png`, { type: "image/png" }),
  );
}

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});


describe("TasksPage", () => {
  it("disables start when 101 files are selected", async () => {
    render(<TasksPage client={fakeClient()} />);
    await screen.findByText("YOLO11l-OBB");

    fireEvent.change(screen.getByLabelText("选择图片"), {
      target: { files: files(101) },
    });

    expect(screen.getByText("单次最多 100 张图片")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "开始检测" })).toBeDisabled();
  });

  it("removing and clearing files updates the selected count", async () => {
    const user = userEvent.setup();
    render(<TasksPage client={fakeClient()} />);
    await screen.findByText("YOLO11l-OBB");
    fireEvent.change(screen.getByLabelText("选择图片"), {
      target: { files: files(2) },
    });
    expect(screen.getByText(/已选择 2 张图片/)).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "移除 terminal-0.png" }));
    expect(screen.getByText(/已选择 1 张图片/)).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "清空已选图片" }));
    expect(screen.getByText(/已选择 0 张图片/)).toBeInTheDocument();
  });

  it("explains which model is unavailable and disables start", async () => {
    const unavailable = {
      ...health,
      modelsReady: false,
      models: health.models.map((model) =>
        model.name === "ResNet18-label5" ? { ...model, ready: false } : model,
      ),
    };
    render(<TasksPage client={fakeClient({ getHealth: vi.fn().mockResolvedValue(unavailable) })} />);
    await screen.findByText(/ResNet18-label5 尚未就绪/);
    fireEvent.change(screen.getByLabelText("选择图片"), {
      target: { files: files(1) },
    });
    expect(screen.getByRole("button", { name: "开始检测" })).toBeDisabled();
  });

  it("starts a valid batch and renders task progress and comparison controls", async () => {
    const user = userEvent.setup();
    render(<TasksPage client={fakeClient()} />);
    await screen.findByText("YOLO11l-OBB");
    fireEvent.change(screen.getByLabelText("选择图片"), {
      target: { files: files(1) },
    });
    await user.type(screen.getByLabelText("操作员"), "张三");

    await user.click(screen.getByRole("button", { name: "开始检测" }));

    expect((await screen.findAllByText("T20260920-0001")).length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText("正在检测 0 / 1")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "上一张图片" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "下一张图片" })).toBeInTheDocument();
    expect(screen.getByAltText("检测前原图")).toHaveAttribute("width", "1440");
    expect(screen.getByAltText("检测前原图")).toHaveAttribute("height", "3072");
    expect(screen.getByAltText("检测后结果")).toHaveAttribute("width", "1440");
    expect(screen.getByAltText("检测后结果")).toHaveAttribute("height", "3072");
  });

  it("requires operator and clears metadata only after creation succeeds", async () => {
    const api = fakeClient();
    const user = userEvent.setup();
    const { unmount } = render(<TasksPage client={api} />);
    await screen.findByText("YOLO11l-OBB");
    fireEvent.change(screen.getByLabelText("选择图片"), {
      target: { files: files(1) },
    });
    expect(screen.getByRole("button", { name: "开始检测" })).toBeDisabled();

    await user.type(screen.getByLabelText("操作员"), "张三");
    await user.type(screen.getByLabelText("任务名称"), "早班");
    await user.type(screen.getByLabelText("备注"), "首件");
    await user.click(screen.getByRole("button", { name: "开始检测" }));

    expect(api.createTask).toHaveBeenCalledWith(expect.any(Array), {
      operator: "张三",
      name: "早班",
      note: "首件",
    });
    expect(screen.getByLabelText("操作员")).toHaveValue("");
    expect(screen.getByLabelText("任务名称")).toHaveValue("");
    expect(screen.getByLabelText("备注")).toHaveValue("");

    unmount();
    render(<TasksPage client={fakeClient()} />);
    expect(screen.getByLabelText("操作员")).toHaveValue("");
  });

  it.each(["succeeded", "partial_failed", "failed"] as const)(
    "does not continue polling after %s",
    async (status) => {
      const terminal = { ...runningTask, status, currentStage: "complete" as const };
      const client = fakeClient({ getTask: vi.fn().mockResolvedValue(terminal) });
      render(<TasksPage client={client} initialTask={terminal} />);
      await screen.findByText("T20260920-0001");
      await new Promise((resolve) => setTimeout(resolve, 10));
      expect(client.getTask).not.toHaveBeenCalled();
    },
  );

  it("uses a ten-second polling delay when the page is hidden", () => {
    expect(pollingDelay(false)).toBe(1000);
    expect(pollingDelay(true)).toBe(10000);
  });
});
