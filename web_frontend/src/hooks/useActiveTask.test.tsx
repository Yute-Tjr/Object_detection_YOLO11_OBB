import "@testing-library/jest-dom/vitest";

import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ApiClient } from "../api/client";
import type { TaskDetail } from "../api/types";
import { pollingDelay, useActiveTask } from "./useActiveTask";


const runningTask: TaskDetail = {
  id: "task-1",
  displayId: "T20260920-0001",
  operator: "张三",
  status: "running",
  currentStage: "object_detection",
  totalImages: 1,
  completedImages: 0,
  succeededImages: 0,
  failedImages: 0,
  detectorModel: "YOLO11l-OBB",
  createdAt: "2026-09-20T08:00:00Z",
  images: [],
};

function Probe({ client }: { client: ApiClient }) {
  const { task } = useActiveTask(runningTask, client);
  return <span>{task?.completedImages}</span>;
}

function fakeClient(getTask: ApiClient["getTask"]): ApiClient {
  return {
    getHealth: vi.fn(),
    createTask: vi.fn(),
    listTasks: vi.fn(),
    getTask,
    getTaskImages: vi.fn(),
    getImage: vi.fn(),
    retryImage: vi.fn(),
  };
}

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe("useActiveTask", () => {
  it("backs off after failure and resets delay after success", async () => {
    vi.useFakeTimers();
    const refreshed = { ...runningTask, completedImages: 1 };
    const getTask = vi.fn()
      .mockRejectedValueOnce(new Error("服务暂不可用"))
      .mockResolvedValue(refreshed);
    render(<Probe client={fakeClient(getTask)} />);

    await vi.advanceTimersByTimeAsync(1_000);
    expect(getTask).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1_999);
    expect(getTask).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1);
    expect(getTask).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(999);
    expect(getTask).toHaveBeenCalledTimes(2);
    await vi.advanceTimersByTimeAsync(1);
    expect(getTask).toHaveBeenCalledTimes(3);
  });

  it("caps visible and hidden polling delays at ten seconds", () => {
    expect(pollingDelay(false, 0)).toBe(1_000);
    expect(pollingDelay(false, 1)).toBe(2_000);
    expect(pollingDelay(false, 4)).toBe(10_000);
    expect(pollingDelay(true, 4)).toBe(10_000);
  });
});
