import "@testing-library/jest-dom/vitest";

import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { TaskSummary } from "../api/types";
import { TaskTable } from "./TaskTable";


const tasks: TaskSummary[] = [
  {
    id: "running-id",
    displayId: "T20260920-0001",
    status: "running",
    currentStage: "anomaly_classification",
    totalImages: 100,
    completedImages: 63,
    succeededImages: 63,
    failedImages: 0,
    hasFeedback: false,
    createdAt: "2026-09-20T02:24:00Z",
  },
  {
    id: "success-id",
    displayId: "T20260920-0002",
    status: "succeeded",
    currentStage: "complete",
    totalImages: 12,
    completedImages: 12,
    succeededImages: 12,
    failedImages: 0,
    hasFeedback: false,
    createdAt: "2026-09-20T01:24:00Z",
  },
  {
    id: "partial-id",
    displayId: "T20260920-0003",
    status: "partial_failed",
    currentStage: "complete",
    totalImages: 20,
    completedImages: 20,
    succeededImages: 18,
    failedImages: 2,
    hasFeedback: false,
    createdAt: "2026-09-19T18:24:00Z",
  },
];

afterEach(cleanup);

describe("TaskTable", () => {
  it("shows status counts and treats partial failures as failed", async () => {
    const user = userEvent.setup();
    render(<TaskTable tasks={tasks} onSelectTask={vi.fn()} />);

    expect(screen.getByRole("button", { name: "进行中 1" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "全部 3" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "成功 1" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "失败 1" })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: "失败 1" }));
    expect(screen.getByText("T20260920-0003")).toBeInTheDocument();
    expect(screen.queryByText("T20260920-0002")).not.toBeInTheDocument();
    expect(screen.getByText("部分失败")).toHaveClass("status-label--failed");
  });

  it("shows only operational columns without model identity or expansion", () => {
    render(<TaskTable tasks={tasks} onSelectTask={vi.fn()} />);

    expect(screen.queryByRole("columnheader", { name: "模型" })).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /展开任务/ })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "预览 T20260920-0001" })).toBeInTheDocument();
  });

  it("uses an explicit preview action instead of a clickable task id", async () => {
    const user = userEvent.setup();
    const onPreview = vi.fn();
    render(<TaskTable tasks={tasks} onSelectTask={onPreview} />);

    expect(screen.getByText("T20260920-0001").tagName).not.toBe("BUTTON");
    await user.click(
      screen.getByRole("button", { name: "预览 T20260920-0001" }),
    );

    expect(onPreview).toHaveBeenCalledWith(tasks[0]);
  });

  it("always displays task timestamps in Beijing time", async () => {
    const successTask = tasks[1];
    render(<TaskTable tasks={[successTask]} onSelectTask={vi.fn()} initialFilter="all" />);

    expect(screen.getByText("2026-09-20 09:24")).toBeInTheDocument();
  });

  it("disables deletion for tasks that contain feedback", async () => {
    const onDeleteTask = vi.fn();
    const user = userEvent.setup();
    const feedbackTask = { ...tasks[1], hasFeedback: true };
    render(
      <TaskTable
        tasks={[feedbackTask]}
        onSelectTask={vi.fn()}
        onDeleteTask={onDeleteTask}
        initialFilter="all"
      />,
    );

    const button = screen.getByRole("button", { name: "删除 T20260920-0002" });
    expect(button).toBeDisabled();
    expect(button).toHaveAttribute("title", "包含人工反馈，已作为模型优化数据保留");
    await user.click(button);
    expect(onDeleteTask).not.toHaveBeenCalled();
  });
});
