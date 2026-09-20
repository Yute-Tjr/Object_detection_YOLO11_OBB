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
    name: "端子_产线A_早班",
    note: "首件确认",
    status: "running",
    currentStage: "anomaly_classification",
    totalImages: 100,
    completedImages: 63,
    succeededImages: 63,
    failedImages: 0,
    detectorModel: "YOLO11l-OBB",
    createdAt: "2026-09-20T02:24:00Z",
  },
  {
    id: "success-id",
    displayId: "T20260920-0002",
    name: "端子_产线B_早班",
    note: "",
    status: "succeeded",
    currentStage: "complete",
    totalImages: 12,
    completedImages: 12,
    succeededImages: 12,
    failedImages: 0,
    detectorModel: "YOLO11l-OBB",
    createdAt: "2026-09-20T01:24:00Z",
  },
  {
    id: "partial-id",
    displayId: "T20260920-0003",
    name: "端子_产线C_夜班",
    note: "复检批次",
    status: "partial_failed",
    currentStage: "complete",
    totalImages: 20,
    completedImages: 20,
    succeededImages: 18,
    failedImages: 2,
    detectorModel: "YOLO11l-OBB",
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

  it("expands a row to show task metadata", async () => {
    const user = userEvent.setup();
    render(<TaskTable tasks={tasks} onSelectTask={vi.fn()} />);

    await user.click(screen.getByRole("button", { name: "展开任务 T20260920-0001" }));
    expect(screen.getByText("端子_产线A_早班")).toBeInTheDocument();
    expect(screen.getByText("running-id")).toBeInTheDocument();
    expect(screen.getByText("操作员")).toBeInTheDocument();
    expect(screen.getByText("首件确认")).toBeInTheDocument();
  });
});
