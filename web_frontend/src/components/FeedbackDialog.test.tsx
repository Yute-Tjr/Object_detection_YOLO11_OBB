import "@testing-library/jest-dom/vitest";

import { cleanup, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ImageFeedbackView } from "../api/types";
import { FeedbackDialog } from "./FeedbackDialog";


const emptyFeedback: ImageFeedbackView = {
  imageId: "image-1",
  originalFilename: "terminal.png",
  status: "succeeded",
  detections: [
    {
      detectionId: "d3",
      regionLabel: "label3",
      logicalRegion: "label3",
      anomaly: {
        classifierType: "anomaly",
        predictedLabel: "OK",
        confidence: 0.97,
      },
      color: null,
    },
    {
      detectionId: "d1",
      regionLabel: "label1_thick",
      logicalRegion: "label1",
      anomaly: null,
      color: null,
    },
    {
      detectionId: "d6",
      regionLabel: "label6",
      logicalRegion: "label6",
      anomaly: null,
      color: null,
    },
    {
      detectionId: "d2",
      regionLabel: "label2",
      logicalRegion: "label2",
      anomaly: null,
      color: null,
    },
  ],
  missedRegionCandidates: ["label5", "label4"],
  feedback: null,
};

function client(view: ImageFeedbackView = emptyFeedback) {
  return {
    getImageFeedback: vi.fn().mockResolvedValue(view),
    deleteImageFeedback: vi.fn().mockResolvedValue({ ...view, feedback: null }),
    updateImageFeedback: vi.fn().mockImplementation(async (_imageId, payload) => ({
      ...view,
      feedback: {
        id: "feedback-1",
        items: payload.items.map((item: { detectionId: string; verdict: "OK" | "NG"; color?: "B" | "G" | "R" | "W" }) => ({
          ...item,
          source: "manual" as const,
          regionLabel: item.detectionId === "d1" ? "label1_thick" : "label3",
          logicalRegion: item.detectionId === "d1" ? "label1" : "label3",
        })),
        missedRegions: payload.missedRegions,
        createdAt: "2026-09-23T01:00:00Z",
        updatedAt: "2026-09-23T01:00:00Z",
      },
    })),
  };
}

afterEach(cleanup);


describe("FeedbackDialog", () => {
  it("sorts logical regions and submits only the manually selected region", async () => {
    const api = client();
    const onSaved = vi.fn();
    const user = userEvent.setup();
    render(
      <FeedbackDialog
        imageId="image-1"
        filename="terminal.png"
        client={api}
        onClose={vi.fn()}
        onSaved={onSaved}
      />,
    );

    const dialog = await screen.findByRole("dialog", { name: "检测结果反馈" });
    expect(within(dialog).getAllByTestId("feedback-region-name").map((item) => item.textContent)).toEqual([
      "label1实际检测：label1_thick",
      "label2",
      "label3",
      "label6",
    ]);
    expect(within(dialog).getByText("实际检测：label1_thick")).toBeInTheDocument();
    expect(within(dialog).getByText("沿用模型：OK")).toBeInTheDocument();
    expect(within(dialog).getAllByText("暂无分类结果")).toHaveLength(3);
    expect(within(dialog).queryByRole("checkbox", { name: /人工反馈/ })).not.toBeInTheDocument();
    expect(within(dialog).queryByRole("button", { name: "撤销已提交反馈" })).not.toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: "提交反馈" }));
    expect(within(dialog).getByRole("alert")).toHaveTextContent("请至少反馈一个区域或选择一个漏检区域");

    await user.click(within(dialog).getByRole("radio", { name: "label3 NG" }));
    await user.click(within(dialog).getByRole("button", { name: "提交反馈" }));

    await waitFor(() => expect(api.updateImageFeedback).toHaveBeenCalledWith("image-1", {
      items: [{ detectionId: "d3", verdict: "NG" }],
      missedRegions: [],
    }));
    expect(onSaved).toHaveBeenCalledTimes(1);
  });

  it("treats a color choice as manual feedback and still requires a verdict", async () => {
    const api = client();
    const user = userEvent.setup();
    render(<FeedbackDialog imageId="image-1" filename="terminal.png" client={api} onClose={vi.fn()} onSaved={vi.fn()} />);

    await user.click(await screen.findByRole("radio", { name: "label1 颜色 B" }));
    await user.click(screen.getByRole("button", { name: "提交反馈" }));

    expect(screen.getByRole("alert")).toHaveTextContent("请完成已选择人工反馈区域的判定");
    expect(screen.getByRole("button", { name: "label1 恢复模型结果" })).toBeInTheDocument();

    await user.click(screen.getByRole("radio", { name: "label1 OK" }));
    await user.click(screen.getByRole("button", { name: "提交反馈" }));

    await waitFor(() => expect(api.updateImageFeedback).toHaveBeenCalledWith("image-1", {
      items: [{ detectionId: "d1", verdict: "OK", color: "B" }],
      missedRegions: [],
    }));
  });

  it("requires at least one missed region when no detection exists", async () => {
    const api = client({
      ...emptyFeedback,
      detections: [],
      missedRegionCandidates: ["label1", "label2", "label3", "label4", "label5", "label6"],
    });
    const user = userEvent.setup();
    render(<FeedbackDialog imageId="image-1" filename="terminal.png" client={api} onClose={vi.fn()} onSaved={vi.fn()} />);

    await user.click(await screen.findByRole("button", { name: "提交反馈" }));

    expect(screen.getByRole("alert")).toHaveTextContent("请至少选择一个漏检区域");
    expect(api.updateImageFeedback).not.toHaveBeenCalled();
  });

  it("restores existing feedback and keeps the draft after submit failure", async () => {
    const view: ImageFeedbackView = {
      ...emptyFeedback,
      feedback: {
        id: "feedback-1",
        items: [
          { detectionId: "d1", regionLabel: "label1_thick", logicalRegion: "label1", source: "manual", verdict: "OK", color: "G" },
          { detectionId: "d2", regionLabel: "label2", logicalRegion: "label2", source: "unreviewed", verdict: null, color: null },
          { detectionId: "d3", regionLabel: "label3", logicalRegion: "label3", source: "model", verdict: "OK", color: null },
          { detectionId: "d6", regionLabel: "label6", logicalRegion: "label6", source: "unreviewed", verdict: null, color: null },
        ],
        missedRegions: ["label4"],
        createdAt: "2026-09-23T01:00:00Z",
        updatedAt: "2026-09-23T01:00:00Z",
      },
    };
    const api = client(view);
    api.updateImageFeedback.mockRejectedValueOnce(new Error("保存失败"));
    const user = userEvent.setup();
    render(<FeedbackDialog imageId="image-1" filename="terminal.png" client={api} onClose={vi.fn()} onSaved={vi.fn()} />);

    expect(await screen.findByRole("radio", { name: "label1 OK" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "label1 颜色 G" })).toBeChecked();
    expect(screen.getByRole("button", { name: "label1 恢复模型结果" })).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "label3 恢复模型结果" })).not.toBeInTheDocument();
    expect(screen.getByRole("checkbox", { name: "label4 漏检" })).toBeChecked();
    await user.click(screen.getByRole("radio", { name: "label1 NG" }));
    await user.click(screen.getByRole("radio", { name: "label1 颜色 R" }));
    await user.click(screen.getByRole("button", { name: "提交反馈" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("保存失败");
    expect(screen.getByRole("radio", { name: "label1 NG" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "label1 颜色 R" })).toBeChecked();
  });

  it("can remove an existing manual item and keep only a missed-region correction", async () => {
    const view: ImageFeedbackView = {
      ...emptyFeedback,
      feedback: {
        id: "feedback-1",
        items: [
          { detectionId: "d1", regionLabel: "label1_thick", logicalRegion: "label1", source: "manual", verdict: "OK", color: "G" },
          { detectionId: "d2", regionLabel: "label2", logicalRegion: "label2", source: "unreviewed", verdict: null, color: null },
          { detectionId: "d3", regionLabel: "label3", logicalRegion: "label3", source: "model", verdict: "OK", color: null },
          { detectionId: "d6", regionLabel: "label6", logicalRegion: "label6", source: "unreviewed", verdict: null, color: null },
        ],
        missedRegions: [],
        createdAt: "2026-09-23T01:00:00Z",
        updatedAt: "2026-09-23T01:00:00Z",
      },
    };
    const api = client(view);
    const user = userEvent.setup();
    render(<FeedbackDialog imageId="image-1" filename="terminal.png" client={api} onClose={vi.fn()} onSaved={vi.fn()} />);

    await user.click(await screen.findByRole("button", { name: "label1 恢复模型结果" }));
    await user.click(screen.getByRole("checkbox", { name: "label4 漏检" }));
    await user.click(screen.getByRole("button", { name: "提交反馈" }));

    await waitFor(() => expect(api.updateImageFeedback).toHaveBeenCalledWith("image-1", {
      items: [],
      missedRegions: ["label4"],
    }));
  });

  it("requires confirmation before deleting saved feedback and resets the draft after deletion", async () => {
    const view: ImageFeedbackView = {
      ...emptyFeedback,
      feedback: {
        id: "feedback-1",
        items: [
          { detectionId: "d1", regionLabel: "label1_thick", logicalRegion: "label1", source: "manual", verdict: "OK", color: "G" },
          { detectionId: "d2", regionLabel: "label2", logicalRegion: "label2", source: "unreviewed", verdict: null, color: null },
          { detectionId: "d3", regionLabel: "label3", logicalRegion: "label3", source: "model", verdict: "OK", color: null },
          { detectionId: "d6", regionLabel: "label6", logicalRegion: "label6", source: "unreviewed", verdict: null, color: null },
        ],
        missedRegions: ["label4"],
        createdAt: "2026-09-23T01:00:00Z",
        updatedAt: "2026-09-23T01:00:00Z",
      },
    };
    const api = client(view);
    const onDeleted = vi.fn();
    const user = userEvent.setup();
    render(
      <FeedbackDialog
        imageId="image-1"
        filename="terminal.png"
        client={api}
        onClose={vi.fn()}
        onSaved={vi.fn()}
        onDeleted={onDeleted}
      />,
    );

    await user.click(await screen.findByRole("button", { name: "撤销已提交反馈" }));
    const confirmation = screen.getByRole("alertdialog", { name: "撤销已提交反馈？" });
    expect(within(confirmation).getByText("将永久删除你对这张图片提交的反馈，检测结果不会被删除。")).toBeInTheDocument();
    await user.click(within(confirmation).getByRole("button", { name: "取消" }));
    expect(api.deleteImageFeedback).not.toHaveBeenCalled();

    await user.click(screen.getByRole("button", { name: "撤销已提交反馈" }));
    await user.click(screen.getByRole("button", { name: "确认撤销" }));

    await waitFor(() => expect(api.deleteImageFeedback).toHaveBeenCalledWith("image-1"));
    expect(onDeleted).toHaveBeenCalledWith(expect.objectContaining({ feedback: null }));
    expect(screen.queryByRole("button", { name: "撤销已提交反馈" })).not.toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "label1 OK" })).not.toBeChecked();
    expect(screen.getByRole("checkbox", { name: "label4 漏检" })).not.toBeChecked();
  });

  it("keeps saved feedback when deletion fails", async () => {
    const view: ImageFeedbackView = {
      ...emptyFeedback,
      feedback: {
        id: "feedback-1",
        items: [
          { detectionId: "d1", regionLabel: "label1_thick", logicalRegion: "label1", source: "manual", verdict: "OK", color: "G" },
        ],
        missedRegions: [],
        createdAt: "2026-09-23T01:00:00Z",
        updatedAt: "2026-09-23T01:00:00Z",
      },
    };
    const api = client(view);
    api.deleteImageFeedback.mockRejectedValueOnce(new Error("撤销失败"));
    const user = userEvent.setup();
    render(<FeedbackDialog imageId="image-1" filename="terminal.png" client={api} onClose={vi.fn()} onSaved={vi.fn()} />);

    await user.click(await screen.findByRole("button", { name: "撤销已提交反馈" }));
    await user.click(screen.getByRole("button", { name: "确认撤销" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("撤销失败");
    expect(screen.queryByRole("alertdialog", { name: "撤销已提交反馈？" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "撤销已提交反馈" })).toBeInTheDocument();
    expect(screen.getByRole("radio", { name: "label1 OK" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "label1 颜色 G" })).toBeChecked();
  });

  it("allows retry after loading fails", async () => {
    const api = client();
    api.getImageFeedback
      .mockRejectedValueOnce(new Error("加载失败"))
      .mockResolvedValueOnce(emptyFeedback);
    const user = userEvent.setup();
    render(<FeedbackDialog imageId="image-1" filename="terminal.png" client={api} onClose={vi.fn()} onSaved={vi.fn()} />);

    expect(await screen.findByRole("alert")).toHaveTextContent("加载失败");
    await user.click(screen.getByRole("button", { name: "重试加载反馈" }));

    expect(await screen.findByRole("radio", { name: "label1 OK" })).toBeInTheDocument();
    expect(api.getImageFeedback).toHaveBeenCalledTimes(2);
  });
});
