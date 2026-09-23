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
      detectionId: "d1",
      regionLabel: "label1_thick",
      logicalRegion: "label1",
      anomaly: null,
      color: null,
    },
    {
      detectionId: "d3",
      regionLabel: "label3",
      logicalRegion: "label3",
      anomaly: null,
      color: null,
    },
  ],
  missedRegionCandidates: ["label2", "label4", "label5", "label6"],
  feedback: null,
};

function client(view: ImageFeedbackView = emptyFeedback) {
  return {
    getImageFeedback: vi.fn().mockResolvedValue(view),
    updateImageFeedback: vi.fn().mockImplementation(async (_imageId, payload) => ({
      ...view,
      feedback: {
        id: "feedback-1",
        items: payload.items.map((item: { detectionId: string; verdict: "OK" | "NG"; color?: "B" | "G" | "R" | "W" }) => ({
          ...item,
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
  it("starts blank, distinguishes label1 variant, and submits detected plus missed regions", async () => {
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
    expect(within(dialog).getByText("实际检测：label1_thick")).toBeInTheDocument();
    expect(within(dialog).getByRole("radio", { name: "label1 OK" })).not.toBeChecked();
    expect(within(dialog).getByRole("radio", { name: "label1 NG" })).not.toBeChecked();
    expect(within(dialog).queryByRole("radio", { name: "label3 颜色 B" })).not.toBeInTheDocument();

    await user.click(within(dialog).getByRole("button", { name: "提交反馈" }));
    expect(within(dialog).getByRole("alert")).toHaveTextContent("请完成所有已检测区域的判定");

    await user.click(within(dialog).getByRole("radio", { name: "label1 NG" }));
    await user.click(within(dialog).getByRole("radio", { name: "label1 颜色 R" }));
    await user.click(within(dialog).getByRole("radio", { name: "label3 OK" }));
    await user.click(within(dialog).getByRole("checkbox", { name: "label2 漏检" }));
    await user.click(within(dialog).getByRole("button", { name: "提交反馈" }));

    await waitFor(() => expect(api.updateImageFeedback).toHaveBeenCalledWith("image-1", {
      items: [
        { detectionId: "d1", verdict: "NG", color: "R" },
        { detectionId: "d3", verdict: "OK" },
      ],
      missedRegions: ["label2"],
    }));
    expect(onSaved).toHaveBeenCalledTimes(1);
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
          { detectionId: "d1", regionLabel: "label1_thick", logicalRegion: "label1", verdict: "OK", color: "G" },
          { detectionId: "d3", regionLabel: "label3", logicalRegion: "label3", verdict: "NG", color: null },
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
    expect(screen.getByRole("radio", { name: "label3 NG" })).toBeChecked();
    expect(screen.getByRole("checkbox", { name: "label4 漏检" })).toBeChecked();
    await user.click(screen.getByRole("radio", { name: "label1 NG" }));
    await user.click(screen.getByRole("radio", { name: "label1 颜色 R" }));
    await user.click(screen.getByRole("button", { name: "提交反馈" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("保存失败");
    expect(screen.getByRole("radio", { name: "label1 NG" })).toBeChecked();
    expect(screen.getByRole("radio", { name: "label1 颜色 R" })).toBeChecked();
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
