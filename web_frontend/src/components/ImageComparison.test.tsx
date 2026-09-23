import "@testing-library/jest-dom/vitest";

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { ImageSummary } from "../api/types";
import { ImageComparison } from "./ImageComparison";


const completedImage: ImageSummary = {
  id: "image-1",
  sequenceNo: 0,
  originalFilename: "terminal.png",
  status: "succeeded",
  stage: "complete",
  overallResult: "OK",
  width: 380,
  height: 1606,
  sizeBytes: 1800000,
  originalUrl: "/images/original",
  resultUrl: "/images/result",
};

function client(overrides = {}) {
  return {
    getImageFeedback: vi.fn().mockResolvedValue({
      imageId: "image-1",
      originalFilename: "terminal.png",
      status: "succeeded",
      detections: [],
      missedRegionCandidates: ["label1"],
      feedback: null,
    }),
    updateImageFeedback: vi.fn(),
    deleteImageFeedback: vi.fn(),
    ...overrides,
  };
}

afterEach(cleanup);


describe("ImageComparison feedback entry", () => {
  it("opens feedback for the selected completed image", async () => {
    const api = client();
    const user = userEvent.setup();
    render(
      <ImageComparison
        images={[completedImage]}
        selectedIndex={0}
        onSelectedIndexChange={vi.fn()}
        feedbackClient={api}
      />,
    );

    const button = screen.getByRole("button", { name: "结果反馈" });
    expect(button).toBeEnabled();
    await user.click(button);

    expect(await screen.findByRole("dialog", { name: "检测结果反馈" })).toBeInTheDocument();
    expect(api.getImageFeedback).toHaveBeenCalledWith("image-1", expect.any(AbortSignal));
  });

  it("shows why feedback is disabled while the image is processing", () => {
    render(
      <ImageComparison
        images={[{ ...completedImage, status: "running", stage: "rendering" }]}
        selectedIndex={0}
        onSelectedIndexChange={vi.fn()}
        feedbackClient={client()}
      />,
    );

    expect(screen.getByRole("button", { name: "结果反馈" })).toBeDisabled();
    expect(screen.getByText("图片处理完成后可反馈")).toBeInTheDocument();
  });

  it("reports the selected image when its saved feedback is deleted", async () => {
    const feedbackView = {
      imageId: "image-1",
      originalFilename: "terminal.png",
      status: "succeeded" as const,
      detections: [],
      missedRegionCandidates: ["label1" as const],
      feedback: {
        id: "feedback-1",
        items: [],
        missedRegions: ["label1" as const],
        createdAt: "2026-09-23T01:00:00Z",
        updatedAt: "2026-09-23T01:00:00Z",
      },
    };
    const api = client({
      getImageFeedback: vi.fn().mockResolvedValue(feedbackView),
      deleteImageFeedback: vi.fn().mockResolvedValue({ ...feedbackView, feedback: null }),
    });
    const onFeedbackDeleted = vi.fn();
    const user = userEvent.setup();
    render(
      <ImageComparison
        images={[completedImage]}
        selectedIndex={0}
        onSelectedIndexChange={vi.fn()}
        feedbackClient={api}
        onFeedbackDeleted={onFeedbackDeleted}
      />,
    );

    await user.click(screen.getByRole("button", { name: "结果反馈" }));
    await user.click(await screen.findByRole("button", { name: "撤销已提交反馈" }));
    await user.click(screen.getByRole("button", { name: "确认撤销" }));

    await waitFor(() => expect(onFeedbackDeleted).toHaveBeenCalledWith("image-1"));
  });
});
