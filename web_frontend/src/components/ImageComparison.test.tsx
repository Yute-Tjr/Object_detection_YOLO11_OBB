import "@testing-library/jest-dom/vitest";

import { cleanup, render, screen } from "@testing-library/react";
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

function client() {
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
});
