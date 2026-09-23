import { CaretLeft, CaretRight, ChatCircleText, ImageSquare } from "@phosphor-icons/react";
import { useState } from "react";

import type { ImageSummary } from "../api/types";
import { FeedbackDialog, type FeedbackClient } from "./FeedbackDialog";


interface ImageComparisonProps {
  images: ImageSummary[];
  selectedIndex: number;
  onSelectedIndexChange: (index: number) => void;
  feedbackClient?: FeedbackClient;
  onFeedbackSaved?: (imageId: string) => void;
  onFeedbackDeleted?: (imageId: string) => void;
}


function imageMeta(image: ImageSummary) {
  const size = image.sizeBytes >= 1_000_000
    ? `${(image.sizeBytes / 1_000_000).toFixed(1)} MB`
    : `${Math.max(1, Math.round(image.sizeBytes / 1000))} KB`;
  return `${image.originalFilename}  ·  ${image.width} × ${image.height}  ·  ${size}`;
}


export function ImageComparison({
  images,
  selectedIndex,
  onSelectedIndexChange,
  feedbackClient,
  onFeedbackSaved,
  onFeedbackDeleted,
}: ImageComparisonProps) {
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const image = images[selectedIndex];
  if (!image) return null;
  const feedbackEnabled = image.status === "succeeded" && image.stage === "complete";
  return (
    <>
      <section className="comparison-section" aria-label="检测前后对比">
        <header className="comparison-section__header">
          <div>
            <h2>检测结果</h2>
            <span className={`result-badge result-badge--${image.overallResult.toLowerCase()}`}>
              {image.overallResult === "UNKNOWN" ? "分析中" : image.overallResult}
            </span>
            <span className="reserved-copy">颜色分类：接口预留</span>
          </div>
          <div className="comparison-actions">
            {!feedbackEnabled ? <span className="feedback-disabled-reason">图片处理完成后可反馈</span> : null}
            <button
              className="feedback-button"
              type="button"
              disabled={!feedbackEnabled}
              aria-label="结果反馈"
              onClick={() => setFeedbackOpen(true)}
            >
              <ChatCircleText size={17} />反馈
            </button>
            <div className="image-pagination">
              <span>{selectedIndex + 1} / {images.length}</span>
              <button
                aria-label="上一张图片"
                disabled={selectedIndex === 0}
                onClick={() => onSelectedIndexChange(selectedIndex - 1)}
              ><CaretLeft size={18} /></button>
              <button
                aria-label="下一张图片"
                disabled={selectedIndex >= images.length - 1}
                onClick={() => onSelectedIndexChange(selectedIndex + 1)}
              ><CaretRight size={18} /></button>
            </div>
          </div>
        </header>
        <div className="comparison-grid">
          <figure className="comparison-card">
            <figcaption>检测前</figcaption>
            <div className="comparison-card__canvas">
              <img
                src={image.originalUrl}
                alt="检测前原图"
                width={image.width}
                height={image.height}
              />
            </div>
            <p>{imageMeta(image)}</p>
          </figure>
          <figure className="comparison-card">
            <figcaption>检测后</figcaption>
            <div className="comparison-card__canvas">
              {image.resultUrl ? (
                <img
                  src={image.resultUrl}
                  alt="检测后结果"
                />
              ) : (
                <div className="result-placeholder">
                  <ImageSquare size={34} />
                  <span>结果生成中</span>
                </div>
              )}
            </div>
            <p>{stageLabelForImage(image.stage)}</p>
          </figure>
        </div>
      </section>
      {feedbackOpen ? (
        <FeedbackDialog
          key={image.id}
          imageId={image.id}
          filename={image.originalFilename}
          client={feedbackClient}
          onClose={() => setFeedbackOpen(false)}
          onSaved={() => onFeedbackSaved?.(image.id)}
          onDeleted={() => onFeedbackDeleted?.(image.id)}
        />
      ) : null}
    </>
  );
}

function stageLabelForImage(stage: ImageSummary["stage"]) {
  const labels: Record<ImageSummary["stage"], string> = {
    pending: "等待处理",
    object_detection: "正在进行分区域目标检测",
    anomaly_classification: "正在进行异常分类",
    rendering: "正在生成结果图",
    complete: "结果图已生成",
  };
  return labels[stage];
}
