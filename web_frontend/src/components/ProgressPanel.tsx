import { Check, CircleNotch } from "@phosphor-icons/react";

import type { ImageStage, TaskDetail } from "../api/types";


export const stageLabel: Record<ImageStage, string> = {
  pending: "等待处理",
  object_detection: "分区域目标检测",
  anomaly_classification: "异常分类",
  rendering: "结果生成",
  complete: "已完成",
};

interface ProgressPanelProps {
  task: TaskDetail;
}


export function ProgressPanel({ task }: ProgressPanelProps) {
  const percent = task.totalImages
    ? Math.round((task.completedImages / task.totalImages) * 100)
    : 0;
  const currentImage = task.images.find((image) => image.status === "running")
    ?? task.images.find((image) => image.status === "queued")
    ?? task.images.at(-1);
  const classificationActive = task.currentStage === "anomaly_classification"
    || task.currentStage === "rendering"
    || task.currentStage === "complete";

  return (
    <section className="progress-panel" aria-live="polite">
      <div className="progress-panel__main">
        <div className="progress-panel__headline">
          <div>
            <span className="sr-only">
              {task.status === "running" ? "正在检测" : "检测进度"} {task.completedImages} / {task.totalImages}
            </span>
            <strong>
              {task.status === "running" ? "正在检测" : "检测进度"}{" "}
              <span>{task.completedImages} / {task.totalImages}</span>
            </strong>
            <span className="task-reference">任务 <span>{task.displayId}</span></span>
          </div>
          <div className="model-reference">模型：<strong>{task.detectorModel}</strong></div>
        </div>
        <div className="progress-track" aria-label={`完成进度 ${percent}%`}>
          <span style={{ width: `${percent}%` }} />
        </div>
        <div className="progress-panel__meta">
          <span>{currentImage?.originalFilename ?? "等待图片"}</span>
          <strong>{percent}%</strong>
        </div>
      </div>
      <div className="stage-flow" aria-label="推理阶段">
        <div className="stage-flow__item is-active">
          <span className="stage-number">
            {classificationActive ? <Check size={16} weight="bold" /> : <CircleNotch size={18} />}
          </span>
          <div><strong>分区域目标检测</strong><small>定位端子结构</small></div>
        </div>
        <span className={classificationActive ? "stage-line is-active" : "stage-line"} />
        <div className={classificationActive ? "stage-flow__item is-active" : "stage-flow__item"}>
          <span className="stage-number">2</span>
          <div><strong>异常分类</strong><small>label3 / label5</small></div>
        </div>
      </div>
    </section>
  );
}
