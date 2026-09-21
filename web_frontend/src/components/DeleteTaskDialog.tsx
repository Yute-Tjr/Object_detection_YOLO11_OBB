import { Trash, Warning } from "@phosphor-icons/react";
import { useEffect, useRef } from "react";

import type { TaskSummary } from "../api/types";


interface DeleteTaskDialogProps {
  task: TaskSummary;
  deleting: boolean;
  onCancel: () => void;
  onConfirm: () => void;
}


export function DeleteTaskDialog({
  task,
  deleting,
  onCancel,
  onConfirm,
}: DeleteTaskDialogProps) {
  const cancelButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const previouslyFocused = document.activeElement as HTMLElement | null;
    cancelButtonRef.current?.focus();
    return () => previouslyFocused?.focus();
  }, []);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !deleting) onCancel();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [deleting, onCancel]);

  return (
    <div
      className="delete-dialog-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !deleting) onCancel();
      }}
    >
      <section
        className="delete-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="delete-dialog-title"
        aria-describedby="delete-dialog-description"
      >
        <div className="delete-dialog__heading">
          <span className="delete-dialog__icon" aria-hidden="true">
            <Warning size={24} weight="fill" />
          </span>
          <div>
            <h2 id="delete-dialog-title">确认删除任务</h2>
            <p>此操作不可撤销，请确认任务信息。</p>
          </div>
        </div>

        <dl className="delete-dialog__summary">
          <div><dt>任务 ID</dt><dd>{task.displayId}</dd></div>
          <div><dt>任务名称</dt><dd>{task.name || "未命名任务"}</dd></div>
          <div><dt>图像数量</dt><dd>{task.totalImages} 张</dd></div>
        </dl>

        <p id="delete-dialog-description" className="delete-dialog__warning">
          删除后，该任务的 {task.totalImages} 张原图、结果图和检测记录都将被永久删除。
        </p>

        <div className="delete-dialog__actions">
          <button
            ref={cancelButtonRef}
            type="button"
            className="button button--secondary"
            disabled={deleting}
            onClick={onCancel}
          >
            取消删除
          </button>
          <button
            type="button"
            className="button delete-dialog__confirm"
            disabled={deleting}
            onClick={onConfirm}
          >
            <Trash size={18} />
            {deleting ? "正在删除…" : "确认删除"}
          </button>
        </div>
      </section>
    </div>
  );
}
