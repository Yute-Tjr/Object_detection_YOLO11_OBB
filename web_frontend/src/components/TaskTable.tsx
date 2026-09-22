import { Eye, Trash } from "@phosphor-icons/react";
import { useMemo, useState } from "react";

import type { ImageStage, TaskStatus, TaskSummary } from "../api/types";
import { belongsToFilter, TaskFilters, type TaskFilter } from "./TaskFilters";


interface TaskTableProps {
  tasks: TaskSummary[];
  onSelectTask: (task: TaskSummary) => void;
  onDeleteTask?: (task: TaskSummary) => void;
  deletingTaskId?: string | null;
  initialFilter?: TaskFilter;
}

const statusText: Record<TaskStatus, string> = {
  queued: "等待中",
  running: "进行中",
  succeeded: "成功",
  partial_failed: "部分失败",
  failed: "失败",
};

const stageText: Record<ImageStage, string> = {
  pending: "等待处理",
  object_detection: "分区域目标检测",
  anomaly_classification: "异常分类",
  rendering: "结果生成",
  complete: "已完成",
};

function statusTone(status: TaskStatus) {
  if (status === "succeeded") return "success";
  if (status === "failed" || status === "partial_failed") return "failed";
  return "running";
}

function formatDate(value: string) {
  const date = new Date(value);
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric", month: "2-digit", day: "2-digit",
    hour: "2-digit", minute: "2-digit", hour12: false,
    timeZone: "Asia/Shanghai",
  }).format(date).replaceAll("/", "-");
}

export function TaskTable({
  tasks,
  onSelectTask,
  onDeleteTask,
  deletingTaskId,
  initialFilter = "ongoing",
}: TaskTableProps) {
  const [filter, setFilter] = useState<TaskFilter>(initialFilter);
  const counts = useMemo(() => ({
    ongoing: tasks.filter((task) => belongsToFilter(task.status, "ongoing")).length,
    all: tasks.length,
    success: tasks.filter((task) => belongsToFilter(task.status, "success")).length,
    failed: tasks.filter((task) => belongsToFilter(task.status, "failed")).length,
  }), [tasks]);
  const visible = tasks.filter((task) => belongsToFilter(task.status, filter));

  return (
    <section className="task-list-panel" aria-label="检测任务列表">
      <TaskFilters value={filter} onChange={setFilter} counts={counts} />
      <div className="task-table-scroll">
        <table className="task-table">
          <thead>
            <tr>
              <th>#</th>
              <th>任务ID</th>
              <th>图像数量</th>
              <th>完成进度</th>
              <th>当前阶段</th>
              <th>状态</th>
              <th>创建时间</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((task, index) => {
              const percent = task.totalImages
                ? Math.round((task.completedImages / task.totalImages) * 100)
                : 0;
              const tone = statusTone(task.status);
              return (
                  <tr className="task-row" key={task.id}>
                    <td>{index + 1}</td>
                    <td><span className="task-id-text">{task.displayId}</span></td>
                    <td>{task.totalImages}</td>
                    <td>
                      <div className="table-progress-copy">{task.completedImages} / {task.totalImages}</div>
                      <div className="table-progress" aria-label={`完成 ${percent}%`}><span style={{ width: `${percent}%` }} /></div>
                    </td>
                    <td>{stageText[task.currentStage]}</td>
                    <td>
                      <span className={`status-label status-label--${tone}`}>
                        <span className="status-dot" />{statusText[task.status]}
                      </span>
                    </td>
                    <td>{formatDate(task.createdAt)}</td>
                    <td>
                      <div className="task-actions">
                        <button
                          type="button"
                          className="preview-button"
                          aria-label={`预览 ${task.displayId}`}
                          onClick={() => onSelectTask(task)}
                        >
                          <Eye size={17} />
                          预览
                        </button>
                        {onDeleteTask && (
                          <button
                            type="button"
                            className="delete-button"
                            aria-label={`删除 ${task.displayId}`}
                            disabled={
                              task.status === "queued"
                              || task.status === "running"
                              || deletingTaskId === task.id
                            }
                            title={
                              task.status === "queued" || task.status === "running"
                                ? "进行中的任务不能删除"
                                : undefined
                            }
                            onClick={() => onDeleteTask(task)}
                          >
                            <Trash size={17} />
                            {deletingTaskId === task.id ? "删除中" : "删除"}
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
              );
            })}
          </tbody>
        </table>
        {visible.length === 0 && <div className="table-empty">当前筛选条件下没有任务</div>}
      </div>
    </section>
  );
}
