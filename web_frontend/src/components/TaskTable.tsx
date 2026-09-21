import { CaretDown, CaretRight, Eye } from "@phosphor-icons/react";
import { Fragment, useMemo, useState } from "react";

import type { ImageStage, TaskStatus, TaskSummary } from "../api/types";
import { TaskDetails } from "./TaskDetails";
import { belongsToFilter, TaskFilters, type TaskFilter } from "./TaskFilters";


interface TaskTableProps {
  tasks: TaskSummary[];
  onSelectTask: (task: TaskSummary) => void;
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

export function TaskTable({ tasks, onSelectTask, initialFilter = "ongoing" }: TaskTableProps) {
  const [filter, setFilter] = useState<TaskFilter>(initialFilter);
  const [expandedId, setExpandedId] = useState<string | null>(null);
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
              <th aria-label="展开" />
              <th>#</th>
              <th>任务ID</th>
              <th>图像数量</th>
              <th>完成进度</th>
              <th>模型</th>
              <th>当前阶段</th>
              <th>状态</th>
              <th>创建时间</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((task, index) => {
              const expanded = expandedId === task.id;
              const percent = task.totalImages
                ? Math.round((task.completedImages / task.totalImages) * 100)
                : 0;
              const tone = statusTone(task.status);
              return (
                <Fragment key={task.id}>
                  <tr className={expanded ? "task-row is-expanded" : "task-row"}>
                    <td>
                      <button
                        className="icon-button"
                        aria-label={`${expanded ? "收起" : "展开"}任务 ${task.displayId}`}
                        onClick={() => setExpandedId(expanded ? null : task.id)}
                      >
                        {expanded ? <CaretDown size={17} /> : <CaretRight size={17} />}
                      </button>
                    </td>
                    <td>{index + 1}</td>
                    <td><span className="task-id-text">{task.displayId}</span></td>
                    <td>{task.totalImages}</td>
                    <td>
                      <div className="table-progress-copy">{task.completedImages} / {task.totalImages}</div>
                      <div className="table-progress" aria-label={`完成 ${percent}%`}><span style={{ width: `${percent}%` }} /></div>
                    </td>
                    <td>{task.detectorModel}</td>
                    <td>{stageText[task.currentStage]}</td>
                    <td>
                      <span className={`status-label status-label--${tone}`}>
                        <span className="status-dot" />{statusText[task.status]}
                      </span>
                    </td>
                    <td>{formatDate(task.createdAt)}</td>
                    <td>
                      <button
                        type="button"
                        className="preview-button"
                        aria-label={`预览 ${task.displayId}`}
                        onClick={() => onSelectTask(task)}
                      >
                        <Eye size={17} />
                        预览
                      </button>
                    </td>
                  </tr>
                  {expanded && (
                    <tr className="task-details-row">
                      <td colSpan={10}><TaskDetails task={task} /></td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
        {visible.length === 0 && <div className="table-empty">当前筛选条件下没有任务</div>}
      </div>
    </section>
  );
}
