import type { TaskSummary } from "../api/types";


interface TaskDetailsProps {
  task: TaskSummary;
}

export function TaskDetails({ task }: TaskDetailsProps) {
  return (
    <div className="task-details" aria-label={`任务 ${task.displayId} 详情`}>
      <dl>
        <div><dt>任务名称</dt><dd>{task.name || "未命名任务"}</dd></div>
        <div><dt>操作员</dt><dd>{task.operator || "-"}</dd></div>
        <div><dt>备注</dt><dd>{task.note || "-"}</dd></div>
      </dl>
    </div>
  );
}
