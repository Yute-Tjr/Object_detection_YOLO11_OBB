import type { TaskSummary } from "../api/types";


interface TaskDetailsProps {
  task: TaskSummary;
}

export function TaskDetails({ task }: TaskDetailsProps) {
  return (
    <div className="task-details" aria-label={`任务 ${task.displayId} 详情`}>
      <dl>
        <div><dt>任务名称</dt><dd>{task.name || "未命名任务"}</dd></div>
        <div><dt>存储标识</dt><dd className="mono-value">{task.id}</dd></div>
        <div><dt>创建人</dt><dd>操作员</dd></div>
        <div><dt>备注</dt><dd>{task.note || "-"}</dd></div>
      </dl>
    </div>
  );
}
