import type { TaskStatus } from "../api/types";


export type TaskFilter = "ongoing" | "all" | "success" | "failed";

interface TaskFiltersProps {
  value: TaskFilter;
  onChange: (value: TaskFilter) => void;
  counts: Record<TaskFilter, number>;
}

const labels: Array<[TaskFilter, string]> = [
  ["ongoing", "进行中"],
  ["all", "全部"],
  ["success", "成功"],
  ["failed", "失败"],
];

export function belongsToFilter(status: TaskStatus, filter: TaskFilter) {
  if (filter === "all") return true;
  if (filter === "ongoing") return status === "queued" || status === "running";
  if (filter === "success") return status === "succeeded";
  return status === "failed" || status === "partial_failed";
}

export function TaskFilters({ value, onChange, counts }: TaskFiltersProps) {
  return (
    <div className="task-filters" aria-label="任务状态筛选">
      {labels.map(([filter, label]) => (
        <button
          key={filter}
          className={value === filter ? `task-filter task-filter--${filter} is-active` : `task-filter task-filter--${filter}`}
          aria-pressed={value === filter}
          aria-label={`${label} ${counts[filter]}`}
          onClick={() => onChange(filter)}
        >
          <span>{label}</span>
          <strong>{counts[filter]}</strong>
        </button>
      ))}
    </div>
  );
}
