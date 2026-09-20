import { useEffect, useState } from "react";

import type { ApiClient } from "../api/client";
import type { TaskDetail } from "../api/types";


const terminalStatuses = new Set(["succeeded", "partial_failed", "failed"]);

export function pollingDelay(hidden: boolean, consecutiveFailures = 0) {
  const failures = Math.max(0, consecutiveFailures);
  const retryDelay = Math.min(1_000 * (2 ** failures), 10_000);
  return hidden ? 10_000 : retryDelay;
}

export function useActiveTask(
  initialTask: TaskDetail | null,
  client: ApiClient,
) {
  const [task, setTask] = useState<TaskDetail | null>(initialTask);

  useEffect(() => {
    setTask(initialTask);
  }, [initialTask]);

  useEffect(() => {
    if (!task || terminalStatuses.has(task.status)) return;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let stopped = false;
    let controller: AbortController | undefined;
    let consecutiveFailures = 0;

    const schedule = () => {
      timer = setTimeout(async () => {
        controller = new AbortController();
        try {
          const fresh = await client.getTask(task.id, controller.signal);
          if (stopped) return;
          consecutiveFailures = 0;
          setTask(fresh);
          if (!terminalStatuses.has(fresh.status)) schedule();
        } catch (error) {
          if (!stopped && !(error instanceof DOMException && error.name === "AbortError")) {
            consecutiveFailures += 1;
            schedule();
          }
        }
      }, pollingDelay(document.hidden, consecutiveFailures));
    };

    const reschedule = () => {
      if (timer) clearTimeout(timer);
      schedule();
    };

    schedule();
    document.addEventListener("visibilitychange", reschedule);
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
      controller?.abort();
      document.removeEventListener("visibilitychange", reschedule);
    };
  }, [client, task?.id, task?.status]);

  return { task, setTask };
}
