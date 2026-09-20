import { useEffect, useState } from "react";

import type { ApiClient } from "../api/client";
import type { TaskDetail } from "../api/types";


const terminalStatuses = new Set(["succeeded", "partial_failed", "failed"]);

export function pollingDelay(hidden: boolean) {
  return hidden ? 10_000 : 1_000;
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

    const schedule = () => {
      timer = setTimeout(async () => {
        controller = new AbortController();
        try {
          const fresh = await client.getTask(task.id, controller.signal);
          if (stopped) return;
          setTask(fresh);
          if (!terminalStatuses.has(fresh.status)) schedule();
        } catch (error) {
          if (!stopped && !(error instanceof DOMException && error.name === "AbortError")) {
            schedule();
          }
        }
      }, pollingDelay(document.hidden));
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
