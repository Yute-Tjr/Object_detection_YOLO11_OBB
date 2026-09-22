import { useEffect, useState } from "react";

import { apiClient, type ApiClient } from "../api/client";
import type { HealthResponse, TaskDetail } from "../api/types";
import { ImageComparison } from "../components/ImageComparison";
import { ProgressPanel } from "../components/ProgressPanel";
import { UploadPanel } from "../components/UploadPanel";
import { pollingDelay as getPollingDelay, useActiveTask } from "../hooks/useActiveTask";


export const pollingDelay = getPollingDelay;

interface TasksPageProps {
  client?: ApiClient;
  initialTask?: TaskDetail | null;
}


export function TasksPage({ client = apiClient, initialTask = null }: TasksPageProps) {
  const [files, setFiles] = useState<File[]>([]);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [loadingHealth, setLoadingHealth] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [createdTask, setCreatedTask] = useState<TaskDetail | null>(initialTask);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const { task, setTask } = useActiveTask(createdTask, client);

  useEffect(() => {
    const controller = new AbortController();
    client.getHealth(controller.signal)
      .then(setHealth)
      .catch((reason) => {
        if (!controller.signal.aborted) {
          setError(reason instanceof Error ? reason.message : "无法读取系统状态");
        }
      })
      .finally(() => setLoadingHealth(false));
    return () => controller.abort();
  }, [client]);

  useEffect(() => {
    if (task && selectedIndex >= task.images.length) setSelectedIndex(0);
  }, [selectedIndex, task]);

  const ready = Boolean(
    health?.apiReady && health.databaseReady && health.workerReady && health.modelsReady,
  );

  const start = async () => {
    setSubmitting(true);
    setError(null);
    try {
      const created = await client.createTask(files);
      setCreatedTask(created);
      setTask(created);
      setFiles([]);
      setSelectedIndex(0);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "任务创建失败");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="tasks-page">
      <header className="page-header">
        <div>
          <p>生产线视觉质检</p>
          <h1>检测任务</h1>
        </div>
        <div className={ready ? "service-pill is-ready" : "service-pill"}>
          <span className="status-dot" />
          {loadingHealth ? "正在检查系统" : ready ? "系统已就绪" : "系统未就绪"}
        </div>
      </header>

      {error && <div className="page-error" role="alert">{error}</div>}
      <UploadPanel
        files={files}
        onFilesChange={setFiles}
        onStart={start}
        health={health}
        loadingHealth={loadingHealth}
        submitting={submitting}
      />
      {task && (
        <>
          <ProgressPanel task={task} />
          <ImageComparison
            images={task.images}
            selectedIndex={selectedIndex}
            onSelectedIndexChange={setSelectedIndex}
          />
        </>
      )}
    </div>
  );
}
