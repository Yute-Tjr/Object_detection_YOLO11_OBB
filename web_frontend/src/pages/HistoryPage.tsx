import { ArrowClockwise, MagnifyingGlass, WarningCircle } from "@phosphor-icons/react";
import { useCallback, useEffect, useState } from "react";

import { apiClient, type ApiClient } from "../api/client";
import type { TaskDetail, TaskSummary } from "../api/types";
import { DeleteTaskDialog } from "../components/DeleteTaskDialog";
import { ImageComparison } from "../components/ImageComparison";
import { TaskTable } from "../components/TaskTable";


interface HistoryPageProps {
  client?: ApiClient;
  initialTaskId?: string;
  initialImageId?: string;
  onRouteChange?: (taskId?: string, imageId?: string) => void;
}

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : "加载失败，请稍后重试";
}

export function HistoryPage({
  client = apiClient,
  initialTaskId,
  initialImageId,
  onRouteChange,
}: HistoryPageProps) {
  const [tasks, setTasks] = useState<TaskSummary[]>([]);
  const [selectedTask, setSelectedTask] = useState<TaskDetail | null>(null);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(true);
  const [retrying, setRetrying] = useState(false);
  const [deletingTaskId, setDeletingTaskId] = useState<string | null>(null);
  const [pendingDeleteTask, setPendingDeleteTask] = useState<TaskSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadTasks = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const page = await client.listTasks({ status: "all", query: query || undefined, limit: 100 });
      setTasks(page.items);
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setLoading(false);
    }
  }, [client, query]);

  const openTask = useCallback(async (taskId: string, imageId?: string) => {
    setError(null);
    const controller = new AbortController();
    try {
      const task = await client.getTask(taskId, controller.signal);
      setSelectedTask(task);
      const index = imageId ? task.images.findIndex((image) => image.id === imageId) : 0;
      setSelectedIndex(index >= 0 ? index : 0);
      const active = task.images[index >= 0 ? index : 0];
      onRouteChange?.(task.id, active?.id);
    } catch (caught) {
      setError(errorMessage(caught));
    }
    return () => controller.abort();
  }, [client, onRouteChange]);

  useEffect(() => { void loadTasks(); }, [loadTasks]);
  useEffect(() => {
    if (initialTaskId) void openTask(initialTaskId, initialImageId);
  }, [initialImageId, initialTaskId, openTask]);

  const selectedImage = selectedTask?.images[selectedIndex];

  const changeImage = (index: number) => {
    setSelectedIndex(index);
    if (selectedTask) onRouteChange?.(selectedTask.id, selectedTask.images[index]?.id);
  };

  const retrySelected = async () => {
    if (!selectedTask || !selectedImage || selectedImage.status !== "failed") return;
    setRetrying(true);
    setError(null);
    try {
      await client.retryImage(selectedImage.id);
      const refreshed = await client.getTask(selectedTask.id);
      setSelectedTask(refreshed);
      setTasks((current) => current.map((task) => task.id === refreshed.id ? refreshed : task));
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setRetrying(false);
    }
  };

  const requestDeleteTask = (task: TaskSummary) => {
    if (task.status !== "queued" && task.status !== "running") {
      setPendingDeleteTask(task);
    }
  };

  const cancelDeleteTask = useCallback(() => {
    setPendingDeleteTask(null);
  }, []);

  const deleteTask = async () => {
    if (!pendingDeleteTask) return;
    const task = pendingDeleteTask;

    setDeletingTaskId(task.id);
    setError(null);
    try {
      await client.deleteTask(task.id);
      setTasks((current) => current.filter((item) => item.id !== task.id));
      if (selectedTask?.id === task.id) {
        setSelectedTask(null);
        setSelectedIndex(0);
        onRouteChange?.();
      }
      setPendingDeleteTask(null);
    } catch (caught) {
      setError(errorMessage(caught));
    } finally {
      setDeletingTaskId(null);
    }
  };

  return (
    <div className="history-page">
      <header className="page-header">
        <div><p>生产数据可追溯</p><h1>检测历史</h1></div>
        <button className="button button--secondary" aria-label="刷新历史记录" onClick={() => void loadTasks()}>
          <ArrowClockwise size={18} />刷新
        </button>
      </header>

      <form className="history-search" onSubmit={(event) => { event.preventDefault(); void loadTasks(); }}>
        <MagnifyingGlass size={18} />
        <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索任务 ID 或图片文件名" aria-label="搜索历史任务" />
        <button className="button button--secondary" type="submit">搜索</button>
      </form>

      {error && (
        <div className="history-error" role="alert">
          <WarningCircle size={19} />
          <span>{error}</span>
          <button className="text-button" onClick={() => void loadTasks()}>重试加载</button>
        </div>
      )}
      {loading && tasks.length === 0 ? (
        <div className="history-loading">正在读取历史任务…</div>
      ) : (
        <TaskTable
          tasks={tasks}
          onSelectTask={(task) => void openTask(task.id)}
          onDeleteTask={requestDeleteTask}
          deletingTaskId={deletingTaskId}
          initialFilter="all"
        />
      )}

      {selectedTask && selectedImage && (
        <section className="history-result">
          <div className="history-result__toolbar">
            <div><strong>{selectedTask.displayId}</strong><span>{selectedImage.originalFilename}</span></div>
            {selectedImage.status === "failed" && (
              <button className="button button--danger" disabled={retrying} onClick={() => void retrySelected()}>
                <ArrowClockwise size={18} />{retrying ? "正在重试" : "重试失败图片"}
              </button>
            )}
          </div>
          {selectedImage.errorMessage && selectedImage.status === "failed" && (
            <p className="image-error-message">{selectedImage.errorMessage}</p>
          )}
          <ImageComparison
            images={selectedTask.images}
            selectedIndex={selectedIndex}
            onSelectedIndexChange={changeImage}
          />
        </section>
      )}

      {pendingDeleteTask && (
        <DeleteTaskDialog
          task={pendingDeleteTask}
          deleting={deletingTaskId === pendingDeleteTask.id}
          onCancel={cancelDeleteTask}
          onConfirm={() => void deleteTask()}
        />
      )}
    </div>
  );
}
