import type {
  CreateTaskMetadata,
  HealthResponse,
  ImageDetail,
  ImagePage,
  ImageSummary,
  ListTaskParams,
  TaskDetail,
  TaskPage,
} from "./types";


const API_ROOT = "/api/v1";


async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, init);
  if (!response.ok) {
    let message = `请求失败（${response.status}）`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) message = payload.detail;
    } catch {
      // Keep the status-based message when the server does not return JSON.
    }
    throw new Error(message);
  }
  return (await response.json()) as T;
}


function queryString(params: Record<string, string | number | undefined>): string {
  const query = new URLSearchParams();
  Object.entries(params).forEach(([key, value]) => {
    if (value !== undefined && value !== "") query.set(key, String(value));
  });
  const serialized = query.toString();
  return serialized ? `?${serialized}` : "";
}


export const apiClient = {
  getHealth(signal?: AbortSignal): Promise<HealthResponse> {
    return request("/health", { signal });
  },

  createTask(
    files: File[],
    metadata: CreateTaskMetadata = {},
    signal?: AbortSignal,
  ): Promise<TaskDetail> {
    const body = new FormData();
    files.forEach((file) => body.append("files", file));
    if (metadata.name) body.append("name", metadata.name);
    if (metadata.note) body.append("note", metadata.note);
    return request("/tasks", { method: "POST", body, signal });
  },

  listTasks(params: ListTaskParams = {}, signal?: AbortSignal): Promise<TaskPage> {
    return request(
      `/tasks${queryString({
        status: params.status,
        query: params.query,
        offset: params.offset,
        limit: params.limit,
      })}`,
      { signal },
    );
  },

  getTask(taskId: string, signal?: AbortSignal): Promise<TaskDetail> {
    return request(`/tasks/${encodeURIComponent(taskId)}`, { signal });
  },

  getTaskImages(
    taskId: string,
    signal?: AbortSignal,
  ): Promise<ImagePage> {
    return request(`/tasks/${encodeURIComponent(taskId)}/images`, { signal });
  },

  getImage(imageId: string, signal?: AbortSignal): Promise<ImageDetail> {
    return request(`/images/${encodeURIComponent(imageId)}`, { signal });
  },

  retryImage(imageId: string, signal?: AbortSignal): Promise<ImageSummary> {
    return request(`/images/${encodeURIComponent(imageId)}/retry`, {
      method: "POST",
      signal,
    });
  },
};
