import type {
  AuthUser,
  HealthResponse,
  FeedbackUpdateRequest,
  ImageDetail,
  ImageFeedbackView,
  ImagePage,
  ImageSummary,
  ListTaskParams,
  TaskDetail,
  TaskPage,
} from "./types";


const API_ROOT = "/api/v1";

type UnauthorizedListener = () => void;
const unauthorizedListeners = new Set<UnauthorizedListener>();


export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}


export function subscribeUnauthorized(listener: UnauthorizedListener): () => void {
  unauthorizedListeners.add(listener);
  return () => unauthorizedListeners.delete(listener);
}


async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, {
    ...init,
    credentials: "same-origin",
  });
  if (!response.ok) {
    let message = `请求失败（${response.status}）`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) message = payload.detail;
    } catch {
      // Keep the status-based message when the server does not return JSON.
    }
    if (response.status === 401) {
      unauthorizedListeners.forEach((listener) => listener());
    }
    throw new ApiError(message, response.status);
  }
  if (response.status === 204) return undefined as T;
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
  getCurrentUser(signal?: AbortSignal): Promise<AuthUser> {
    return request("/auth/me", { signal });
  },

  login(username: string, password: string, signal?: AbortSignal): Promise<AuthUser> {
    return request("/auth/login", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ username, password }),
      signal,
    });
  },

  logout(signal?: AbortSignal): Promise<void> {
    return request("/auth/logout", { method: "POST", signal });
  },

  getHealth(signal?: AbortSignal): Promise<HealthResponse> {
    return request("/health", { signal });
  },

  createTask(files: File[], signal?: AbortSignal): Promise<TaskDetail> {
    const body = new FormData();
    files.forEach((file) => body.append("files", file));
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

  deleteTask(taskId: string, signal?: AbortSignal): Promise<void> {
    return request(`/tasks/${encodeURIComponent(taskId)}`, {
      method: "DELETE",
      signal,
    });
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

  getImageFeedback(imageId: string, signal?: AbortSignal): Promise<ImageFeedbackView> {
    return request(`/images/${encodeURIComponent(imageId)}/feedback`, { signal });
  },

  updateImageFeedback(
    imageId: string,
    payload: FeedbackUpdateRequest,
    signal?: AbortSignal,
  ): Promise<ImageFeedbackView> {
    return request(`/images/${encodeURIComponent(imageId)}/feedback`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
      signal,
    });
  },
};

export type ApiClient = typeof apiClient;
