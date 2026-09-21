import { afterEach, describe, expect, it, vi } from "vitest";

import { apiClient } from "./client";


afterEach(() => {
  vi.unstubAllGlobals();
});


describe("apiClient", () => {
  it("creates multipart tasks without setting a content-type boundary", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ id: "task-1" }), {
        status: 202,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const file = new File(["image"], "terminal.png", { type: "image/png" });

    await apiClient.createTask(
      [file],
      { operator: "张三", name: "早班", note: "首件" },
    );

    const [, options] = fetchMock.mock.calls[0];
    expect(options.body).toBeInstanceOf(FormData);
    expect(options.headers).toBeUndefined();
    const body = options.body as FormData;
    expect(body.get("operator")).toBe("张三");
    expect(body.get("name")).toBe("早班");
    expect(body.get("note")).toBe("首件");
  });

  it("surfaces non-2xx JSON error messages", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "模型尚未就绪" }), {
          status: 503,
          headers: { "content-type": "application/json" },
        }),
      ),
    );

    await expect(apiClient.getHealth()).rejects.toThrow("模型尚未就绪");
  });

  it("preserves the failed task filter", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ items: [], total: 0, offset: 0, limit: 20 }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await apiClient.listTasks({ status: "failed" });

    expect(fetchMock.mock.calls[0][0]).toContain("status=failed");
  });

  it("passes abort signals to polling requests", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ id: "task-1" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const controller = new AbortController();

    await apiClient.getTask("task-1", controller.signal);

    expect(fetchMock.mock.calls[0][1].signal).toBe(controller.signal);
  });

  it("deletes a task and accepts an empty 204 response", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(null, { status: 204 }));
    vi.stubGlobal("fetch", fetchMock);

    await expect(apiClient.deleteTask("task/1")).resolves.toBeUndefined();

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/tasks/task%2F1",
      expect.objectContaining({ method: "DELETE" }),
    );
  });
});
