import { afterEach, describe, expect, it, vi } from "vitest";

import { apiClient, subscribeUnauthorized } from "./client";


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

    await apiClient.createTask([file]);

    const [, options] = fetchMock.mock.calls[0];
    expect(options.body).toBeInstanceOf(FormData);
    expect(options.headers).toBeUndefined();
    const body = options.body as FormData;
    expect(Array.from(body.keys())).toEqual(["files"]);
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

  it("sends same-origin credentials and notifies once on 401", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "authentication required" }), {
        status: 401,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const listener = vi.fn();
    const unsubscribe = subscribeUnauthorized(listener);

    await expect(apiClient.getTask("task-1")).rejects.toEqual(
      expect.objectContaining({
        message: "authentication required",
        status: 401,
      }),
    );

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/tasks/task-1",
      expect.objectContaining({ credentials: "same-origin" }),
    );
    expect(listener).toHaveBeenCalledTimes(1);
    unsubscribe();
  });

  it("preserves username case when logging in", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ id: "user-1", username: "Admin" }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await apiClient.login("Admin", "Password-2026");

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v1/auth/login",
      expect.objectContaining({
        method: "POST",
        credentials: "same-origin",
        body: JSON.stringify({ username: "Admin", password: "Password-2026" }),
      }),
    );
  });

  it("loads, updates, and deletes the selected image feedback", async () => {
    const feedback = {
      imageId: "image-1",
      originalFilename: "terminal.png",
      status: "succeeded",
      detections: [],
      missedRegionCandidates: ["label6"],
      feedback: null,
    };
    const fetchMock = vi.fn().mockImplementation(async () => (
      new Response(JSON.stringify(feedback), {
        status: 200,
        headers: { "content-type": "application/json" },
      })
    ));
    vi.stubGlobal("fetch", fetchMock);

    await apiClient.getImageFeedback("image/1");
    await apiClient.updateImageFeedback("image/1", {
      items: [{ detectionId: "d1", verdict: "NG", color: "R" }],
      missedRegions: ["label6"],
    });
    await apiClient.deleteImageFeedback("image/1");

    expect(fetchMock.mock.calls[0][0]).toBe("/api/v1/images/image%2F1/feedback");
    expect(fetchMock.mock.calls[1]).toEqual([
      "/api/v1/images/image%2F1/feedback",
      expect.objectContaining({
        method: "PUT",
        credentials: "same-origin",
        body: JSON.stringify({
          items: [{ detectionId: "d1", verdict: "NG", color: "R" }],
          missedRegions: ["label6"],
        }),
      }),
    ]);
    expect(fetchMock.mock.calls[2]).toEqual([
      "/api/v1/images/image%2F1/feedback",
      expect.objectContaining({
        method: "DELETE",
        credentials: "same-origin",
      }),
    ]);
  });
});
