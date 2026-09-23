import "@testing-library/jest-dom/vitest";

import { act, cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, apiClient } from "../api/client";
import type { AuthUser } from "../api/types";
import { AuthProvider, useAuth, type AuthClient } from "./AuthContext";


afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  window.history.replaceState({}, "", "/");
});


function fakeClient(overrides: Partial<AuthClient> = {}): AuthClient {
  return {
    getCurrentUser: vi.fn().mockResolvedValue({ id: "user-1", username: "Admin" }),
    login: vi.fn().mockResolvedValue({ id: "user-1", username: "Admin" }),
    logout: vi.fn().mockResolvedValue(undefined),
    ...overrides,
  };
}


function Probe() {
  const auth = useAuth();
  return (
    <div>
      <span data-testid="status">{auth.status}</span>
      <span data-testid="username">{auth.user?.username ?? "none"}</span>
      <button onClick={() => void auth.login("Admin", "Password-2026")}>登录</button>
      <button onClick={() => void auth.logout()}>退出</button>
    </div>
  );
}


describe("AuthProvider", () => {
  it("shows loading until the initial session lookup resolves", async () => {
    let resolveUser!: (user: AuthUser) => void;
    const pending = new Promise<AuthUser>((resolve) => { resolveUser = resolve; });
    const client = fakeClient({ getCurrentUser: vi.fn().mockReturnValue(pending) });

    render(<AuthProvider client={client}><Probe /></AuthProvider>);

    expect(screen.getByTestId("status")).toHaveTextContent("loading");
    await act(async () => resolveUser({ id: "user-1", username: "Admin" }));
    expect(screen.getByTestId("status")).toHaveTextContent("authenticated");
    expect(screen.getByTestId("username")).toHaveTextContent("Admin");
  });

  it("treats an initial 401 as an anonymous session", async () => {
    const client = fakeClient({
      getCurrentUser: vi.fn().mockRejectedValue(new ApiError("authentication required", 401)),
    });

    render(<AuthProvider client={client}><Probe /></AuthProvider>);

    await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("anonymous"));
  });

  it("preserves the current route and username case across login", async () => {
    window.history.replaceState({}, "", "/history/task-1?image=image-2");
    const client = fakeClient({
      getCurrentUser: vi.fn().mockRejectedValue(new ApiError("authentication required", 401)),
    });
    const user = userEvent.setup();
    render(<AuthProvider client={client}><Probe /></AuthProvider>);
    await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("anonymous"));

    await user.click(screen.getByRole("button", { name: "登录" }));

    await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("authenticated"));
    expect(client.login).toHaveBeenCalledWith("Admin", "Password-2026");
    expect(window.location.pathname).toBe("/history/task-1");
    expect(window.location.search).toBe("?image=image-2");
  });

  it("moves to anonymous when any real API request returns 401", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: "authentication required" }), {
          status: 401,
          headers: { "content-type": "application/json" },
        }),
      ),
    );
    render(<AuthProvider client={fakeClient()}><Probe /></AuthProvider>);
    await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("authenticated"));

    await act(async () => {
      await expect(apiClient.getTask("task-1")).rejects.toBeInstanceOf(ApiError);
    });

    expect(screen.getByTestId("status")).toHaveTextContent("anonymous");
  });

  it("clears the current user after logout", async () => {
    const client = fakeClient();
    const user = userEvent.setup();
    render(<AuthProvider client={client}><Probe /></AuthProvider>);
    await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("authenticated"));

    await user.click(screen.getByRole("button", { name: "退出" }));

    await waitFor(() => expect(screen.getByTestId("status")).toHaveTextContent("anonymous"));
    expect(client.logout).toHaveBeenCalledTimes(1);
  });
});
