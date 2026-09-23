import "@testing-library/jest-dom/vitest";

import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import { LoginPage } from "./LoginPage";


afterEach(cleanup);


describe("LoginPage", () => {
  it("contains only the supported login controls", () => {
    render(<LoginPage onLogin={vi.fn()} />);

    expect(screen.getByRole("textbox", { name: "用户名" })).toBeInTheDocument();
    expect(screen.getByLabelText("密码")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "登录" })).toBeInTheDocument();
    expect(screen.queryByText(/注册/)).not.toBeInTheDocument();
    expect(screen.queryByText(/找回密码/)).not.toBeInTheDocument();
  });

  it("submits the exact username case", async () => {
    const onLogin = vi.fn().mockResolvedValue(undefined);
    const user = userEvent.setup();
    render(<LoginPage onLogin={onLogin} />);

    await user.type(screen.getByRole("textbox", { name: "用户名" }), "Admin");
    await user.type(screen.getByLabelText("密码"), "Password-2026");
    await user.click(screen.getByRole("button", { name: "登录" }));

    expect(onLogin).toHaveBeenCalledWith("Admin", "Password-2026");
  });

  it("keeps entered values and shows a login error", async () => {
    const onLogin = vi.fn().mockRejectedValue(new Error("用户名或密码错误"));
    const user = userEvent.setup();
    render(<LoginPage onLogin={onLogin} />);
    const username = screen.getByRole("textbox", { name: "用户名" });
    const password = screen.getByLabelText("密码");

    await user.type(username, "Admin");
    await user.type(password, "Wrong-password");
    await user.click(screen.getByRole("button", { name: "登录" }));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("用户名或密码错误"));
    expect(username).toHaveValue("Admin");
    expect(password).toHaveValue("Wrong-password");
  });
});
