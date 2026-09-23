import { useCallback, useEffect, useState } from "react";

import { AuthProvider, useAuth } from "./auth/AuthContext";
import { Sidebar, type AppPage } from "./components/Sidebar";
import { HistoryPage } from "./pages/HistoryPage";
import { LoginPage } from "./pages/LoginPage";
import { TasksPage } from "./pages/TasksPage";

function routeFromLocation() {
  const match = window.location.pathname.match(/^\/history(?:\/([^/]+))?/);
  return {
    page: (match ? "history" : "tasks") as AppPage,
    taskId: match?.[1] ? decodeURIComponent(match[1]) : undefined,
    imageId: new URLSearchParams(window.location.search).get("image") ?? undefined,
  };
}

function AuthenticatedApplication() {
  const { user, logout } = useAuth();
  const [route, setRoute] = useState(routeFromLocation);
  useEffect(() => {
    const onPopState = () => setRoute(routeFromLocation());
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const navigate = useCallback((page: AppPage) => {
    const path = page === "tasks" ? "/tasks" : "/history";
    window.history.pushState({}, "", path);
    setRoute(routeFromLocation());
  }, []);

  const updateHistoryRoute = useCallback((taskId?: string, imageId?: string) => {
    const path = taskId ? `/history/${encodeURIComponent(taskId)}` : "/history";
    const query = imageId ? `?image=${encodeURIComponent(imageId)}` : "";
    window.history.replaceState({}, "", `${path}${query}`);
    setRoute(routeFromLocation());
  }, []);

  return (
    <main className="app-shell" aria-label="端子分区域检测与异常分类">
      <Sidebar
        active={route.page}
        username={user?.username ?? ""}
        onNavigate={navigate}
        onLogout={logout}
      />
      <section className="app-shell__content">
        {route.page === "tasks" ? <TasksPage /> : (
          <HistoryPage
            initialTaskId={route.taskId}
            initialImageId={route.imageId}
            onRouteChange={updateHistoryRoute}
          />
        )}
      </section>
    </main>
  );
}


function ApplicationGuard() {
  const { status, login } = useAuth();
  if (status === "loading") {
    return (
      <main className="auth-loading" aria-live="polite">
        <span className="auth-loading__indicator" />
        正在验证登录状态…
      </main>
    );
  }
  if (status === "anonymous") return <LoginPage onLogin={login} />;
  return <AuthenticatedApplication />;
}


export function App() {
  return (
    <AuthProvider>
      <ApplicationGuard />
    </AuthProvider>
  );
}
