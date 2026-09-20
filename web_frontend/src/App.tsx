import { useCallback, useEffect, useState } from "react";

import { Sidebar, type AppPage } from "./components/Sidebar";
import { HistoryPage } from "./pages/HistoryPage";
import { TasksPage } from "./pages/TasksPage";

function routeFromLocation() {
  const match = window.location.pathname.match(/^\/history(?:\/([^/]+))?/);
  return {
    page: (match ? "history" : "tasks") as AppPage,
    taskId: match?.[1] ? decodeURIComponent(match[1]) : undefined,
    imageId: new URLSearchParams(window.location.search).get("image") ?? undefined,
  };
}

export function App() {
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
      <Sidebar active={route.page} onNavigate={navigate} />
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
