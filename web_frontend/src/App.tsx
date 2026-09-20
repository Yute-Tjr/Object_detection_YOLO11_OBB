import { useState } from "react";

import { Sidebar, type AppPage } from "./components/Sidebar";
import { TasksPage } from "./pages/TasksPage";


export function App() {
  const [page, setPage] = useState<AppPage>("tasks");
  return (
    <main className="app-shell" aria-label="端子分区域检测与异常分类">
      <Sidebar active={page} onNavigate={setPage} />
      <section className="app-shell__content">
        {page === "tasks" ? <TasksPage /> : (
          <div className="empty-page"><h1>检测历史</h1><p>历史任务视图将在此显示。</p></div>
        )}
      </section>
    </main>
  );
}
