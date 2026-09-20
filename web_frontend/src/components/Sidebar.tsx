import { ClockCounterClockwise, Scan } from "@phosphor-icons/react";


export type AppPage = "tasks" | "history";

interface SidebarProps {
  active: AppPage;
  onNavigate: (page: AppPage) => void;
}


export function Sidebar({ active, onNavigate }: SidebarProps) {
  return (
    <aside className="sidebar">
      <div className="sidebar__brand">
        <span className="sidebar__brand-mark"><Scan size={22} weight="bold" /></span>
        <span>端子智能检测</span>
      </div>
      <nav className="sidebar__nav" aria-label="主导航">
        <button
          className={active === "tasks" ? "sidebar__nav-item is-active" : "sidebar__nav-item"}
          onClick={() => onNavigate("tasks")}
        >
          <Scan size={20} />
          检测任务
        </button>
        <button
          className={active === "history" ? "sidebar__nav-item is-active" : "sidebar__nav-item"}
          onClick={() => onNavigate("history")}
        >
          <ClockCounterClockwise size={20} />
          检测历史
        </button>
      </nav>
      <div className="sidebar__footer">
        <span className="status-dot status-dot--success" />
        系统服务正常
      </div>
    </aside>
  );
}
