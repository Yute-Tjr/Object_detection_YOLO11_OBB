import { ClockCounterClockwise, Scan, SignOut, UserCircle } from "@phosphor-icons/react";


export type AppPage = "tasks" | "history";

interface SidebarProps {
  active: AppPage;
  username: string;
  onNavigate: (page: AppPage) => void;
  onLogout: () => Promise<void>;
}


export function Sidebar({ active, username, onNavigate, onLogout }: SidebarProps) {
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
        <div className="sidebar__service">
          <span className="status-dot status-dot--success" />
          系统服务正常
        </div>
        <div className="sidebar__account">
          <UserCircle size={20} aria-hidden="true" />
          <span title={username}>{username}</span>
          <button type="button" onClick={() => void onLogout()} aria-label="退出登录">
            <SignOut size={19} aria-hidden="true" />
          </button>
        </div>
      </div>
    </aside>
  );
}
