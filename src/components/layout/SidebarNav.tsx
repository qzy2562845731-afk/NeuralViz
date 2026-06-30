import { useState, useRef, useCallback } from 'react';
import { NavLink } from 'react-router-dom';
import { useWorkbench } from '../workbench/WorkbenchContext';

const SIDEBAR_ITEMS = [
  {
    id: 'home',
    label: '首页',
    path: '/',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M3 9l9-7 9 7v11a2 2 0 01-2 2H5a2 2 0 01-2-2z" />
        <polyline points="9 22 9 12 15 12 15 22" />
      </svg>
    ),
  },
  {
    id: 'workbench',
    label: '3D 结构',
    path: '/workbench',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 2l9 4.5v11L12 22l-9-4.5v-11L12 2z" />
        <path d="M12 12l9-4.5M12 12v10M12 12L3 7.5" />
      </svg>
    ),
  },
  {
    id: 'visualization',
    label: '数据可视化',
    path: '/workbench/visualization',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M3 3v18h18" />
        <path d="M7 16l4-6 4 3 5-8" />
      </svg>
    ),
  },
  {
    id: 'training-log',
    label: '训练日志',
    path: '/workbench/training-log',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
        <path d="M14 2v6h6M16 13H8M16 17H8M10 9H8" />
      </svg>
    ),
  },
  {
    id: 'experiments',
    label: '实验库',
    path: '/experiments',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
      </svg>
    ),
  },
  {
    id: 'datasets',
    label: '数据集',
    path: '/datasets',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <ellipse cx="12" cy="5" rx="9" ry="3" />
        <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
        <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
      </svg>
    ),
  },
  {
    id: 'auto-training',
    label: '自动训练',
    path: '/auto-training',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M5 3l14 9-14 9V3z" />
      </svg>
    ),
  },
  {
    id: 'ai-settings',
    label: 'AI 设置',
    path: '/workbench/ai-settings',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="3" />
        <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.07l1.42-1.42M18.36 5.64l1.42-1.42" />
      </svg>
    ),
  },
];

const EXPANDED_WIDTH = 224;
const COLLAPSED_WIDTH = 56;

export function SidebarNav() {
  const { navSidebarCollapsed, setNavSidebarCollapsed } = useWorkbench();
  const [isHovered, setIsHovered] = useState(false);
  const leaveTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const isExpanded = !navSidebarCollapsed || isHovered;
  const sidebarWidth = isExpanded ? EXPANDED_WIDTH : COLLAPSED_WIDTH;

  const handleMouseEnter = useCallback(() => {
    if (leaveTimerRef.current) {
      clearTimeout(leaveTimerRef.current);
      leaveTimerRef.current = null;
    }
    setIsHovered(true);
  }, []);

  const handleMouseLeave = useCallback(() => {
    leaveTimerRef.current = setTimeout(() => {
      setIsHovered(false);
    }, 150);
  }, []);

  const toggleCollapse = useCallback(() => {
    setNavSidebarCollapsed(!navSidebarCollapsed);
  }, [navSidebarCollapsed, setNavSidebarCollapsed]);

  return (
    <aside
      className="relative flex h-screen flex-shrink-0 flex-col border-r border-white/[0.06] bg-[#0c0e17] transition-[width] duration-300 ease-in-out overflow-hidden"
      style={{ width: `${sidebarWidth}px` }}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      {/* Logo 区域 */}
      <div className="flex items-center gap-2.5 border-b border-white/[0.06] px-3 py-4" style={{ minHeight: '64px' }}>
        <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-400 to-cyan-500 shadow-lg shadow-emerald-500/20">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="text-white">
            <circle cx="12" cy="12" r="3" />
            <path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M5 19l2-2M17 7l2-2" />
          </svg>
        </div>
        <div className={`min-w-0 transition-opacity duration-200 ${isExpanded ? 'opacity-100' : 'opacity-0 w-0 overflow-hidden'}`}>
          <h1 className="text-sm font-bold tracking-tight text-white whitespace-nowrap">NeuralViz</h1>
          <p className="text-[10px] text-muted-foreground whitespace-nowrap">CNN 可视化工作台</p>
        </div>
        {isExpanded && (
          <button
            onClick={toggleCollapse}
            className="ml-auto rounded p-1 text-muted-foreground/50 transition hover:bg-white/[0.06] hover:text-muted-foreground flex-shrink-0"
            title="收起侧边栏"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <polyline points="15 18 9 12 15 6" />
            </svg>
          </button>
        )}
      </div>

      {/* 展开按钮（收起状态下显示） */}
      {!isExpanded && (
        <button
          onClick={toggleCollapse}
          className="absolute top-16 left-1/2 -translate-x-1/2 z-10 flex h-6 w-6 items-center justify-center rounded-md border border-white/[0.08] bg-[#0c0e17] text-muted-foreground/50 transition hover:bg-white/[0.06] hover:text-muted-foreground"
          title="展开侧边栏"
        >
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polyline points="9 18 15 12 9 6" />
          </svg>
        </button>
      )}

      {/* 导航 */}
      <nav className="flex-1 overflow-y-auto overflow-x-hidden py-3 px-2">
        {isExpanded && (
          <div className="mb-2 px-3">
            <span className="text-[9px] font-semibold uppercase tracking-widest text-muted-foreground/50">导航</span>
          </div>
        )}
        {SIDEBAR_ITEMS.map((item) => (
          <NavLink
            key={item.id}
            to={item.path}
            end={item.path === '/workbench' || item.path === '/'}
            className={({ isActive }) =>
              `relative flex items-center gap-3 rounded-lg px-3 py-2.5 text-xs font-medium transition-all mb-0.5 ${
                isActive
                  ? 'bg-emerald-500/10 text-emerald-400 shadow-sm shadow-emerald-500/5'
                  : 'text-muted-foreground hover:bg-white/[0.04] hover:text-foreground/80'
              } ${!isExpanded ? 'justify-center px-0' : ''}`
            }
            title={!isExpanded ? item.label : undefined}
          >
            <span className="flex-shrink-0">{item.icon}</span>
            <span className={`truncate transition-opacity duration-200 ${isExpanded ? 'opacity-100' : 'opacity-0 w-0 overflow-hidden'}`}>{item.label}</span>
          </NavLink>
        ))}
      </nav>

      {/* 底部用户信息 */}
      <div className="border-t border-white/[0.06] p-2">
        <div className={`flex items-center gap-2.5 rounded-lg ${isExpanded ? 'bg-white/[0.02] px-3 py-2.5' : 'justify-center py-2'}`}>
          <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-full bg-gradient-to-br from-violet-500 to-purple-600 text-xs font-bold text-white">
            U
          </div>
          <div className={`min-w-0 flex-1 transition-opacity duration-200 ${isExpanded ? 'opacity-100' : 'opacity-0 w-0 overflow-hidden'}`}>
            <p className="text-xs font-medium text-foreground truncate">研究者</p>
            <p className="text-[10px] text-muted-foreground truncate">本地工作区</p>
          </div>
        </div>
      </div>
    </aside>
  );
}
