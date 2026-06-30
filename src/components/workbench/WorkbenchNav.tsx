import { Link, useLocation } from 'react-router-dom';
import { useGlobalTraining } from '../../contexts/GlobalTrainingContext';

/* ============================================
   WorkbenchNav — 工作台内二级导航
   图标+文字 + 高亮指示条 + 训练状态入口
   ============================================ */

const NAV_ITEMS = [
  {
    id: 'visualization',
    label: '数据可视化',
    path: '/workbench/visualization',
    icon: (
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
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
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
        <path d="M14 2v6h6M16 13H8M16 17H8M10 9H8" />
      </svg>
    ),
  },
  {
    id: '3d',
    label: '3D 结构',
    path: '/workbench',
    icon: (
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 2l9 4.5v11L12 22l-9-4.5v-11L12 2z" />
        <path d="M12 12l9-4.5M12 12v10M12 12L3 7.5" />
      </svg>
    ),
  },
  {
    id: 'ai-settings',
    label: 'AI 分析设置',
    path: '/workbench/ai-settings',
    icon: (
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="3" />
        <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
      </svg>
    ),
  },
  {
    id: 'experiments',
    label: '我的实验',
    path: '/experiments',
    icon: (
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M9 3H5a2 2 0 0 0-2 2v4m6-6h10a2 2 0 0 1 2 2v4M9 3v18m0 0H5a2 2 0 0 1-2-2v-4m6 6h10a2 2 0 0 0 2-2v-4" />
      </svg>
    ),
  },
  {
    id: 'datasets',
    label: '数据集',
    path: '/datasets',
    icon: (
      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V7" />
        <path d="M21 7l-9 6-9-6" />
        <path d="M3 7l9-4 9 4" />
      </svg>
    ),
  },
];

export function WorkbenchNav() {
  const location = useLocation();
  const { isRealTraining, hasActiveTraining, backendStatus } = useGlobalTraining();

  const isTraining = isRealTraining && backendStatus === 'running';
  const trainingColor = isTraining ? 'text-blue-400' : hasActiveTraining ? 'text-amber-400' : '';

  return (
    <div className="flex items-center gap-1.5 flex-shrink-0">
      <nav className="flex items-center gap-0.5 rounded-lg border border-white/[0.06] bg-white/[0.02] p-0.5 flex-shrink-0">
        {NAV_ITEMS.map((item) => {
          const isActive = location.pathname === item.path;
          return (
            <Link
              key={item.id}
              to={item.path}
              title={item.label}
              className={`relative flex items-center gap-1 rounded-md px-1.5 py-1 text-[10px] font-medium transition-all whitespace-nowrap flex-shrink-0 ${
                isActive
                  ? 'bg-primary/15 text-primary shadow-sm shadow-primary/10'
                  : 'text-muted-foreground hover:bg-white/[0.04] hover:text-foreground/80'
              }`}
            >
              {item.icon}
              <span className="hidden xl:inline">{item.label}</span>
            </Link>
          );
        })}
      </nav>

      {/* 全局训练状态入口（紧凑） */}
      {(isTraining || hasActiveTraining) && (
        <Link
          to="/workbench/training-log"
          className={`flex items-center gap-1 rounded-md border border-white/[0.06] bg-white/[0.02] px-1.5 py-1 transition-all hover:bg-white/[0.04] flex-shrink-0 ${trainingColor}`}
          title="查看训练进度"
        >
          <span className={`h-1.5 w-1.5 rounded-full ${isTraining ? 'animate-pulse bg-blue-400' : 'bg-amber-400'}`} />
        </Link>
      )}
    </div>
  );
}
