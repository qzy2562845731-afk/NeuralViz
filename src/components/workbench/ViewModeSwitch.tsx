import { useWorkbench } from './WorkbenchContext';
import type { ViewMode } from '../cnn3d/types';

/* ============================================
   ViewModeSwitch — 视图模式切换（美化版）
   - 分段胶囊控制
   - 每种模式配独立色带预览
   - 当前模式说明卡片
   ============================================ */

const MODES: Array<{
  id: ViewMode;
  shortName: string;
  fullName: string;
  icon: React.ReactNode;
  gradient: string;
  activeGradient: string;
  ring: string;
  text: string;
  description: string;
}> = [
  {
    id: 'structure',
    shortName: '结构',
    fullName: '结构视图',
    icon: (
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.64 5.64l2.12 2.12M16.24 16.24l2.12 2.12M5.64 18.36l2.12-2.12M16.24 7.76l2.12-2.12" />
        <circle cx="12" cy="12" r="4" />
      </svg>
    ),
    gradient: 'from-slate-500/40 to-slate-300/40',
    activeGradient: 'from-slate-400 to-slate-200',
    ring: 'ring-slate-400/40',
    text: 'text-slate-200',
    description: '按层类型着色，展示网络整体拓扑与层级关系。',
  },
  {
    id: 'activation',
    shortName: '激活',
    fullName: '激活视图',
    icon: (
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z" />
      </svg>
    ),
    gradient: 'from-cyan-400/60 via-white/60 to-red-400/60',
    activeGradient: 'from-cyan-400 via-white to-red-400',
    ring: 'ring-cyan-400/40',
    text: 'text-cyan-200',
    description: '冷暖色映射激活值：青/白表示正激活，红/暗表示负激活。',
  },
  {
    id: 'parameter',
    shortName: '参数',
    fullName: '参数视图',
    icon: (
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <circle cx="12" cy="12" r="10" />
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v4M12 18v4M2 12h4M18 12h4" />
      </svg>
    ),
    gradient: 'from-blue-600/70 via-cyan-400/70 via-yellow-400/70 to-red-500/70',
    activeGradient: 'from-blue-500 via-cyan-400 via-yellow-400 to-red-500',
    ring: 'ring-amber-400/40',
    text: 'text-amber-200',
    description: '热图映射参数规模：蓝→青→黄→红，参数越多亮度越高。',
  },
];

export function ViewModeSwitch() {
  const { viewMode, setViewMode } = useWorkbench();

  return (
    <div className="flex items-center gap-0.5 rounded-md border border-white/[0.08] bg-white/[0.02] p-0.5">
      {MODES.map((mode) => {
        const isActive = viewMode === mode.id;
        return (
          <button
            key={mode.id}
            onClick={() => setViewMode(mode.id)}
            title={mode.description}
            className={`flex items-center gap-1 rounded px-1.5 py-1 text-[10px] font-medium transition-all ${
              isActive
                ? 'bg-white/[0.08] text-foreground'
                : 'text-muted-foreground hover:bg-white/[0.04] hover:text-foreground/80'
            }`}
          >
            <span
              className={`h-2.5 w-0.5 rounded-full bg-gradient-to-b ${
                isActive ? mode.activeGradient : mode.gradient
              }`}
            />
            {mode.icon}
            <span className="hidden xl:inline">{mode.shortName}</span>
          </button>
        );
      })}
    </div>
  );
}

export default ViewModeSwitch;
