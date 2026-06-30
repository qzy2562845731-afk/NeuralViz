import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useGlobalTraining } from '../contexts/GlobalTrainingContext';
import { useToast } from '../contexts/ToastContext';

/* ============================================
   FloatingActions — 全局快捷操作悬浮栏
   - 默认收起为圆形按钮
   - 展开后包含 4 个高频操作：暂停训练、重置训练、查看日志、返回首页
   - 支持拖拽调整位置
   - 任何页面都能一键触发核心操作
   ============================================ */

type ActionType = 'pause' | 'reset' | 'log' | 'home';

const ACTIONS: Array<{
  type: ActionType;
  label: string;
  icon: React.ReactNode;
  color: string;
}> = [
  {
    type: 'pause',
    label: '暂停训练',
    color: 'text-amber-400 hover:bg-amber-400/10',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
        <rect x="6" y="4" width="4" height="16" rx="1" />
        <rect x="14" y="4" width="4" height="16" rx="1" />
      </svg>
    ),
  },
  {
    type: 'reset',
    label: '重置训练',
    color: 'text-red-400 hover:bg-red-400/10',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
        <path d="M3 3v5h5" />
      </svg>
    ),
  },
  {
    type: 'log',
    label: '查看日志',
    color: 'text-emerald-400 hover:bg-emerald-400/10',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
        <polyline points="14 2 14 8 20 8" />
        <line x1="8" y1="13" x2="16" y2="13" />
        <line x1="8" y1="17" x2="16" y2="17" />
      </svg>
    ),
  },
  {
    type: 'home',
    label: '返回首页',
    color: 'text-primary hover:bg-primary/10',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
        <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
        <polyline points="9 22 9 12 15 12 15 22" />
      </svg>
    ),
  },
];

export function FloatingActions() {
  const [expanded, setExpanded] = useState(false);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const dragStartRef = useRef<{ x: number; y: number; posX: number; posY: number } | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const navigate = useNavigate();
  const { isRealTraining, hasActiveTraining, togglePlay, reset } = useGlobalTraining();
  const toast = useToast();

  const isTraining = isRealTraining || hasActiveTraining;

  // 初始化位置（右下角）
  useEffect(() => {
    setPosition({ x: window.innerWidth - 80, y: window.innerHeight - 120 });
  }, []);

  // 拖拽逻辑
  const handleMouseDown = (e: React.MouseEvent) => {
    if (expanded) return;
    setDragging(true);
    dragStartRef.current = {
      x: e.clientX,
      y: e.clientY,
      posX: position.x,
      posY: position.y,
    };
  };

  useEffect(() => {
    if (!dragging) return;
    const handleMove = (e: MouseEvent) => {
      if (!dragStartRef.current) return;
      const dx = e.clientX - dragStartRef.current.x;
      const dy = e.clientY - dragStartRef.current.y;
      const newX = Math.max(10, Math.min(window.innerWidth - 60, dragStartRef.current.posX + dx));
      const newY = Math.max(10, Math.min(window.innerHeight - 60, dragStartRef.current.posY + dy));
      setPosition({ x: newX, y: newY });
    };
    const handleUp = () => {
      setDragging(false);
      dragStartRef.current = null;
    };
    document.addEventListener('mousemove', handleMove);
    document.addEventListener('mouseup', handleUp);
    return () => {
      document.removeEventListener('mousemove', handleMove);
      document.removeEventListener('mouseup', handleUp);
    };
  }, [dragging]);

  // 点击外部收起
  useEffect(() => {
    if (!expanded) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setExpanded(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [expanded]);

  const handleAction = (type: ActionType) => {
    switch (type) {
      case 'pause':
        if (isTraining) {
          togglePlay();
          toast.showInfo('已暂停训练', '可随时继续播放');
        } else {
          toast.showInfo('当前无进行中的训练', '请先到 3D 工作台启动训练');
        }
        break;
      case 'reset':
        if (isTraining) {
          reset();
          toast.showSuccess('已重置训练', '历史数据已清空，可重新启动');
        } else {
          reset();
          toast.showInfo('已清空训练状态');
        }
        break;
      case 'log':
        navigate('/workbench/training-log');
        toast.showInfo('已跳转到训练日志');
        break;
      case 'home':
        navigate('/');
        toast.showInfo('已返回首页');
        break;
    }
    setExpanded(false);
  };

  return (
    <div
      ref={containerRef}
      className="fixed z-[9998]"
      style={{ left: position.x, top: position.y }}
    >
      {/* 展开后的操作列表 */}
      {expanded && (
        <div className="absolute bottom-16 right-0 flex flex-col gap-1.5 rounded-xl border border-white/[0.08] bg-[#0c0e17]/95 p-1.5 shadow-2xl shadow-black/50 backdrop-blur-md"
          style={{ animation: 'fab-expand 180ms ease-out' }}
        >
          {ACTIONS.map(action => (
            <button
              key={action.type}
              onClick={() => handleAction(action.type)}
              className={`flex items-center gap-2 rounded-lg px-3 py-2 text-[11px] font-medium text-foreground/80 transition-all hover:bg-white/[0.04] ${action.color}`}
            >
              {action.icon}
              <span>{action.label}</span>
            </button>
          ))}
        </div>
      )}

      {/* 主按钮 */}
      <button
        onClick={() => setExpanded(prev => !prev)}
        onMouseDown={handleMouseDown}
        title={dragging ? '拖拽调整位置' : '快捷操作'}
        className={`relative flex h-12 w-12 items-center justify-center rounded-full border shadow-xl transition-all ${
          expanded
            ? 'border-primary/40 bg-primary/20 text-primary'
            : isTraining
            ? 'border-emerald-400/40 bg-emerald-400/15 text-emerald-400'
            : 'border-white/[0.1] bg-[#0c0e17]/90 text-foreground/80 hover:border-primary/30 hover:bg-primary/10'
        } ${dragging ? 'cursor-grabbing' : 'cursor-pointer hover:scale-105'}`}
      >
        {/* 训练中脉冲指示 */}
        {isTraining && !expanded && (
          <span className="absolute -right-0.5 -top-0.5 flex h-3 w-3">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex h-3 w-3 rounded-full bg-emerald-400" />
          </span>
        )}
        {expanded ? (
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        ) : (
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
            <circle cx="12" cy="12" r="3" />
          </svg>
        )}
      </button>
      <style>{`
        @keyframes fab-expand {
          from { opacity: 0; transform: translateY(8px) scale(0.95); }
          to { opacity: 1; transform: translateY(0) scale(1); }
        }
      `}</style>
    </div>
  );
}
