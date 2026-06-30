import { createContext, useContext, useState, useCallback, useRef, type ReactNode } from 'react';

/* ============================================
   ToastContext — 统一操作反馈规范
   - 所有操作（保存配置、重置训练、切换页面）都有轻量 toast 反馈
   - 支持 success / error / info 三种类型
   - 自动消失，支持手动关闭
   ============================================ */

type ToastType = 'success' | 'error' | 'info';

interface ToastItem {
  id: number;
  type: ToastType;
  message: string;
  description?: string;
}

interface ToastContextValue {
  showSuccess: (message: string, description?: string) => void;
  showError: (message: string, description?: string) => void;
  showInfo: (message: string, description?: string) => void;
}

const ToastCtx = createContext<ToastContextValue | null>(null);

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastCtx);
  if (!ctx) throw new Error('useToast must be used within ToastProvider');
  return ctx;
}

const TOAST_DURATION = 3000;
const MAX_TOASTS = 3;

const TYPE_STYLES: Record<ToastType, { color: string; bg: string; border: string; icon: ReactNode }> = {
  success: {
    color: 'text-emerald-400',
    bg: 'bg-emerald-400/[0.06]',
    border: 'border-emerald-400/30',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
        <path d="M20 6L9 17l-5-5" />
      </svg>
    ),
  },
  error: {
    color: 'text-red-400',
    bg: 'bg-red-400/[0.06]',
    border: 'border-red-400/30',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
        <circle cx="12" cy="12" r="10" />
        <line x1="15" y1="9" x2="9" y2="15" />
        <line x1="9" y1="9" x2="15" y2="15" />
      </svg>
    ),
  },
  info: {
    color: 'text-primary',
    bg: 'bg-primary/[0.06]',
    border: 'border-primary/30',
    icon: (
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="16" x2="12" y2="12" />
        <line x1="12" y1="8" x2="12.01" y2="8" />
      </svg>
    ),
  },
};

export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);
  const idRef = useRef(0);

  const remove = useCallback((id: number) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  const show = useCallback((type: ToastType, message: string, description?: string) => {
    const id = ++idRef.current;
    setToasts(prev => {
      const next = [...prev, { id, type, message, description }];
      // 限制最大显示数量
      if (next.length > MAX_TOASTS) {
        return next.slice(next.length - MAX_TOASTS);
      }
      return next;
    });
    // 自动消失
    window.setTimeout(() => remove(id), TOAST_DURATION);
  }, [remove]);

  const showSuccess = useCallback((m: string, d?: string) => show('success', m, d), [show]);
  const showError = useCallback((m: string, d?: string) => show('error', m, d), [show]);
  const showInfo = useCallback((m: string, d?: string) => show('info', m, d), [show]);

  return (
    <ToastCtx.Provider value={{ showSuccess, showError, showInfo }}>
      {children}
      {/* Toast 容器 - 固定在右上角 */}
      <div className="pointer-events-none fixed right-4 top-4 z-[9999] flex flex-col gap-2">
        {toasts.map(toast => {
          const style = TYPE_STYLES[toast.type];
          return (
            <div
              key={toast.id}
              className={`pointer-events-auto flex min-w-[260px] max-w-[360px] items-start gap-2.5 rounded-lg border ${style.border} ${style.bg} px-3 py-2.5 shadow-lg shadow-black/30 backdrop-blur-md`}
              style={{ animation: 'toast-slide-in 200ms ease-out' }}
            >
              <span className={`mt-0.5 flex-shrink-0 ${style.color}`}>{style.icon}</span>
              <div className="flex-1 min-w-0">
                <p className="text-[12px] font-medium text-foreground">{toast.message}</p>
                {toast.description && (
                  <p className="mt-0.5 text-[10px] text-muted-foreground">{toast.description}</p>
                )}
              </div>
              <button
                onClick={() => remove(toast.id)}
                className="flex-shrink-0 text-muted-foreground/60 transition hover:text-foreground"
                aria-label="关闭"
              >
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>
          );
        })}
      </div>
      <style>{`
        @keyframes toast-slide-in {
          from { opacity: 0; transform: translateX(20px); }
          to { opacity: 1; transform: translateX(0); }
        }
      `}</style>
    </ToastCtx.Provider>
  );
}
