import { Component, type ErrorInfo, type ReactNode } from 'react';

/* ============================================
   ErrorBoundary — 全局错误边界组件
   - 捕获子树渲染过程中的 JavaScript 错误
   - 展示深色科技风格的友好错误提示，避免整树黑屏
   - 支持自定义 fallback 渲染函数
   - 支持错误状态重置（"重试"按钮恢复渲染）
   ============================================ */

interface ErrorBoundaryProps {
  children: ReactNode;
  /** 可选的自定义 fallback 渲染函数，接收错误对象与重置回调 */
  fallback?: (error: Error, reset: () => void) => ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  // 渲染阶段抛出错误时更新状态，触发 fallback 显示
  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  // 提交阶段记录错误信息（便于日志上报 / 调试）
  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    console.error('[ErrorBoundary] 捕获到渲染错误:', error, errorInfo);
  }

  // 重置错误状态，重新渲染子树
  handleReset = (): void => {
    this.setState({ hasError: false, error: null });
  };

  // 刷新当前页面
  handleReload = (): void => {
    window.location.reload();
  };

  render(): ReactNode {
    const { hasError, error } = this.state;
    const { children, fallback } = this.props;

    // 出错时优先使用自定义 fallback
    if (hasError) {
      if (fallback && error) {
        return fallback(error, this.handleReset);
      }
      return <DefaultFallback error={error} onReset={this.handleReset} onReload={this.handleReload} />;
    }

    return children;
  }
}

/* ============================================
   DefaultFallback — 默认错误提示界面
   - 深色科技风格：bg-[#0a0c14]
   - 红色警告图标 + 错误消息 + "刷新页面"按钮
   ============================================ */

interface DefaultFallbackProps {
  error: Error | null;
  onReset: () => void;
  onReload: () => void;
}

function DefaultFallback({ error, onReset, onReload }: DefaultFallbackProps) {
  return (
    <div className="flex min-h-screen w-full flex-col items-center justify-center bg-[#0a0c14] px-6 text-white">
      <div className="w-full max-w-md rounded-2xl border border-red-400/20 bg-[#0f1119] p-8 shadow-2xl shadow-red-500/10">
        {/* 红色警告图标 */}
        <div className="mb-5 flex h-16 w-16 items-center justify-center rounded-full border border-red-400/30 bg-red-400/10">
          <svg
            width="32"
            height="32"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            className="text-red-400"
          >
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
            <line x1="12" y1="9" x2="12" y2="13" />
            <line x1="12" y1="17" x2="12.01" y2="17" />
          </svg>
        </div>

        {/* 标题与错误消息 */}
        <h2 className="text-lg font-bold text-foreground">页面渲染出错</h2>
        <p className="mt-1.5 text-sm text-muted-foreground">
          该模块在渲染过程中发生异常，可尝试重试或刷新页面恢复。
        </p>

        {/* 错误详情（可折叠，便于调试） */}
        {error && (
          <pre className="mt-4 max-h-40 overflow-auto rounded-lg border border-white/[0.06] bg-[#0a0c14] p-3 font-mono text-[11px] leading-relaxed text-red-300/80">
            {error.message}
          </pre>
        )}

        {/* 操作按钮 */}
        <div className="mt-6 flex items-center gap-3">
          <button
            onClick={onReset}
            className="flex flex-1 items-center justify-center gap-1.5 rounded-lg border border-white/[0.08] bg-white/[0.02] px-4 py-2 text-sm font-medium text-foreground/85 transition-all hover:bg-white/[0.05]"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
              <path d="M3 3v5h5" />
            </svg>
            重试
          </button>
          <button
            onClick={onReload}
            className="flex flex-1 items-center justify-center gap-1.5 rounded-lg border border-red-400/30 bg-red-400/10 px-4 py-2 text-sm font-semibold text-red-400 transition-all hover:bg-red-400/15"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8" />
              <path d="M21 3v5h-5" />
            </svg>
            刷新页面
          </button>
        </div>
      </div>
    </div>
  );
}

export default ErrorBoundary;
