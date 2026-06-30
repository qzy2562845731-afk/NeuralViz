import type { PlaybackSpeed } from '../../types/training';
import { DarkSelect } from '../ui/DarkSelect';

/* ============================================
   WorkbenchTopBar — 顶部控制栏（紧凑版）
   一行内包含：Logo · 模型标签 · 播放控制 · Speed · 4 个指标
   高度：约 52px，不做两层分离
   ============================================ */

interface WorkbenchTopBarProps {
  currentModel: string;
  currentStep: number;
  maxStep: number;
  isPlaying: boolean;
  speed: number;
  onTogglePlay: () => void;
  onStepForward: () => void;
  onStepBackward: () => void;
  onReset: () => void;
  onSpeedChange: (speed: PlaybackSpeed) => void;
  onDownload: () => void;
  // 新增：当前训练数据（用于直接在顶栏展示指标）
  currentData?: { trainLoss?: number; valLoss?: number; trainAccuracy?: number; f1Score?: number; learningRate?: number } | null;
}

export function WorkbenchTopBar({
  currentModel,
  currentStep,
  maxStep,
  isPlaying,
  speed,
  onTogglePlay,
  onStepForward,
  onStepBackward,
  onReset,
  onSpeedChange,
  onDownload,
  currentData,
}: WorkbenchTopBarProps) {
  const speeds: PlaybackSpeed[] = [1, 2, 5];

  return (
    <div className="flex items-center justify-between border-b border-white/[0.08] bg-[#0f1119] px-5" style={{ height: '56px', minHeight: '56px' }}>

      {/* ===== 左侧：Logo + 模型 ===== */}
      <div className="flex items-center gap-3">
        {/* Logo */}
        <div className="flex items-center gap-2">
          <div className="flex h-7 w-7 items-center justify-center rounded-md bg-primary/20">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="h-4 w-4 text-primary">
              <circle cx="12" cy="12" r="3" />
              <circle cx="12" cy="12" r="9" strokeDasharray="4 2" />
            </svg>
          </div>
          <span className="text-sm font-semibold tracking-tight text-foreground/90">
            Neural<span className="text-primary">Viz</span>
          </span>
        </div>

        {/* 分隔 */}
        <div className="h-5 w-px bg-white/[0.08]" />

        {/* 模型标签 */}
        <span className="rounded-md border border-white/[0.08] bg-white/[0.02] px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground/80">
          {currentModel}
        </span>
      </div>

      {/* ===== 中间：播放控制 ===== */}
      <div className="flex items-center gap-2">

        {/* Step 显示 */}
        <div className="mr-2 flex items-baseline gap-1">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/60">
            Step
          </span>
          <span className="font-mono text-sm font-bold tabular-nums text-primary">
            {currentStep}
          </span>
          <span className="font-mono text-[11px] tabular-nums text-muted-foreground/40">
            / {maxStep}
          </span>
        </div>

        {/* 后退 */}
        <button
          onClick={onStepBackward}
          className="flex h-8 w-8 items-center justify-center rounded-md border border-white/[0.06] bg-white/[0.02] text-[11px] text-muted-foreground/80 transition hover:border-white/[0.15] hover:bg-white/[0.05] hover:text-foreground"
          title="上一步"
        >
          ◀◀
        </button>

        {/* 播放/暂停（主按钮） */}
        <button
          onClick={onTogglePlay}
          className={`flex h-8 items-center gap-1.5 rounded-md px-3 text-[11px] font-bold shadow-md transition-all ${
            isPlaying
              ? 'bg-green-500/20 text-green-400 shadow-green-500/10 hover:bg-green-500/30'
              : 'bg-primary/25 text-primary shadow-primary/20 hover:bg-primary/35'
          }`}
        >
          {isPlaying ? (
            <>
              <span className="size-1.5 animate-pulse rounded-full bg-green-400" />
              ⏸ Pause
            </>
          ) : (
            <>
              ▶ Play
            </>
          )}
        </button>

        {/* 前进 */}
        <button
          onClick={onStepForward}
          className="flex h-8 w-8 items-center justify-center rounded-md border border-white/[0.06] bg-white/[0.02] text-[11px] text-muted-foreground/80 transition hover:border-white/[0.15] hover:bg-white/[0.05] hover:text-foreground"
          title="下一步"
        >
          ▶▶
        </button>

        {/* 重置 */}
        <button
          onClick={onReset}
          className="flex h-8 w-8 items-center justify-center rounded-md border border-white/[0.06] bg-white/[0.02] text-[11px] text-muted-foreground/70 transition hover:border-white/[0.15] hover:bg-white/[0.05] hover:text-foreground"
          title="重置"
        >
          ⟲
        </button>

        {/* 分隔 */}
        <div className="mx-1 h-5 w-px bg-white/[0.08]" />

        {/* Speed */}
        <div className="flex items-center gap-1.5">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground/55">
            Speed
          </span>
          <DarkSelect
            options={speeds.map(s => ({ value: String(s), label: `${s}x` }))}
            value={String(speed)}
            onChange={(v) => onSpeedChange(Number(v) as PlaybackSpeed)}
            className="w-16"
          />
        </div>
      </div>

      {/* ===== 右侧：4 个指标（直接整合到顶栏，避免两层） ===== */}
      <div className="flex items-center gap-2">

        {/* Loss */}
        <TopBarMetric
          label="Loss"
          value={currentData?.trainLoss !== undefined && currentData?.trainLoss !== null
            ? currentData.trainLoss.toFixed(3)
            : '—'}
          color="red"
        />

        {/* Accuracy */}
        <TopBarMetric
          label="Acc"
          value={currentData?.trainAccuracy !== undefined && currentData?.trainAccuracy !== null
            ? `${(currentData.trainAccuracy * 100).toFixed(1)}%`
            : '—'}
          color="green"
        />

        {/* F1 */}
        <TopBarMetric
          label="F1"
          value={currentData?.f1Score !== undefined && currentData?.f1Score !== null
            ? currentData.f1Score.toFixed(3)
            : '—'}
          color="blue"
        />

        {/* LR */}
        <TopBarMetric
          label="LR"
          value={currentData?.learningRate !== undefined && currentData?.learningRate !== null
            ? (currentData.learningRate ? currentData.learningRate.toFixed(4) : '0.0000')
            : '—'}
          color="orange"
        />

        {/* 分隔 */}
        <div className="mx-1 h-5 w-px bg-white/[0.08]" />

        {/* Download */}
        <button
          onClick={onDownload}
          className="flex h-8 items-center gap-1.5 rounded-md border border-white/[0.06] bg-white/[0.02] px-2.5 text-[11px] font-semibold text-muted-foreground/80 transition hover:border-white/[0.15] hover:bg-white/[0.05] hover:text-foreground"
          title="导出数据"
        >
          <span>↓</span>
          <span>Export</span>
        </button>
      </div>

    </div>
  );
}

/* ---- 顶栏内置小型指标组件 ---- */
type MetricColor = 'red' | 'green' | 'blue' | 'orange';

function TopBarMetric({ label, value, color }: { label: string; value: string; color: MetricColor }) {
  const colorClasses: Record<MetricColor, string> = {
    red:    'text-red-400',
    green:  'text-emerald-400',
    blue:   'text-blue-400',
    orange: 'text-orange-400',
  };

  return (
    <div className="flex items-center gap-1.5 rounded-md border border-white/[0.04] bg-white/[0.015] px-2 py-1">
      <span className="text-[9.5px] font-semibold uppercase tracking-wider text-muted-foreground/55">
        {label}
      </span>
      <span className={`font-mono text-[11.5px] font-bold tabular-nums ${colorClasses[color]}`}>
        {value}
      </span>
    </div>
  );
}
