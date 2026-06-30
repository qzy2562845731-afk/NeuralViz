import { useState, useMemo, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useGlobalTraining } from '../contexts/GlobalTrainingContext';
import { LossChart } from '../components/LossChart';
import {
  generateLearningInsight,
  summarizeTraining,
  downloadJSON,
  downloadCSV,
} from '../utils/trainingInsights';
import type { TrainingStepData } from '../types/training';

/* ============================================
   TrainingLogPage — 训练日志与学习记录
   - 使用全局训练上下文，支持实时数据
   - 顶部进度总览栏（固定吸顶）
   - 关键节点高亮标记
   - 快捷检索与筛选
   - 支持正序/倒序切换
   ============================================ */

const PHASE_STYLES: Record<string, string> = {
  initial: 'border-blue-400/30 bg-blue-400/10 text-blue-300',
  learning: 'border-primary/30 bg-primary/10 text-primary',
  refining: 'border-amber-400/30 bg-amber-400/10 text-amber-300',
  converged: 'border-emerald-400/30 bg-emerald-400/10 text-emerald-300',
  overfitting: 'border-red-400/30 bg-red-400/10 text-red-300',
};

export default function TrainingLogPage() {
  const navigate = useNavigate();
  const {
    data: trainingData,
    currentStep,
    hasActiveTraining,
    isRealTraining,
    backendStatus,
    displayTotalSteps,
  } = useGlobalTraining();

  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [sortOrder, setSortOrder] = useState<'desc' | 'asc'>('desc');
  const [filterMode, setFilterMode] = useState<'all' | 'key' | 'abnormal'>('all');
  const listRef = useRef<HTMLDivElement>(null);

  const displayedStep = selectedStep ?? currentStep;
  const displayedData = trainingData[displayedStep] ?? null;
  const previousData = displayedStep > 0 ? trainingData[displayedStep - 1] : null;

  const visibleData = useMemo(() => {
    return trainingData.slice(0, currentStep + 1);
  }, [trainingData, currentStep]);

  const summary = useMemo(() => summarizeTraining(trainingData), [trainingData]);
  const insight = useMemo(
    () => (displayedData ? generateLearningInsight(displayedData, previousData) : null),
    [displayedData, previousData]
  );

  // 识别关键节点
  const keySteps = useMemo(() => {
    const steps = new Set<number>();
    trainingData.forEach((d, i) => {
      // 首次突破50%准确率（需要前一步作为对比基准，i===0 不标记）
      if (i > 0 && d.trainAccuracy > 0.5 && trainingData[i - 1].trainAccuracy <= 0.5) steps.add(d.step);
      // 首次突破80%准确率
      if (i > 0 && d.trainAccuracy > 0.8 && trainingData[i - 1].trainAccuracy <= 0.8) steps.add(d.step);
      // 学习率下降点
      if (i > 0 && d.learningRate < trainingData[i - 1].learningRate * 0.9) steps.add(d.step);
      // 最佳验证精度
      if (trainingData.every(other => other.valAccuracy <= d.valAccuracy)) steps.add(d.step);
      // 损失骤变异常点
      if (i > 0 && Math.abs(d.trainLoss - trainingData[i - 1].trainLoss) > 0.3) steps.add(d.step);
    });
    return steps;
  }, [trainingData]);

  // 识别异常step
  const abnormalSteps = useMemo(() => {
    const steps = new Set<number>();
    trainingData.forEach((d, i) => {
      if (i > 0) {
        const lossDelta = Math.abs(d.trainLoss - trainingData[i - 1].trainLoss);
        if (lossDelta > 0.3) steps.add(d.step);
        if (d.trainAccuracy < trainingData[i - 1].trainAccuracy - 0.05) steps.add(d.step);
      }
    });
    return steps;
  }, [trainingData]);

  // 过滤+排序后的日志列表
  const filteredAndSortedData = useMemo(() => {
    let result = [...trainingData];

    // 搜索过滤
    if (searchQuery.trim()) {
      const q = parseInt(searchQuery.trim(), 10);
      if (!isNaN(q)) {
        result = result.filter(d => d.step === q);
      }
    }

    // 筛选模式
    if (filterMode === 'key') {
      result = result.filter(d => keySteps.has(d.step));
    } else if (filterMode === 'abnormal') {
      result = result.filter(d => abnormalSteps.has(d.step));
    }

    // 排序
    result.sort((a, b) => sortOrder === 'desc' ? b.step - a.step : a.step - b.step);

    return result;
  }, [trainingData, searchQuery, filterMode, sortOrder, keySteps, abnormalSteps]);

  // 搜索定位：自动滚动到匹配项
  useEffect(() => {
    if (searchQuery.trim() && filteredAndSortedData.length > 0 && listRef.current) {
      const target = filteredAndSortedData[0];
      const el = listRef.current.querySelector(`[data-step="${target.step}"]`);
      el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [searchQuery, filteredAndSortedData]);

  const handleDownloadJSON = () => downloadJSON(trainingData);
  const handleDownloadCSV = () => downloadCSV(trainingData);

  // 根因修复：终态判断优先使用后端状态，避免训练完成后仍显示"训练中"
  const isTerminal = ['completed', 'failed', 'stopped'].includes(backendStatus);
  const trainingStatus = isTerminal
    ? (backendStatus === 'completed' ? '已完成' : backendStatus === 'failed' ? '训练失败' : '已停止')
    : (isRealTraining && backendStatus === 'running') ? '训练中' : hasActiveTraining ? '已暂停' : '未开始';
  const statusColor = isTerminal
    ? (backendStatus === 'completed' ? 'text-emerald-400' : backendStatus === 'failed' ? 'text-red-400' : 'text-amber-400')
    : (isRealTraining && backendStatus === 'running') ? 'text-emerald-400' : hasActiveTraining ? 'text-amber-400' : 'text-muted-foreground';
  const currentLoss = displayedData?.trainLoss ?? 0;
  const currentAcc = displayedData ? (displayedData.trainAccuracy * 100).toFixed(1) : '0.0';

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-[#0a0c14] text-white">
      {/* 顶部栏 */}
      <header className="flex items-center justify-between border-b border-white/[0.06] bg-[#0c0e17] px-5 py-3">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-primary/60 shadow-lg shadow-primary/20">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-white">
              <path d="M12 3v3M12 18v3M3 12h3M18 12h3M5.64 5.64l2.12 2.12M16.24 16.24l2.12 2.12M5.64 18.36l2.12-2.12M16.24 7.76l2.12-2.12" />
              <circle cx="12" cy="12" r="4" />
            </svg>
          </div>
          <div>
            <h1 className="text-[13px] font-bold tracking-tight">训练日志</h1>
            <p className="text-[10px] text-muted-foreground">训练过程记录 · 模型学习解读 · 数据导出</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleDownloadJSON}
            className="flex h-8 items-center gap-1.5 rounded-lg border border-white/[0.08] bg-white/[0.02] px-3 text-[11px] font-semibold text-foreground/85 transition-all hover:bg-white/[0.05]"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" />
            </svg>
            JSON
          </button>
          <button
            onClick={handleDownloadCSV}
            className="flex h-8 items-center gap-1.5 rounded-lg border border-white/[0.08] bg-white/[0.02] px-3 text-[11px] font-semibold text-foreground/85 transition-all hover:bg-white/[0.05]"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3" />
            </svg>
            CSV
          </button>
        </div>
      </header>

      {/* 进度总览栏（固定吸顶） */}
      <div className="border-b border-white/[0.06] bg-[#0c0e17]/95 px-5 py-3 backdrop-blur-sm">
        <div className="flex items-center gap-6">
          {/* 进度条 */}
          <div className="flex flex-1 items-center gap-3">
            <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">进度</span>
            <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-white/[0.04]">
              <div
                className="h-full rounded-full bg-gradient-to-r from-primary via-purple-400 to-emerald-400 transition-all duration-500"
                style={{ width: `${displayTotalSteps > 0 ? Math.min(100, ((currentStep + 1) / displayTotalSteps) * 100) : 0}%` }}
              />
            </div>
            <span className="font-mono text-[11px] font-bold text-primary">
              {currentStep + 1} / {displayTotalSteps}
            </span>
          </div>

          {/* 训练状态 */}
          <div className="flex items-center gap-1.5">
            <span className={`size-2 rounded-full ${isRealTraining && backendStatus === 'running' ? 'animate-pulse bg-emerald-400' : hasActiveTraining ? 'bg-amber-400' : 'bg-muted-foreground/40'}`} />
            <span className={`text-[11px] font-semibold ${statusColor}`}>{trainingStatus}</span>
          </div>

          {/* 当前Loss */}
          <div className="flex items-center gap-1.5">
            <span className="text-[9px] uppercase text-muted-foreground">Loss</span>
            <span className="font-mono text-[11px] font-bold text-red-400">{currentLoss.toFixed(4)}</span>
          </div>

          {/* 当前准确率 */}
          <div className="flex items-center gap-1.5">
            <span className="text-[9px] uppercase text-muted-foreground">Acc</span>
            <span className="font-mono text-[11px] font-bold text-emerald-400">{currentAcc}%</span>
          </div>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden">
        {/* 左侧：Step 列表 */}
        <aside className="flex w-64 flex-shrink-0 flex-col border-r border-white/[0.06] bg-[#0c0e17]">
          {/* 搜索框 */}
          <div className="border-b border-white/[0.06] p-3">
            <div className="relative flex items-center gap-2 rounded-md border border-white/[0.06] bg-white/[0.02] px-2.5 py-1.5 transition-all focus-within:border-primary/30">
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-muted-foreground">
                <circle cx="11" cy="11" r="7" />
                <path d="m21 21-4.3-4.3" />
              </svg>
              <input
                type="text"
                placeholder="搜索 step 号..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full bg-transparent font-mono text-[11px] text-foreground placeholder:text-muted-foreground/60 focus:outline-none"
              />
              {searchQuery && (
                <button onClick={() => setSearchQuery('')} className="text-muted-foreground hover:text-foreground">
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                    <path d="M18 6 6 18M6 6l12 12" />
                  </svg>
                </button>
              )}
            </div>

            {/* 筛选与排序 */}
            <div className="mt-2 flex items-center gap-1">
              <button
                onClick={() => setFilterMode('all')}
                className={`rounded px-2 py-1 text-[10px] font-medium transition-all ${filterMode === 'all' ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground'}`}
              >
                全部
              </button>
              <button
                onClick={() => setFilterMode('key')}
                className={`rounded px-2 py-1 text-[10px] font-medium transition-all ${filterMode === 'key' ? 'bg-amber-400/15 text-amber-400' : 'text-muted-foreground hover:text-foreground'}`}
              >
                关键节点
              </button>
              <button
                onClick={() => setFilterMode('abnormal')}
                className={`rounded px-2 py-1 text-[10px] font-medium transition-all ${filterMode === 'abnormal' ? 'bg-red-400/15 text-red-400' : 'text-muted-foreground hover:text-foreground'}`}
              >
                异常
              </button>
              <button
                onClick={() => setSortOrder(prev => prev === 'desc' ? 'asc' : 'desc')}
                className="ml-auto rounded px-2 py-1 text-[10px] font-medium text-muted-foreground transition-all hover:text-foreground"
                title={sortOrder === 'desc' ? '当前：倒序（最新在顶部）' : '当前：正序（从1开始）'}
              >
                {sortOrder === 'desc' ? '↓ 倒序' : '↑ 正序'}
              </button>
            </div>
          </div>

          {/* 日志列表 */}
          <div ref={listRef} className="flex-1 overflow-y-auto p-3">
            <h2 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
              训练步骤 ({filteredAndSortedData.length})
            </h2>
            {filteredAndSortedData.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-8 text-center">
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="mb-2 text-muted-foreground/40">
                  <circle cx="11" cy="11" r="7" />
                  <path d="m21 21-4.3-4.3" />
                </svg>
                <p className="text-[10px] text-muted-foreground/60">
                  {trainingData.length === 0 ? '暂无训练数据' : '未找到匹配的步骤'}
                </p>
              </div>
            ) : (
              <div className="space-y-1">
                {filteredAndSortedData.map((step) => {
                  const isActive = displayedStep === step.step;
                  const isKey = keySteps.has(step.step);
                  const isAbnormal = abnormalSteps.has(step.step);
                  return (
                    <button
                      key={step.step}
                      data-step={step.step}
                      onClick={() => setSelectedStep(step.step)}
                      className={`w-full rounded-lg border px-3 py-2 text-left text-[11px] transition-all ${
                        isActive
                          ? 'border-primary/30 bg-primary/10 text-primary'
                          : 'border-transparent bg-white/[0.02] text-muted-foreground hover:bg-white/[0.04] hover:text-foreground/80'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-[13px] font-bold">
                          Step {step.step + 1}
                        </span>
                        <div className="flex items-center gap-1">
                          {isKey && (
                            <span className="rounded bg-amber-400/15 px-1 py-0.5 text-[8px] font-semibold text-amber-400">
                              关键
                            </span>
                          )}
                          {isAbnormal && (
                            <span className="rounded bg-red-400/15 px-1 py-0.5 text-[8px] font-semibold text-red-400">
                              异常
                            </span>
                          )}
                          <span className="text-[10px]">{(step.trainAccuracy * 100).toFixed(0)}%</span>
                        </div>
                      </div>
                      <div className="mt-1 flex items-center gap-2 text-[9px] text-muted-foreground/70">
                        <span>Loss {step.trainLoss.toFixed(3)}</span>
                        <span>·</span>
                        <span>Grad {step.gradientNorm.toFixed(2)}</span>
                      </div>
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        </aside>

        {/* 主内容 */}
        <main className="flex-1 overflow-y-auto p-5">
          {trainingData.length === 0 ? (
            <div className="flex h-full flex-col items-center justify-center">
              <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="mb-4 text-muted-foreground/30">
                <path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z" />
                <path d="M14 2v6h6M16 13H8M16 17H8M10 9H8" />
              </svg>
              <h3 className="text-sm font-semibold text-foreground/70">暂无训练日志</h3>
              <p className="mt-1 text-[11px] text-muted-foreground/60">请先在 3D 工作台启动训练</p>
              <button
                onClick={() => navigate('/workbench')}
                className="mt-4 flex h-8 items-center gap-1.5 rounded-lg border border-primary/30 bg-primary/10 px-4 text-[11px] font-semibold text-primary transition-all hover:bg-primary/15"
              >
                前往工作台
              </button>
            </div>
          ) : (
            <>
              {/* 整体总结 */}
              <section className="mb-5 rounded-xl border border-white/[0.06] bg-[#0c0e17] p-4">
                <div className="mb-3 flex items-center justify-between">
                  <h2 className="text-sm font-bold">训练总结</h2>
                  <span className={`rounded-md border px-2 py-0.5 text-[10px] font-semibold ${PHASE_STYLES[summary.phase]}`}>
                    {summary.title}
                  </span>
                </div>
                <p className="mb-3 text-[11px] text-muted-foreground">{summary.description}</p>
                <ul className="space-y-1.5">
                  {summary.observations.map((obs, idx) => (
                    <li key={idx} className="flex items-start gap-2 text-[11px] text-foreground/80">
                      <span className="mt-0.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-primary" />
                      {obs}
                    </li>
                  ))}
                </ul>
              </section>

              <div className="grid grid-cols-12 gap-4">
                {/* 当前 Step 学习解读 */}
                <div className="col-span-12 lg:col-span-5">
                  {insight && displayedData && (
                    <div className="rounded-xl border border-white/[0.06] bg-[#0c0e17] p-4">
                      <div className="mb-3 flex items-center justify-between">
                        <h2 className="text-sm font-bold">Step {displayedData.step + 1} 学习解读</h2>
                        <span className={`rounded-md border px-2 py-0.5 text-[10px] font-semibold ${PHASE_STYLES[insight.phase]}`}>
                          {insight.title}
                        </span>
                      </div>
                      <p className="mb-3 text-[11px] text-muted-foreground">{insight.description}</p>
                      <ul className="space-y-2">
                        {insight.observations.map((obs, idx) => (
                          <li key={idx} className="flex items-start gap-2 text-[11px] leading-relaxed text-foreground/80">
                            <span className="mt-0.5 h-1.5 w-1.5 flex-shrink-0 rounded-full bg-primary" />
                            {obs}
                          </li>
                        ))}
                      </ul>

                      <div className="mt-4 grid grid-cols-2 gap-2 rounded-lg border border-white/[0.04] bg-white/[0.02] p-2.5">
                        <Metric label="Train Loss" value={displayedData.trainLoss.toFixed(4)} color="text-red-400" />
                        <Metric label="Val Loss" value={displayedData.valLoss.toFixed(4)} color="text-orange-400" />
                        <Metric label="Train Acc" value={`${(displayedData.trainAccuracy * 100).toFixed(1)}%`} color="text-emerald-400" />
                        <Metric label="Val Acc" value={`${(displayedData.valAccuracy * 100).toFixed(1)}%`} color="text-blue-400" />
                        <Metric label="F1 Score" value={`${(displayedData.f1Score * 100).toFixed(1)}%`} color="text-purple-400" />
                        <Metric label="LR" value={displayedData.learningRate ? displayedData.learningRate.toFixed(4) : '0.0000'} color="text-pink-400" />
                      </div>
                    </div>
                  )}
                </div>

                {/* Loss 曲线 */}
                <div className="col-span-12 lg:col-span-7">
                  <LossChart visibleData={visibleData} currentStep={displayedStep} />
                </div>

                {/* 关键学习节点 */}
                <div className="col-span-12">
                  <div className="rounded-xl border border-white/[0.06] bg-[#0c0e17] p-4">
                    <h2 className="mb-3 text-sm font-bold">关键学习节点</h2>
                    <div className="grid grid-cols-1 gap-3 md:grid-cols-2 lg:grid-cols-4">
                      <MilestoneCard
                        title="首次突破 50% 准确率"
                        step={findFirstStep(trainingData, (d) => d.trainAccuracy > 0.5)}
                      />
                      <MilestoneCard
                        title="首次突破 80% 准确率"
                        step={findFirstStep(trainingData, (d) => d.trainAccuracy > 0.8)}
                      />
                      <MilestoneCard
                        title="验证损失最低点"
                        step={findBestStep(trainingData, (d) => -d.valLoss)}
                      />
                      <MilestoneCard
                        title="验证准确率最高点"
                        step={findBestStep(trainingData, (d) => d.valAccuracy)}
                      />
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}

          <div className="h-8" />
        </main>
      </div>
    </div>
  );
}

function Metric({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="text-[9px] uppercase tracking-wider text-muted-foreground/60">{label}</span>
      <span className={`font-mono text-[11px] font-bold ${color}`}>{value}</span>
    </div>
  );
}

function findFirstStep(data: TrainingStepData[], predicate: (d: TrainingStepData) => boolean): TrainingStepData | null {
  for (const d of data) {
    if (predicate(d)) return d;
  }
  return null;
}

function findBestStep(data: TrainingStepData[], scorer: (d: TrainingStepData) => number): TrainingStepData | null {
  if (data.length === 0) return null;
  return data.reduce((best, d) => (scorer(d) > scorer(best) ? d : best));
}

function MilestoneCard({ title, step }: { title: string; step: TrainingStepData | null }) {
  if (!step) {
    return (
      <div className="rounded-lg border border-white/[0.04] bg-white/[0.02] p-3">
        <div className="mb-1 text-[10px] text-muted-foreground">{title}</div>
        <div className="text-[11px] text-foreground/50">未达到</div>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3 transition hover:border-primary/20 hover:bg-white/[0.03]">
      <div className="mb-1 text-[10px] text-muted-foreground">{title}</div>
      <div className="text-[12px] font-bold text-foreground">Step {step.step + 1}</div>
      <div className="mt-1 text-[10px] text-foreground/60">
        Loss {step.trainLoss.toFixed(3)} · Acc {(step.trainAccuracy * 100).toFixed(1)}%
      </div>
    </div>
  );
}
