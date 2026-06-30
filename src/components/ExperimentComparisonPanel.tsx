import { useState, useEffect, useRef } from 'react';
import apiService, { type TrainingMetricItem } from '../services/api';

/* ============================================
   ExperimentComparisonPanel — 实验对比面板
   - 可视化展示不同实验的性能指标对比
   - 支持超参数、训练曲线、模型结构的对比
   - 支持实验重命名（2-50字符，中英文数字）
   ============================================ */

interface ExperimentSummary {
  experiment_id: string;
  name: string;
  config_name: string;
  status: string;
  channels: number[];
  attention: string;
  use_bn: boolean;
  use_dropout: boolean;
  use_residual: boolean;
  learning_rate: number;
  best_accuracy: number | null;
  final_loss: number | null;
  total_params: number | null;
  total_epochs: number | null;
}

interface MetricData {
  experiment_id: string;
  metrics: TrainingMetricItem[];
}

interface ComparisonPanelProps {
  isOpen: boolean;
  onClose: () => void;
  ablationGroup?: string;
}

// 颜色方案
const COLORS = [
  '#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
  '#ec4899', '#06b6d4', '#f97316', '#6366f1', '#14b8a6',
];

export function ExperimentComparisonPanel({ isOpen, onClose, ablationGroup }: ComparisonPanelProps) {
  const [experiments, setExperiments] = useState<ExperimentSummary[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [metricsData, setMetricsData] = useState<Map<string, MetricData>>(new Map());
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState<'overview' | 'curves' | 'config'>('overview');
  const [curveMetric, setCurveMetric] = useState<'val_accuracy' | 'val_loss' | 'loss'>('val_accuracy');
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameInput, setRenameInput] = useState('');
  const [renameError, setRenameError] = useState<string | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // 加载消融实验结果
  useEffect(() => {
    if (!isOpen || !ablationGroup) return;
    loadResults();
  }, [isOpen, ablationGroup]);

  // 自动轮询：有运行中的实验时，每5秒刷新状态，确保与实验库状态一致
  useEffect(() => {
    if (!isOpen || !ablationGroup) return;
    const hasRunning = experiments.some(e => e.status === 'running');
    if (hasRunning) {
      pollTimerRef.current = setInterval(() => {
        loadResults();
      }, 5000);
    }
    return () => {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [isOpen, ablationGroup, experiments]);

  const loadResults = async () => {
    if (!ablationGroup) return;
    setLoading(true);
    try {
      const resp = await apiService.getAblationResults(ablationGroup);
      if (resp?.code === 200 && resp.data?.results) {
        const results = resp.data.results as unknown as ExperimentSummary[];
        setExperiments(results);
        // 默认选中所有已完成的实验
        const completed = results.filter(e => e.status === 'completed');
        setSelectedIds(prev => {
          // 保留用户已选中的ID，同时自动选中新完成的实验
          const next = new Set(prev);
          for (const c of completed) {
            next.add(c.experiment_id);
          }
          return next;
        });
      }
    } catch (err) {
      console.error('加载对比实验结果失败:', err);
    } finally {
      setLoading(false);
    }
  };

  // 加载选中实验的指标数据
  const loadMetrics = async () => {
    const newMetrics = new Map(metricsData);
    for (const id of selectedIds) {
      if (!newMetrics.has(id)) {
        try {
          const resp = await apiService.getTrainingMetrics(id);
          if (resp?.code === 200 && resp.data?.metrics) {
            newMetrics.set(id, { experiment_id: id, metrics: resp.data.metrics });
          }
        } catch (err) {
          console.error(`加载实验 ${id} 指标失败:`, err);
        }
      }
    }
    setMetricsData(newMetrics);
  };

  useEffect(() => {
    if (activeTab === 'curves' && selectedIds.size > 0) {
      loadMetrics();
    }
  }, [activeTab, selectedIds]);

  const toggleSelect = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const toggleAll = () => {
    if (selectedIds.size === experiments.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(experiments.map(e => e.experiment_id)));
    }
  };

  const handleStartRename = (id: string, currentName: string) => {
    setRenamingId(id);
    setRenameInput(currentName);
    setRenameError(null);
  };

  const handleCancelRename = () => {
    setRenamingId(null);
    setRenameInput('');
    setRenameError(null);
  };

  const handleConfirmRename = async (id: string) => {
    const name = renameInput.trim();
    if (name.length < 2 || name.length > 50) {
      setRenameError('名称长度需在2-50字符之间');
      return;
    }
    if (!/^[\u4e00-\u9fa5a-zA-Z0-9_\- ]+$/.test(name)) {
      setRenameError('仅支持中文、英文、数字、下划线、连字符和空格');
      return;
    }
    try {
      await apiService.renameExperiment(id, name);
      setExperiments(prev => prev.map(e => e.experiment_id === id ? { ...e, name, config_name: name } : e));
      setRenamingId(null);
      setRenameInput('');
      setRenameError(null);
    } catch (err: any) {
      setRenameError(err.message || '重命名失败');
    }
  };

  const getStatusBadge = (status: string) => {
    const map: Record<string, { color: string; label: string }> = {
      completed: { color: 'bg-green-500/20 text-green-400', label: '已完成' },
      running: { color: 'bg-blue-500/20 text-blue-400', label: '训练中' },
      failed: { color: 'bg-red-500/20 text-red-400', label: '失败' },
      stopped: { color: 'bg-yellow-500/20 text-yellow-400', label: '已停止' },
      paused: { color: 'bg-amber-500/20 text-amber-400', label: '已暂停' },
      draft: { color: 'bg-gray-500/20 text-gray-400', label: '草稿' },
    };
    const s = map[status] || { color: 'bg-gray-500/20 text-gray-400', label: status };
    return <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${s.color}`}>{s.label}</span>;
  };

  const formatPct = (v: number | null) => v != null ? `${(v * 100).toFixed(2)}%` : '-';
  const formatLoss = (v: number | null) => v != null ? v.toFixed(4) : '-';
  const formatParams = (v: number | null) => {
    if (v == null) return '-';
    if (v >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
    if (v >= 1e3) return `${(v / 1e3).toFixed(1)}K`;
    return String(v);
  };

  // 计算曲线图的最大值范围
  const getCurveRange = () => {
    let maxVal = 0, minVal = Infinity;
    for (const id of selectedIds) {
      const data = metricsData.get(id);
      if (!data) continue;
      for (const m of data.metrics) {
        const val = m[curveMetric];
        if (val > maxVal) maxVal = val;
        if (val < minVal) minVal = val;
      }
    }
    return { min: minVal === Infinity ? 0 : minVal, max: maxVal === 0 ? 1 : maxVal };
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-5xl max-h-[85vh] overflow-hidden rounded-2xl border border-white/[0.08] bg-[#0f1119] shadow-2xl flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-white/[0.06] px-6 py-4">
          <div>
            <h2 className="text-base font-bold">实验对比</h2>
            <p className="mt-0.5 text-xs text-muted-foreground">
              {ablationGroup ? `消融实验组: ${ablationGroup}` : '对比不同实验的性能指标'}
              {experiments.length > 0 && (
                <span className="ml-2">
                  {experiments.filter(e => e.status === 'completed').length}/{experiments.length} 已完成
                  {experiments.some(e => e.status === 'running') && (
                    <span className="ml-1.5 inline-flex items-center gap-1 rounded bg-blue-500/15 px-1.5 py-0.5 text-[10px] text-blue-400">
                      <span className="h-1.5 w-1.5 rounded-full bg-blue-400 animate-pulse" />
                      训练中
                    </span>
                  )}
                </span>
              )}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={loadResults}
              disabled={loading}
              className="flex h-8 w-8 items-center justify-center rounded-md border border-white/[0.08] bg-white/[0.02] text-muted-foreground transition-all hover:text-foreground disabled:opacity-50"
              title="刷新"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M1 4v6h6M23 20v-6h-6" />
                <path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15" />
              </svg>
            </button>
            <button
              onClick={onClose}
              className="flex h-8 w-8 items-center justify-center rounded-md border border-white/[0.08] bg-white/[0.02] text-muted-foreground transition-all hover:text-foreground"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 6 6 18M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-white/[0.06] px-6">
          {(['overview', 'curves', 'config'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 text-xs font-medium transition-all border-b-2 ${
                activeTab === tab
                  ? 'border-primary text-primary'
                  : 'border-transparent text-muted-foreground hover:text-foreground'
              }`}
            >
              {{ overview: '概览对比', curves: '训练曲线', config: '配置对比' }[tab]}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {loading ? (
            <div className="flex items-center justify-center py-20">
              <div className="h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
            </div>
          ) : experiments.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-20 text-muted-foreground">
              <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="mb-4 opacity-30">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="16" y1="13" x2="8" y2="13" />
                <line x1="16" y1="17" x2="8" y2="17" />
              </svg>
              <h3 className="text-sm font-semibold text-foreground/70">暂无对比数据</h3>
              <p className="mt-2 text-xs text-muted-foreground/60 max-w-sm text-center">
                请先在「工作台」页面运行消融实验或对比实验
              </p>
              <div className="mt-4 flex flex-col gap-1.5 text-[10px] text-muted-foreground/40">
                <div className="flex items-center gap-1.5">
                  <span className="flex h-4 w-4 items-center justify-center rounded-full bg-primary/15 text-primary text-[9px] font-bold">1</span>
                  在工作台选择数据集
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="flex h-4 w-4 items-center justify-center rounded-full bg-primary/15 text-primary text-[9px] font-bold">2</span>
                  点击「对比实验 / 消融」中的预设方案
                </div>
                <div className="flex items-center gap-1.5">
                  <span className="flex h-4 w-4 items-center justify-center rounded-full bg-primary/15 text-primary text-[9px] font-bold">3</span>
                  等待训练完成后，回到此页面查看对比结果
                </div>
              </div>
            </div>
          ) : (
            <>
              {activeTab === 'overview' && (
                <OverviewTab
                  experiments={experiments}
                  selectedIds={selectedIds}
                  onToggle={toggleSelect}
                  onToggleAll={toggleAll}
                  formatPct={formatPct}
                  formatLoss={formatLoss}
                  formatParams={formatParams}
                  getStatusBadge={getStatusBadge}
                  renamingId={renamingId}
                  renameInput={renameInput}
                  renameError={renameError}
                  onStartRename={handleStartRename}
                  onCancelRename={handleCancelRename}
                  onConfirmRename={handleConfirmRename}
                  onRenameInputChange={setRenameInput}
                />
              )}

              {activeTab === 'curves' && (
                <CurvesTab
                  experiments={experiments}
                  selectedIds={selectedIds}
                  metricsData={metricsData}
                  curveMetric={curveMetric}
                  onMetricChange={setCurveMetric}
                  colors={COLORS}
                  getCurveRange={getCurveRange}
                  formatPct={formatPct}
                  formatLoss={formatLoss}
                />
              )}

              {activeTab === 'config' && (
                <ConfigTab
                  experiments={experiments}
                  selectedIds={selectedIds}
                  colors={COLORS}
                />
              )}
            </>
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-white/[0.06] px-6 py-3">
          <span className="text-xs text-muted-foreground">
            共 {experiments.length} 个实验，已选 {selectedIds.size} 个
          </span>
          <button
            onClick={onClose}
            className="rounded-md border border-white/[0.08] bg-white/[0.02] px-4 py-1.5 text-xs text-muted-foreground transition-all hover:bg-white/[0.05] hover:text-foreground"
          >
            关闭
          </button>
        </div>
      </div>
    </div>
  );
}

/* ====== 概览对比 Tab ====== */
function OverviewTab({
  experiments, selectedIds, onToggle, onToggleAll,
  formatPct, formatLoss, formatParams, getStatusBadge,
  renamingId, renameInput, renameError,
  onStartRename, onCancelRename, onConfirmRename, onRenameInputChange,
}: {
  experiments: ExperimentSummary[];
  selectedIds: Set<string>;
  onToggle: (id: string) => void;
  onToggleAll: () => void;
  formatPct: (v: number | null) => string;
  formatLoss: (v: number | null) => string;
  formatParams: (v: number | null) => string;
  getStatusBadge: (s: string) => React.ReactNode;
  renamingId: string | null;
  renameInput: string;
  renameError: string | null;
  onStartRename: (id: string, currentName: string) => void;
  onCancelRename: () => void;
  onConfirmRename: (id: string) => void;
  onRenameInputChange: (v: string) => void;
}) {
  // 排序：按最佳准确率降序
  const sorted = [...experiments].sort((a, b) => (b.best_accuracy || 0) - (a.best_accuracy || 0));
  const bestAcc = sorted[0]?.best_accuracy || 0;

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-white/[0.06] text-left text-muted-foreground">
            <th className="pb-2 pr-2 w-8">
              <input
                type="checkbox"
                checked={selectedIds.size === experiments.length && experiments.length > 0}
                onChange={onToggleAll}
                className="h-3 w-3 rounded border-white/[0.15] bg-white/[0.04] accent-primary"
              />
            </th>
            <th className="pb-2 pr-4 font-medium">实验名称</th>
            <th className="pb-2 pr-4 font-medium">状态</th>
            <th className="pb-2 pr-4 font-medium text-right">最佳准确率</th>
            <th className="pb-2 pr-4 font-medium text-right">最终损失</th>
            <th className="pb-2 pr-4 font-medium text-right">参数量</th>
            <th className="pb-2 pr-4 font-medium text-right">总Epoch</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((exp, _i) => {
            const isBest = exp.best_accuracy === bestAcc && bestAcc > 0;
            return (
              <tr
                key={exp.experiment_id}
                className={`border-b border-white/[0.03] transition-colors hover:bg-white/[0.02] ${
                  selectedIds.has(exp.experiment_id) ? 'bg-white/[0.03]' : ''
                }`}
              >
                <td className="py-2 pr-2">
                  <input
                    type="checkbox"
                    checked={selectedIds.has(exp.experiment_id)}
                    onChange={() => onToggle(exp.experiment_id)}
                    className="h-3 w-3 rounded border-white/[0.15] bg-white/[0.04] accent-primary"
                  />
                </td>
                <td className="py-2 pr-4">
                  {renamingId === exp.experiment_id ? (
                    <div className="flex flex-col gap-1">
                      <div className="flex items-center gap-1">
                        <input
                          type="text"
                          value={renameInput}
                          onChange={(e) => onRenameInputChange(e.target.value)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') onConfirmRename(exp.experiment_id);
                            if (e.key === 'Escape') onCancelRename();
                          }}
                          className="w-36 rounded border border-primary/40 bg-white/[0.04] px-1.5 py-0.5 text-xs text-foreground focus:outline-none focus:border-primary"
                          autoFocus
                        />
                        <button
                          onClick={() => onConfirmRename(exp.experiment_id)}
                          className="flex h-5 w-5 items-center justify-center rounded bg-primary/20 text-primary hover:bg-primary/30"
                          title="确认"
                        >
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                            <polyline points="20 6 9 17 4 12" />
                          </svg>
                        </button>
                        <button
                          onClick={onCancelRename}
                          className="flex h-5 w-5 items-center justify-center rounded bg-white/[0.06] text-muted-foreground hover:bg-white/[0.12]"
                          title="取消"
                        >
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M18 6 6 18M6 6l12 12" />
                          </svg>
                        </button>
                      </div>
                      {renameError && (
                        <span className="text-[9px] text-red-400">{renameError}</span>
                      )}
                    </div>
                  ) : (
                    <div className="flex items-center gap-1.5 group/rename">
                      <span className="font-medium text-foreground/90">{exp.config_name || exp.name}</span>
                      {isBest && (
                        <span className="rounded bg-amber-500/20 px-1 py-0.5 text-[9px] text-amber-400">最佳</span>
                      )}
                      <button
                        onClick={(e) => { e.stopPropagation(); onStartRename(exp.experiment_id, exp.config_name || exp.name); }}
                        className="opacity-0 group-hover/rename:opacity-100 flex h-5 w-5 items-center justify-center rounded text-muted-foreground/50 hover:bg-white/[0.08] hover:text-muted-foreground transition-all"
                        title="重命名"
                      >
                        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                          <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                        </svg>
                      </button>
                    </div>
                  )}
                </td>
                <td className="py-2 pr-4">{getStatusBadge(exp.status)}</td>
                <td className={`py-2 pr-4 text-right font-mono ${isBest ? 'text-amber-400 font-bold' : 'text-foreground/80'}`}>
                  {formatPct(exp.best_accuracy)}
                </td>
                <td className="py-2 pr-4 text-right font-mono text-foreground/70">{formatLoss(exp.final_loss)}</td>
                <td className="py-2 pr-4 text-right font-mono text-foreground/50">{formatParams(exp.total_params)}</td>
                <td className="py-2 pr-4 text-right font-mono text-foreground/50">{exp.total_epochs ?? '-'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ====== 训练曲线 Tab ====== */
function CurvesTab({
  experiments, selectedIds, metricsData, curveMetric, onMetricChange, colors, getCurveRange, formatPct, formatLoss,
}: {
  experiments: ExperimentSummary[];
  selectedIds: Set<string>;
  metricsData: Map<string, MetricData>;
  curveMetric: 'val_accuracy' | 'val_loss' | 'loss';
  onMetricChange: (m: 'val_accuracy' | 'val_loss' | 'loss') => void;
  colors: string[];
  getCurveRange: () => { min: number; max: number };
  formatPct: (v: number | null) => string;
  formatLoss: (v: number | null) => string;
}) {
  const { min, max } = getCurveRange();
  const range = max - min || 1;
  const selectedExps = experiments.filter(e => selectedIds.has(e.experiment_id));

  const metricLabels: Record<string, string> = {
    val_accuracy: '验证准确率',
    val_loss: '验证损失',
    loss: '训练损失',
  };

  const formatVal = (v: number) => {
    if (curveMetric === 'val_accuracy') return formatPct(v);
    return formatLoss(v);
  };

  return (
    <div className="space-y-4">
      {/* 指标选择器 */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground">指标:</span>
        {(['val_accuracy', 'val_loss', 'loss'] as const).map(m => (
          <button
            key={m}
            onClick={() => onMetricChange(m)}
            className={`rounded px-2.5 py-1 text-xs transition-all ${
              curveMetric === m
                ? 'bg-primary/20 text-primary'
                : 'bg-white/[0.04] text-muted-foreground hover:bg-white/[0.08]'
            }`}
          >
            {metricLabels[m]}
          </button>
        ))}
      </div>

      {/* 简易曲线图 */}
      {selectedExps.length === 0 ? (
        <div className="flex items-center justify-center py-10 text-xs text-muted-foreground">
          请在上方选择要对比的实验
        </div>
      ) : (
        <div className="space-y-3">
          {selectedExps.map((exp, i) => {
            const data = metricsData.get(exp.experiment_id);
            if (!data || data.metrics.length === 0) {
              return (
                <div key={exp.experiment_id} className="flex items-center gap-2 text-xs text-muted-foreground">
                  <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: colors[i % colors.length] }} />
                  <span>{exp.config_name || exp.name}</span>
                  <span className="text-[10px]">- 无数据</span>
                </div>
              );
            }

            const maxEpoch = Math.max(...data.metrics.map(m => m.epoch));
            const values = data.metrics.map(m => m[curveMetric]);

            return (
              <div key={exp.experiment_id} className="rounded-lg border border-white/[0.06] bg-white/[0.01] p-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="h-2.5 w-2.5 rounded-full" style={{ backgroundColor: colors[i % colors.length] }} />
                    <span className="text-xs font-medium text-foreground/80">{exp.config_name || exp.name}</span>
                  </div>
                  <span className="font-mono text-[10px] text-muted-foreground">
                    最终: {formatVal(values[values.length - 1])}
                  </span>
                </div>
                {/* 迷你曲线图 */}
                <div className="relative h-16">
                  <svg width="100%" height="100%" className="overflow-visible">
                    {/* 网格线 */}
                    {[0, 0.25, 0.5, 0.75, 1].map(ratio => (
                      <line
                        key={ratio}
                        x1="0" y1={`${ratio * 100}%`}
                        x2="100%" y2={`${ratio * 100}%`}
                        stroke="rgba(255,255,255,0.04)"
                        strokeWidth="1"
                      />
                    ))}
                    {/* 曲线 */}
                    {values.length > 1 && (
                      <polyline
                        points={values.map((v, j) => {
                          const x = maxEpoch > 1 ? (j / (values.length - 1)) * 100 : 50;
                          const y = ((v - min) / range) * 100;
                          return `${x},${100 - y}`;
                        }).join(' ')}
                        fill="none"
                        stroke={colors[i % colors.length]}
                        strokeWidth="1.5"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                      />
                    )}
                  </svg>
                </div>
                {/* X轴刻度 */}
                <div className="mt-1 flex justify-between text-[9px] text-muted-foreground/50">
                  <span>Epoch 1</span>
                  <span>Epoch {maxEpoch}</span>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ====== 配置对比 Tab ====== */
function ConfigTab({
  experiments, selectedIds, colors,
}: {
  experiments: ExperimentSummary[];
  selectedIds: Set<string>;
  colors: string[];
}) {
  const selected = experiments.filter(e => selectedIds.has(e.experiment_id));

  if (selected.length === 0) {
    return (
      <div className="flex items-center justify-center py-10 text-xs text-muted-foreground">
        请在上方选择要对比的实验
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-white/[0.06] text-left text-muted-foreground">
            <th className="pb-2 pr-4 font-medium">配置项</th>
            {selected.map((exp, i) => (
              <th key={exp.experiment_id} className="pb-2 pr-4 font-medium">
                <div className="flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full" style={{ backgroundColor: colors[i % colors.length] }} />
                  <span className="text-foreground/80">{exp.config_name || exp.name}</span>
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {[
            { label: '注意力机制', key: 'attention' as const },
            { label: '通道配置', key: 'channels' as const, format: (v: number[]) => `[${v.join(',')}]` },
            { label: '批归一化', key: 'use_bn' as const, format: (v: boolean) => v ? '是' : '否' },
            { label: 'Dropout', key: 'use_dropout' as const, format: (v: boolean) => v ? '是' : '否' },
            { label: '残差连接', key: 'use_residual' as const, format: (v: boolean) => v ? '是' : '否' },
            { label: '学习率', key: 'learning_rate' as const },
          ].map(row => (
            <tr key={row.key} className="border-b border-white/[0.03]">
              <td className="py-2 pr-4 font-medium text-foreground/70">{row.label}</td>
              {selected.map(exp => {
                const val = (exp as any)[row.key];
                return (
                  <td key={exp.experiment_id} className="py-2 pr-4 font-mono text-foreground/60">
                    {row.format ? row.format(val) : String(val)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}