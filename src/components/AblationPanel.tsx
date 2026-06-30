import { useState, useEffect, useRef, useCallback } from 'react';
import { trainingApi, type AblationResultItem } from '@/services/api';
import { useWorkbench } from './workbench/WorkbenchContext';
import { CustomExperimentPanel } from './CustomExperimentPanel';

interface AblationPanelProps {
  datasetId: string | null;
}

interface PresetConfig {
  key: string;
  label: string;
  prefix: string;
  configs: Array<{
    name: string;
    channels: number[];
    attention?: 'none' | 'se' | 'cbam' | 'self_attention';
    use_bn?: boolean;
    use_dropout?: boolean;
    use_residual?: boolean;
    learning_rate?: number;
  }>;
}

const ATTENTION_PRESETS: PresetConfig = {
  key: 'attention',
  label: '注意力对比',
  prefix: 'attn_cmp',
  configs: [
    { name: 'Baseline (无注意力)', channels: [32, 64], attention: 'none' },
    { name: 'SE 通道注意力', channels: [32, 64], attention: 'se' },
    { name: 'CBAM 注意力', channels: [32, 64], attention: 'cbam' },
    { name: 'Self-Attention', channels: [32, 64], attention: 'self_attention' },
  ],
};

const CHANNEL_PRESETS: PresetConfig = {
  key: 'channels',
  label: '卷积核对比',
  prefix: 'channel_cmp',
  configs: [
    { name: '[16, 32]', channels: [16, 32] },
    { name: '[32, 64]', channels: [32, 64] },
    { name: '[64, 128]', channels: [64, 128] },
    { name: '[64,128,256]', channels: [64, 128, 256] },
  ],
};

const LR_PRESETS: PresetConfig = {
  key: 'lr',
  label: '学习率对比',
  prefix: 'lr_cmp',
  configs: [
    { name: 'lr=0.01', channels: [32, 64], learning_rate: 0.01 },
    { name: 'lr=0.001', channels: [32, 64], learning_rate: 0.001 },
    { name: 'lr=0.0001', channels: [32, 64], learning_rate: 0.0001 },
  ],
};

const ABLATION_PRESETS: PresetConfig = {
  key: 'ablation',
  label: '组件消融',
  prefix: 'ablation_cmp',
  configs: [
    { name: '完整模型 (BN+DO+Res)', channels: [32, 64], use_bn: true, use_dropout: true, use_residual: true },
    { name: 'w/o BN', channels: [32, 64], use_bn: false, use_dropout: true, use_residual: false },
    { name: 'w/o Dropout', channels: [32, 64], use_bn: true, use_dropout: false, use_residual: false },
    { name: 'w/o Residual', channels: [32, 64], use_bn: true, use_dropout: true, use_residual: false },
  ],
};

const ALL_PRESETS = [ATTENTION_PRESETS, CHANNEL_PRESETS, LR_PRESETS, ABLATION_PRESETS];

function getStatusBadge(status: string) {
  const statusMap: Record<string, { label: string; className: string }> = {
    pending: { label: '等待中', className: 'bg-yellow-500/15 text-yellow-400 border-yellow-500/20' },
    running: { label: '训练中', className: 'bg-blue-500/15 text-blue-400 border-blue-500/20' },
    completed: { label: '已完成', className: 'bg-green-500/15 text-green-400 border-green-500/20' },
    failed: { label: '失败', className: 'bg-red-500/15 text-red-400 border-red-500/20' },
    stopped: { label: '已停止', className: 'bg-gray-500/15 text-gray-400 border-gray-500/20' },
    draft: { label: '待启动', className: 'bg-white/10 text-muted-foreground border-white/10' },
  };
  const s = statusMap[status] || { label: status || '未知', className: 'bg-white/10 text-muted-foreground border-white/10' };
  return (
    <span className={`inline-flex items-center rounded-md border px-1.5 py-0.5 text-[10px] font-medium ${s.className}`}>
      {s.label}
    </span>
  );
}

interface DisplayResultItem extends AblationResultItem {
  channels?: number[];
  attention?: string;
  use_bn?: boolean;
  use_dropout?: boolean;
  use_residual?: boolean;
  learning_rate?: number;
}

export function AblationPanel({ datasetId }: AblationPanelProps) {
  const { ablationResults, setAblationGroup, markAblationSaved, activeAblationGroupName, setActiveAblationGroupName } = useWorkbench();
  const [isExpanded, setIsExpanded] = useState(true);
  const [results, setResults] = useState<DisplayResultItem[]>([]);
  const [runningPreset, setRunningPreset] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameInput, setRenameInput] = useState('');
  const [renameError, setRenameError] = useState<string | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const activeGroupName = activeAblationGroupName;

  const normalizeResults = useCallback((rawResults: AblationResultItem[]): DisplayResultItem[] => {
    return rawResults.map(r => {
      // 后端 getAblationResults 返回的字段已在顶层，直接使用 r 即可
      return {
        ...r,
        attention: (r as any).attention || 'none',
        best_accuracy: r.best_accuracy ?? undefined,
        final_loss: r.final_loss ?? undefined,
      };
    });
  }, []);

  useEffect(() => {
    if (activeGroupName && ablationResults[activeGroupName]) {
      const saved = ablationResults[activeGroupName];
      setResults(normalizeResults(saved.results));
      // 切换到历史实验组时，从后端刷新最新状态，确保与实验库状态一致
      pollResults(activeGroupName, saved.presetKey, saved.presetLabel);
    } else if (!activeGroupName) {
      setResults([]);
    }
  }, [activeGroupName]);

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  }, []);

  const pollResults = useCallback(async (groupName: string, presetKey?: string, presetLabel?: string) => {
    try {
      const res = await trainingApi.getAblationResults(groupName);
      if (res.data?.results) {
        const normalized = normalizeResults(res.data.results);
        setResults(normalized);

        // 持久化到 context
        setAblationGroup({
          groupName,
          presetKey: presetKey || '',
          presetLabel: presetLabel || '',
          datasetId: datasetId || '',
          results: res.data.results,
          savedAt: Date.now(),
          saved: false,
        });

        const allDone = normalized.every(
          (r) => r.status === 'completed' || r.status === 'failed' || r.status === 'stopped'
        );
        if (allDone) {
          stopPolling();
          setRunningPreset(null);
        }
      }
    } catch (err) {
      console.error('获取消融结果失败:', err);
    }
  }, [stopPolling, setAblationGroup, datasetId]);

  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  const runPreset = async (preset: PresetConfig) => {
    if (!datasetId) {
      setError('请先选择数据集');
      return;
    }
    setError(null);
    setResults([]);
    setRunningPreset(preset.key);

    try {
      const timestamp = Date.now();
      const groupName = `${preset.prefix}_${timestamp}`;
      const configs = preset.configs.map(cfg => {
        const attentionType = cfg.attention || 'none';
        return {
          name: cfg.name,
          // 顶层字段：供后端 AblationExperimentRequest 直接读取
          channels: cfg.channels,
          attention: attentionType,           // 字符串：'none'|'se'|'cbam'|'self_attention'
          use_bn: cfg.use_bn !== undefined ? cfg.use_bn : true,
          use_dropout: cfg.use_dropout !== undefined ? cfg.use_dropout : true,
          use_residual: cfg.use_residual || false,
          learning_rate: cfg.learning_rate,   // 顶层：供后端按配置组覆盖学习率
          model_config: {
            channels: cfg.channels,
            attention: attentionType,         // 字符串：与后端 model_config 格式一致
            use_attention: attentionType !== 'none',
            attention_type: attentionType,
            use_bn: cfg.use_bn !== undefined ? cfg.use_bn : true,
            use_dropout: cfg.use_dropout !== undefined ? cfg.use_dropout : true,
            dropout_rate: 0.2,
            use_residual: cfg.use_residual || false,
            fc_hidden: [128],
          } as any,
        };
      });
      const params = {
        dataset_id: datasetId,
        name_prefix: groupName,
        epochs: 5,
        batch_size: 64,
        learning_rate: 0.001,
        val_split: 0.2,
        configs,
      };
      const res = await trainingApi.runAblationExperiment(params);
      const gn = res.data?.group_name || groupName;
      setActiveAblationGroupName(gn);
      pollResults(gn, preset.key, preset.label);
      pollTimerRef.current = setInterval(() => pollResults(gn, preset.key, preset.label), 3000);
    } catch (err: any) {
      setError(err.message || '启动消融实验失败');
      setRunningPreset(null);
    }
  };

  const handleSaveToLibrary = () => {
    if (activeGroupName) {
      // 标记为已保存——消融实验的子实验已经通过后端创建到实验库中
      // 这里只标记组已保存状态，用户可在"实验库"页面查看所有子实验
      markAblationSaved(activeGroupName);
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
      await trainingApi.renameExperiment(id, name);
      setResults(prev => prev.map(r => r.experiment_id === id ? { ...r, config_name: name } : r));
      setRenamingId(null);
      setRenameInput('');
      setRenameError(null);
    } catch (err: any) {
      setRenameError(err.message || '重命名失败');
    }
  };

  // 检查是否所有结果都已完成
  const allCompleted = results.length > 0 && results.every(
    (r) => r.status === 'completed' || r.status === 'failed' || r.status === 'stopped'
  );
  const isCurrentSaved = activeGroupName ? ablationResults[activeGroupName]?.saved : false;

  const sortedResults = [...results].sort((a, b) => (b.best_accuracy ?? -1) - (a.best_accuracy ?? -1));
  const maxAccuracy = sortedResults.length > 0 ? Math.max(...sortedResults.map(r => r.best_accuracy ?? 0)) : 0;
  const completedCount = results.filter(r => r.status === 'completed').length;
  const isRunning = runningPreset !== null;

  return (
    <div className="rounded-xl border border-white/[0.08] bg-[#12151e] overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex w-full items-center justify-between px-4 py-3 transition hover:bg-white/[0.02]"
      >
        <div className="flex items-center gap-2">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
            <path d="M3 3v18h18" />
            <path d="M7 16l4-8 4 4 5-9" />
          </svg>
          <h3 className="text-sm font-bold">对比实验 / 消融</h3>
          {isRunning && (
            <span className="flex items-center gap-1 rounded bg-blue-500/15 border border-blue-500/20 px-1.5 py-0.5 text-[10px] text-blue-400">
              <span className="h-1.5 w-1.5 rounded-full bg-blue-400 animate-pulse" />
              {completedCount}/{results.length}
            </span>
          )}
        </div>
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
          className={`text-muted-foreground transition-transform ${isExpanded ? 'rotate-180' : ''}`}>
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {isExpanded && (
        <div className="border-t border-white/[0.06] px-3 py-3 space-y-3">
          {!datasetId && (
            <div className="flex items-center gap-2 rounded-md bg-yellow-500/10 border border-yellow-500/20 px-3 py-2 text-xs text-yellow-400">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              请先选择数据集
            </div>
          )}
          {error && (
            <div className="flex items-center gap-2 rounded-md bg-red-500/10 border border-red-500/20 px-3 py-2 text-xs text-red-400">
              {error}
            </div>
          )}

          <div className="grid grid-cols-2 gap-1.5">
            {ALL_PRESETS.map(preset => (
              <button
                key={preset.key}
                onClick={() => runPreset(preset)}
                disabled={!datasetId || isRunning}
                className={`flex flex-col items-start gap-0.5 rounded-md border px-2.5 py-2 text-left transition-all ${
                  !datasetId || isRunning
                    ? 'cursor-not-allowed border-white/[0.04] bg-white/[0.01] text-muted-foreground/40'
                    : runningPreset === preset.key
                    ? 'border-primary/40 bg-primary/15 text-primary'
                    : 'border-white/[0.08] bg-white/[0.02] text-muted-foreground hover:bg-white/[0.05] hover:text-foreground'
                }`}
              >
                <div className="flex items-center gap-1.5">
                  {runningPreset === preset.key && (
                    <div className="h-2.5 w-2.5 animate-spin rounded-full border-2 border-primary border-t-transparent" />
                  )}
                  <span className="text-[11px] font-medium">{preset.label}</span>
                </div>
                <span className="text-[9px] text-muted-foreground/60">{preset.configs.length} 组对比</span>
              </button>
            ))}
          </div>

          <CustomExperimentPanel datasetId={datasetId} />

          {Object.keys(ablationResults).length > 0 && (
            <div className="space-y-1.5">
              <h4 className="text-[10px] font-semibold text-muted-foreground/70 uppercase tracking-wider">历史实验组</h4>
              <div className="max-h-32 overflow-y-auto space-y-1">
                {Object.values(ablationResults)
                  .sort((a, b) => b.savedAt - a.savedAt)
                  .map((group) => {
                    const isActive = group.groupName === activeGroupName;
                    const bestAcc = group.results
                      .map(r => r.best_accuracy ?? 0)
                      .reduce((max, acc) => Math.max(max, acc), 0);
                    return (
                      <button
                        key={group.groupName}
                        onClick={() => setActiveAblationGroupName(group.groupName)}
                        className={`flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-left text-[10px] transition-colors ${
                          isActive
                            ? 'bg-primary/15 text-primary border border-primary/30'
                            : 'bg-white/[0.02] text-muted-foreground hover:bg-white/[0.05] border border-transparent'
                        }`}
                      >
                        {group.saved ? (
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="text-emerald-400 flex-shrink-0">
                            <polyline points="20 6 9 17 4 12" />
                          </svg>
                        ) : (
                          <span className="h-2 w-2 rounded-full bg-amber-400/60 flex-shrink-0" />
                        )}
                        <span className="truncate flex-1">{group.presetLabel || group.groupName}</span>
                        {bestAcc > 0 && (
                          <span className="font-mono text-[9px] text-emerald-400/80 flex-shrink-0">
                            {(bestAcc * 100).toFixed(1)}%
                          </span>
                        )}
                        <span className="text-[9px] text-muted-foreground/50 flex-shrink-0">{group.results.length}组</span>
                      </button>
                    );
                  })}
              </div>
            </div>
          )}

          {sortedResults.length > 0 && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="text-[11px] font-semibold text-foreground/80">实验结果</h4>
                <div className="flex items-center gap-2">
                  <span className="text-[9px] text-muted-foreground font-mono">{activeGroupName?.slice(0, 25)}</span>
                  {allCompleted && !isCurrentSaved && (
                    <button
                      onClick={handleSaveToLibrary}
                      className="flex items-center gap-1 rounded border border-emerald-400/30 bg-emerald-400/10 px-2 py-0.5 text-[9px] font-semibold text-emerald-400 transition hover:bg-emerald-400/15"
                    >
                      <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                        <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
                        <path d="M17 21v-8H7v8M7 3v5h8" />
                      </svg>
                      保存到实验库
                    </button>
                  )}
                  {isCurrentSaved && (
                    <span className="flex items-center gap-1 rounded border border-emerald-400/20 bg-emerald-400/5 px-2 py-0.5 text-[9px] text-emerald-400">
                      <svg width="9" height="9" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                        <polyline points="20 6 9 17 4 12" />
                      </svg>
                      已保存
                    </span>
                  )}
                </div>
              </div>
              <div className="space-y-1.5">
                {sortedResults.map((item, idx) => {
                  const isBest = idx === 0 && item.best_accuracy != null && item.status === 'completed';
                  const acc = item.best_accuracy ?? 0;
                  const barW = maxAccuracy > 0 ? Math.max(5, (acc / maxAccuracy) * 100) : 0;
                  return (
                    <div key={item.experiment_id}
                      className={`rounded-md border p-2 ${isBest ? 'border-primary/40 bg-primary/[0.06]' : 'border-white/[0.06] bg-white/[0.02]'}`}>
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center gap-1.5 group/rename">
                          {renamingId === item.experiment_id ? (
                            <div className="flex flex-col gap-0.5">
                              <div className="flex items-center gap-1">
                                <input
                                  type="text"
                                  value={renameInput}
                                  onChange={(e) => setRenameInput(e.target.value)}
                                  onKeyDown={(e) => {
                                    if (e.key === 'Enter') handleConfirmRename(item.experiment_id);
                                    if (e.key === 'Escape') handleCancelRename();
                                  }}
                                  className="w-32 rounded border border-primary/40 bg-white/[0.04] px-1.5 py-0.5 text-[10px] text-foreground focus:outline-none focus:border-primary"
                                  autoFocus
                                />
                                <button
                                  onClick={() => handleConfirmRename(item.experiment_id)}
                                  className="flex h-4 w-4 items-center justify-center rounded bg-primary/20 text-primary hover:bg-primary/30"
                                  title="确认"
                                >
                                  <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                    <polyline points="20 6 9 17 4 12" />
                                  </svg>
                                </button>
                                <button
                                  onClick={handleCancelRename}
                                  className="flex h-4 w-4 items-center justify-center rounded bg-white/[0.06] text-muted-foreground hover:bg-white/[0.12]"
                                  title="取消"
                                >
                                  <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                    <path d="M18 6 6 18M6 6l12 12" />
                                  </svg>
                                </button>
                              </div>
                              {renameError && (
                                <span className="text-[8px] text-red-400">{renameError}</span>
                              )}
                            </div>
                          ) : (
                            <>
                              <span className={`text-[11px] font-medium ${isBest ? 'text-primary' : 'text-foreground/90'}`}>
                                {isBest && '🏆 '}{item.config_name}
                              </span>
                              <button
                                onClick={(e) => { e.stopPropagation(); handleStartRename(item.experiment_id, item.config_name); }}
                                className="opacity-0 group-hover/rename:opacity-100 flex h-4 w-4 items-center justify-center rounded text-muted-foreground/40 hover:bg-white/[0.08] hover:text-muted-foreground transition-all"
                                title="重命名"
                              >
                                <svg width="8" height="8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                  <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                                  <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                                </svg>
                              </button>
                            </>
                          )}
                        </div>
                        {getStatusBadge(item.status)}
                      </div>
                      {item.status === 'completed' && (
                        <>
                          <div className="flex items-center gap-2 mb-1">
                            <div className="flex-1 h-1.5 bg-white/[0.06] rounded-full overflow-hidden">
                              <div className={`h-full rounded-full ${isBest ? 'bg-gradient-to-r from-primary to-blue-400' : 'bg-white/25'}`} style={{ width: `${barW}%` }} />
                            </div>
                            <span className={`text-[10px] font-mono ${isBest ? 'text-primary font-bold' : 'text-muted-foreground'}`}>
                              {(acc * 100).toFixed(2)}%
                            </span>
                          </div>
                          <div className="flex flex-wrap gap-x-2 gap-y-0.5 text-[9px] text-muted-foreground/70">
                            <span>[{item.channels?.join(',')}]</span>
                            <span>attn={item.attention || 'none'}</span>
                            {item.use_bn !== undefined && <span>BN={item.use_bn ? 'Y' : 'N'}</span>}
                            {item.use_dropout !== undefined && <span>DO={item.use_dropout ? 'Y' : 'N'}</span>}
                            {item.total_params && <span>~{(item.total_params / 1000).toFixed(0)}K</span>}
                          </div>
                        </>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {!results.length && !isRunning && (
            <div className="flex flex-col items-center justify-center py-4 text-center">
              <p className="text-[10px] text-muted-foreground/60">选择一个对比项，系统自动训练多组配置并对比结果</p>
              <p className="mt-1 text-[9px] text-muted-foreground/40">完成后可保存到实验库</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
