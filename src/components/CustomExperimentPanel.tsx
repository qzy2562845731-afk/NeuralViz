import { useState, useEffect } from 'react';
import { trainingApi } from '../services/api';
import { useWorkbench } from './workbench/WorkbenchContext';
import { DarkSelect } from './ui/DarkSelect';

/* ============================================
   CustomExperimentPanel — 自定义实验面板
   - 自定义对比实验 / 自定义消融实验
   - 支持自定义参数配置、指标选择、对照组设置
   - 支持模板保存/加载/复用
   ============================================ */

interface CustomConfig {
  name: string;
  channels: number[];
  attention: string;
  use_bn: boolean;
  use_dropout: boolean;
  use_residual: boolean;
  learning_rate: number;
  optimizer: string;
  epochs: number;
}

interface CustomTemplate {
  template_id: string;
  name: string;
  description: string;
  template_type: 'comparison' | 'ablation';
  configs: CustomConfig[];
  comparison_metrics: string[];
  created_at: string;
}

interface CustomExperimentPanelProps {
  datasetId: string | null;
}

const AVAILABLE_METRICS = [
  { key: 'val_acc', label: '验证准确率' },
  { key: 'val_loss', label: '验证损失' },
  { key: 'train_loss', label: '训练损失' },
  { key: 'best_accuracy', label: '最佳准确率' },
  { key: 'final_loss', label: '最终损失' },
  { key: 'total_params', label: '参数量' },
  { key: 'training_duration', label: '训练耗时' },
  { key: 'f1', label: 'F1分数' },
  { key: 'precision', label: '精确率' },
  { key: 'recall', label: '召回率' },
];

const ATTENTION_OPTIONS = [
  { value: 'none', label: '无注意力' },
  { value: 'se', label: 'SE通道注意力' },
  { value: 'cbam', label: 'CBAM注意力' },
  { value: 'self_attention', label: 'Self-Attention' },
];

const OPTIMIZER_OPTIONS = [
  { value: 'adam', label: 'Adam' },
  { value: 'sgd', label: 'SGD' },
  { value: 'adamw', label: 'AdamW' },
];

function createDefaultConfig(name: string): CustomConfig {
  return {
    name,
    channels: [32, 64],
    attention: 'none',
    use_bn: true,
    use_dropout: true,
    use_residual: false,
    learning_rate: 0.001,
    optimizer: 'adam',
    epochs: 5,
  };
}

export function CustomExperimentPanel({ datasetId }: CustomExperimentPanelProps) {
  const { setActiveAblationGroupName } = useWorkbench();
  const [isOpen, setIsOpen] = useState(false);
  const [experimentType, setExperimentType] = useState<'comparison' | 'ablation'>('comparison');
  const [configs, setConfigs] = useState<CustomConfig[]>([
    createDefaultConfig('对照组'),
    createDefaultConfig('实验组'),
  ]);
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['val_acc', 'val_loss', 'best_accuracy']);
  const [globalEpochs, setGlobalEpochs] = useState(5);
  const [globalBatchSize, setGlobalBatchSize] = useState(64);
  const [globalLearningRate, setGlobalLearningRate] = useState(0.001);
  const [globalValSplit, setGlobalValSplit] = useState(0.2);
  const [namePrefix, setNamePrefix] = useState('');
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 模板相关
  const [templates, setTemplates] = useState<CustomTemplate[]>([]);
  const [showTemplates, setShowTemplates] = useState(false);
  const [templateName, setTemplateName] = useState('');
  const [templateDesc, setTemplateDesc] = useState('');
  const [showSaveDialog, setShowSaveDialog] = useState(false);
  const [saveTemplateError, setSaveTemplateError] = useState<string | null>(null);

  // 加载模板
  useEffect(() => {
    if (isOpen) {
      loadTemplates();
    }
  }, [isOpen, experimentType]);

  const loadTemplates = async () => {
    try {
      const resp = await trainingApi.listTemplates(experimentType);
      if (resp?.code === 200 && resp.data) {
        setTemplates(resp.data);
      }
    } catch (err) {
      // 静默失败
    }
  };

  // 添加配置组
  const addConfig = () => {
    setConfigs(prev => [...prev, createDefaultConfig(`配置组 ${prev.length + 1}`)]);
  };

  // 删除配置组
  const removeConfig = (index: number) => {
    if (configs.length <= 2) {
      setError('至少需要2个配置组进行对比');
      return;
    }
    setConfigs(prev => prev.filter((_, i) => i !== index));
  };

  // 更新配置
  const updateConfig = (index: number, field: keyof CustomConfig, value: any) => {
    setConfigs(prev => prev.map((cfg, i) => i === index ? { ...cfg, [field]: value } : cfg));
  };

  // 切换指标选择
  const toggleMetric = (key: string) => {
    setSelectedMetrics(prev => {
      if (prev.includes(key)) {
        if (prev.length <= 1) return prev;
        return prev.filter(k => k !== key);
      }
      return [...prev, key];
    });
  };

  // 运行自定义实验
  const runExperiment = async () => {
    if (!datasetId) {
      setError('请先选择数据集');
      return;
    }
    if (configs.length < 2) {
      setError('至少需要2个配置组');
      return;
    }
    const prefix = namePrefix.trim() || `custom_${experimentType}`;
    const timestamp = Date.now();
    const groupName = `${prefix}_${timestamp}`;

    setError(null);
    setRunning(true);

    try {
      const formattedConfigs = configs.map(cfg => ({
        name: cfg.name,
        channels: cfg.channels,
        attention: cfg.attention,
        use_bn: cfg.use_bn,
        use_dropout: cfg.use_dropout,
        use_residual: cfg.use_residual,
        learning_rate: cfg.learning_rate,
        optimizer: cfg.optimizer,
        epochs: cfg.epochs,
        model_config: {
          channels: cfg.channels,
          attention: cfg.attention !== 'none',
          attention_type: cfg.attention,
          use_attention: cfg.attention !== 'none',
          use_bn: cfg.use_bn,
          use_dropout: cfg.use_dropout,
          use_residual: cfg.use_residual,
          fc_hidden: [128],
          dropout_rate: 0.2,
          learning_rate: cfg.learning_rate,
        },
      }));

      const res = await trainingApi.runCustomExperiment({
        dataset_id: datasetId,
        experiment_type: experimentType,
        name_prefix: groupName,
        epochs: globalEpochs,
        batch_size: globalBatchSize,
        learning_rate: globalLearningRate,
        val_split: globalValSplit,
        configs: formattedConfigs,
        comparison_metrics: selectedMetrics,
      });

      const gn = res.data?.group_name || groupName;
      setActiveAblationGroupName(gn);
      setIsOpen(false);
    } catch (err: any) {
      setError(err.message || '启动自定义实验失败');
    } finally {
      setRunning(false);
    }
  };

  // 保存模板
  const saveTemplate = async () => {
    const name = templateName.trim();
    if (!name || name.length < 2 || name.length > 50) {
      setSaveTemplateError('模板名称需在2-50字符之间');
      return;
    }
    try {
      await trainingApi.saveTemplate({
        name,
        description: templateDesc.trim(),
        template_type: experimentType,
        configs,
        comparison_metrics: selectedMetrics,
      });
      setShowSaveDialog(false);
      setTemplateName('');
      setTemplateDesc('');
      setSaveTemplateError(null);
      loadTemplates();
    } catch (err: any) {
      setSaveTemplateError(err.message || '保存模板失败');
    }
  };

  // 加载模板
  const loadTemplate = (template: CustomTemplate) => {
    setConfigs(template.configs);
    setSelectedMetrics(template.comparison_metrics);
    setNamePrefix(template.name);
    setExperimentType(template.template_type);
    setShowTemplates(false);
  };

  const handleDeleteTemplate = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await trainingApi.deleteTemplate(id);
      setTemplates(prev => prev.filter(t => t.template_id !== id));
    } catch (err: any) {
      setError(err.message || '删除模板失败');
    }
  };

  if (!isOpen) {
    return (
      <button
        onClick={() => setIsOpen(true)}
        disabled={!datasetId}
        className={`flex w-full items-center justify-center gap-2 rounded-md border px-2.5 py-2 text-xs transition-all ${
          !datasetId
            ? 'cursor-not-allowed border-white/[0.04] bg-white/[0.01] text-muted-foreground/30'
            : 'border-dashed border-primary/30 bg-primary/[0.04] text-primary/70 hover:bg-primary/[0.08] hover:text-primary'
        }`}
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <line x1="12" y1="5" x2="12" y2="19" />
          <line x1="5" y1="12" x2="19" y2="12" />
        </svg>
        自定义实验
      </button>
    );
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-3xl max-h-[85vh] overflow-hidden rounded-2xl border border-white/[0.08] bg-[#0f1119] shadow-2xl flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-white/[0.06] px-5 py-3.5">
          <div>
            <h2 className="text-sm font-bold">自定义实验</h2>
            <p className="mt-0.5 text-[10px] text-muted-foreground">
              {experimentType === 'comparison' ? '自定义对比实验' : '自定义消融实验'}
            </p>
          </div>
          <button
            onClick={() => setIsOpen(false)}
            className="flex h-7 w-7 items-center justify-center rounded-md border border-white/[0.08] bg-white/[0.02] text-muted-foreground transition-all hover:text-foreground"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5 space-y-4">
          {error && (
            <div className="flex items-center gap-2 rounded-md bg-red-500/10 border border-red-500/20 px-3 py-2 text-xs text-red-400">
              {error}
              <button onClick={() => setError(null)} className="ml-auto text-red-400/50 hover:text-red-400">
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6 6 18M6 6l12 12" />
                </svg>
              </button>
            </div>
          )}

          {/* 实验类型 & 基础参数 */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="block text-[10px] font-medium text-muted-foreground mb-1">实验类型</label>
              <div className="flex rounded-md border border-white/[0.08] overflow-hidden">
                {(['comparison', 'ablation'] as const).map(t => (
                  <button
                    key={t}
                    onClick={() => setExperimentType(t)}
                    className={`flex-1 py-1.5 text-xs transition-all ${
                      experimentType === t
                        ? 'bg-primary/15 text-primary'
                        : 'bg-white/[0.02] text-muted-foreground hover:bg-white/[0.04]'
                    }`}
                  >
                    {t === 'comparison' ? '对比实验' : '消融实验'}
                  </button>
                ))}
              </div>
            </div>
            <div>
              <label className="block text-[10px] font-medium text-muted-foreground mb-1">实验名称前缀</label>
              <input
                type="text"
                value={namePrefix}
                onChange={(e) => setNamePrefix(e.target.value)}
                placeholder="自定义实验"
                className="w-full rounded-md border border-white/[0.08] bg-white/[0.02] px-2.5 py-1.5 text-xs text-foreground placeholder:text-muted-foreground/50 focus:border-primary/40 focus:outline-none"
              />
            </div>
          </div>

          {/* 全局训练参数 */}
          <div>
            <label className="block text-[10px] font-medium text-muted-foreground mb-2">全局训练参数</label>
            <div className="grid grid-cols-4 gap-2">
              <div>
                <label className="block text-[9px] text-muted-foreground/60 mb-0.5">Epochs</label>
                <input
                  type="number"
                  value={globalEpochs}
                  onChange={(e) => setGlobalEpochs(Math.max(1, parseInt(e.target.value) || 1))}
                  className="w-full rounded-md border border-white/[0.08] bg-white/[0.02] px-2 py-1.5 text-xs text-foreground focus:border-primary/40 focus:outline-none"
                  min={1}
                />
              </div>
              <div>
                <label className="block text-[9px] text-muted-foreground/60 mb-0.5">Batch Size</label>
                <input
                  type="number"
                  value={globalBatchSize}
                  onChange={(e) => setGlobalBatchSize(Math.max(1, parseInt(e.target.value) || 32))}
                  className="w-full rounded-md border border-white/[0.08] bg-white/[0.02] px-2 py-1.5 text-xs text-foreground focus:border-primary/40 focus:outline-none"
                  min={1}
                />
              </div>
              <div>
                <label className="block text-[9px] text-muted-foreground/60 mb-0.5">学习率</label>
                <input
                  type="number"
                  value={globalLearningRate}
                  onChange={(e) => setGlobalLearningRate(parseFloat(e.target.value) || 0.001)}
                  className="w-full rounded-md border border-white/[0.08] bg-white/[0.02] px-2 py-1.5 text-xs text-foreground focus:border-primary/40 focus:outline-none"
                  step={0.0001}
                  min={0.00001}
                />
              </div>
              <div>
                <label className="block text-[9px] text-muted-foreground/60 mb-0.5">验证集比例</label>
                <input
                  type="number"
                  value={globalValSplit}
                  onChange={(e) => setGlobalValSplit(Math.min(0.9, Math.max(0.05, parseFloat(e.target.value) || 0.2)))}
                  className="w-full rounded-md border border-white/[0.08] bg-white/[0.02] px-2 py-1.5 text-xs text-foreground focus:border-primary/40 focus:outline-none"
                  step={0.05}
                  min={0.05}
                  max={0.9}
                />
              </div>
            </div>
          </div>

          {/* 对比指标选择 */}
          <div>
            <label className="block text-[10px] font-medium text-muted-foreground mb-2">对比指标</label>
            <div className="flex flex-wrap gap-1.5">
              {AVAILABLE_METRICS.map(m => (
                <button
                  key={m.key}
                  onClick={() => toggleMetric(m.key)}
                  className={`rounded-md px-2 py-1 text-[10px] transition-all ${
                    selectedMetrics.includes(m.key)
                      ? 'bg-primary/15 text-primary border border-primary/30'
                      : 'bg-white/[0.03] text-muted-foreground/60 border border-white/[0.06] hover:text-muted-foreground'
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>
          </div>

          {/* 配置组列表 */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-[10px] font-medium text-muted-foreground">
                配置组 ({configs.length})
              </label>
              <div className="flex items-center gap-1.5">
                <button
                  onClick={() => setShowTemplates(!showTemplates)}
                  className="flex items-center gap-1 rounded border border-white/[0.08] bg-white/[0.02] px-2 py-1 text-[9px] text-muted-foreground hover:text-foreground transition-all"
                >
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <polyline points="14 2 14 8 20 8" />
                  </svg>
                  模板
                </button>
                <button
                  onClick={addConfig}
                  className="flex items-center gap-1 rounded border border-dashed border-primary/30 bg-primary/[0.04] px-2 py-1 text-[9px] text-primary/70 hover:bg-primary/[0.08] transition-all"
                >
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="12" y1="5" x2="12" y2="19" />
                    <line x1="5" y1="12" x2="19" y2="12" />
                  </svg>
                  添加
                </button>
              </div>
            </div>

            {/* 模板列表 */}
            {showTemplates && (
              <div className="mb-3 rounded-lg border border-white/[0.06] bg-white/[0.01] p-2 max-h-40 overflow-y-auto">
                {templates.length === 0 ? (
                  <p className="text-[9px] text-muted-foreground/50 text-center py-3">暂无已保存的模板</p>
                ) : (
                  <div className="space-y-1">
                    {templates.map(t => (
                      <div
                        key={t.template_id}
                        className="flex items-center justify-between rounded p-1.5 hover:bg-white/[0.04] cursor-pointer transition-colors"
                        onClick={() => loadTemplate(t)}
                      >
                        <div className="flex-1 min-w-0">
                          <div className="text-[10px] font-medium truncate">{t.name}</div>
                          <div className="text-[8px] text-muted-foreground/50">{t.configs.length}组配置</div>
                        </div>
                        <button
                          onClick={(e) => handleDeleteTemplate(t.template_id, e)}
                          className="flex-shrink-0 p-1 text-muted-foreground/30 hover:text-red-400 transition-colors"
                          title="删除模板"
                        >
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
                          </svg>
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {/* 配置卡片 */}
            <div className="space-y-2 max-h-60 overflow-y-auto">
              {configs.map((cfg, index) => (
                <div key={index} className="rounded-lg border border-white/[0.06] bg-white/[0.01] p-3">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="flex h-5 w-5 items-center justify-center rounded bg-primary/10 text-[10px] font-bold text-primary">
                        {index + 1}
                      </span>
                      <input
                        type="text"
                        value={cfg.name}
                        onChange={(e) => updateConfig(index, 'name', e.target.value)}
                        className="w-32 rounded border border-white/[0.08] bg-white/[0.02] px-2 py-1 text-xs font-medium text-foreground focus:border-primary/40 focus:outline-none"
                      />
                    </div>
                    <button
                      onClick={() => removeConfig(index)}
                      className="flex h-5 w-5 items-center justify-center rounded text-muted-foreground/40 hover:bg-red-400/10 hover:text-red-400 transition-colors"
                      title="删除配置"
                    >
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="18" y1="6" x2="6" y2="18" />
                        <line x1="6" y1="6" x2="18" y2="18" />
                      </svg>
                    </button>
                  </div>

                  <div className="grid grid-cols-3 gap-2">
                    <div>
                      <label className="block text-[8px] text-muted-foreground/50 mb-0.5">通道配置</label>
                      <input
                        type="text"
                        value={cfg.channels.join(',')}
                        onChange={(e) => {
                          const chs = e.target.value.split(',').map(s => parseInt(s.trim()) || 32);
                          updateConfig(index, 'channels', chs.length > 0 ? chs : [32, 64]);
                        }}
                        className="w-full rounded border border-white/[0.08] bg-white/[0.02] px-1.5 py-1 text-[10px] text-foreground focus:border-primary/40 focus:outline-none font-mono"
                        placeholder="32,64"
                      />
                    </div>
                    <div>
                      <label className="block text-[8px] text-muted-foreground/50 mb-0.5">注意力</label>
                      <DarkSelect
                        options={ATTENTION_OPTIONS}
                        value={cfg.attention}
                        onChange={(v) => updateConfig(index, 'attention', v)}
                      />
                    </div>
                    <div>
                      <label className="block text-[8px] text-muted-foreground/50 mb-0.5">优化器</label>
                      <DarkSelect
                        options={OPTIMIZER_OPTIONS}
                        value={cfg.optimizer}
                        onChange={(v) => updateConfig(index, 'optimizer', v)}
                      />
                    </div>
                    <div>
                      <label className="block text-[8px] text-muted-foreground/50 mb-0.5">学习率</label>
                      <input
                        type="number"
                        value={cfg.learning_rate}
                        onChange={(e) => updateConfig(index, 'learning_rate', parseFloat(e.target.value) || 0.001)}
                        className="w-full rounded border border-white/[0.08] bg-white/[0.02] px-1.5 py-1 text-[10px] text-foreground focus:border-primary/40 focus:outline-none"
                        step={0.0001}
                      />
                    </div>
                    <div>
                      <label className="block text-[8px] text-muted-foreground/50 mb-0.5">Epochs</label>
                      <input
                        type="number"
                        value={cfg.epochs}
                        onChange={(e) => updateConfig(index, 'epochs', Math.max(1, parseInt(e.target.value) || 5))}
                        className="w-full rounded border border-white/[0.08] bg-white/[0.02] px-1.5 py-1 text-[10px] text-foreground focus:border-primary/40 focus:outline-none"
                        min={1}
                      />
                    </div>
                  </div>

                  {/* 组件开关 */}
                  <div className="flex items-center gap-3 mt-2">
                    {(['use_bn', 'use_dropout', 'use_residual'] as const).map(key => (
                      <label key={key} className="flex items-center gap-1 cursor-pointer">
                        <input
                          type="checkbox"
                          checked={cfg[key]}
                          onChange={(e) => updateConfig(index, key, e.target.checked)}
                          className="h-3 w-3 rounded border-white/20 bg-white/5 accent-primary"
                        />
                        <span className="text-[9px] text-muted-foreground/70">
                          {{ use_bn: 'BN', use_dropout: 'Dropout', use_residual: '残差' }[key]}
                        </span>
                      </label>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-white/[0.06] px-5 py-3">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setShowSaveDialog(true)}
              className="flex items-center gap-1 rounded-md border border-white/[0.08] bg-white/[0.02] px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground transition-all"
            >
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
                <polyline points="17 21 17 13 7 13 7 21" />
                <polyline points="7 3 7 8 15 8" />
              </svg>
              保存模板
            </button>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsOpen(false)}
              className="rounded-md border border-white/[0.08] bg-white/[0.02] px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground transition-all"
            >
              取消
            </button>
            <button
              onClick={runExperiment}
              disabled={running || !datasetId}
              className="flex items-center gap-1.5 rounded-md bg-primary px-3 py-1.5 text-xs font-semibold text-white transition-all hover:bg-primary/80 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {running ? (
                <>
                  <div className="h-3 w-3 animate-spin rounded-full border-2 border-white border-t-transparent" />
                  启动中...
                </>
              ) : (
                <>
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polygon points="5 3 19 12 5 21 5 3" />
                  </svg>
                  运行实验
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* 保存模板弹窗 */}
      {showSaveDialog && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/40">
          <div className="w-72 rounded-xl border border-white/[0.08] bg-[#1a1d2e] p-4 shadow-xl">
            <h3 className="text-xs font-bold mb-3">保存实验模板</h3>
            <div className="space-y-2">
              <div>
                <input
                  type="text"
                  value={templateName}
                  onChange={(e) => setTemplateName(e.target.value)}
                  placeholder="模板名称（2-50字符）"
                  className="w-full rounded border border-white/[0.08] bg-white/[0.02] px-2.5 py-1.5 text-xs text-foreground placeholder:text-muted-foreground/50 focus:border-primary/40 focus:outline-none"
                  autoFocus
                />
              </div>
              <div>
                <input
                  type="text"
                  value={templateDesc}
                  onChange={(e) => setTemplateDesc(e.target.value)}
                  placeholder="模板描述（可选）"
                  className="w-full rounded border border-white/[0.08] bg-white/[0.02] px-2.5 py-1.5 text-xs text-foreground placeholder:text-muted-foreground/50 focus:border-primary/40 focus:outline-none"
                />
              </div>
              {saveTemplateError && (
                <p className="text-[9px] text-red-400">{saveTemplateError}</p>
              )}
            </div>
            <div className="flex justify-end gap-2 mt-3">
              <button
                onClick={() => { setShowSaveDialog(false); setSaveTemplateError(null); }}
                className="rounded border border-white/[0.08] bg-white/[0.02] px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground"
              >
                取消
              </button>
              <button
                onClick={saveTemplate}
                className="rounded bg-primary/20 px-3 py-1.5 text-xs font-semibold text-primary hover:bg-primary/30"
              >
                保存
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}