import { useState, useEffect, useRef, useCallback, Fragment } from 'react';
import { apiService } from '../services/api';
import { useGlobalTraining } from '../contexts/GlobalTrainingContext';
import { useToast } from '../contexts/ToastContext';
import { TrainingCurvesPanel } from './TrainingCurvesPanel';
import { DarkSelect } from './ui/DarkSelect';
import type { ChartColors } from '../hooks/useColorConfig';

/* ============================================
   AutoTrainingPanel — 自动化训练面板
   - 自定义模型架构参数
   - 训练超参数配置
   - 一键启动训练 + 实时可视化
   ============================================ */

interface DatasetOption {
  dataset_id: string;
  name: string;
  status: string;
  dataset_type: string;
  feature_shape: number[];
  num_classes: number;
}

/** 安全地将 feature_shape 格式化为维度文本，处理空值/非数组等异常场景 */
function formatFeatureShape(shape: unknown): string {
  if (!Array.isArray(shape) || shape.length === 0) return '—';
  return shape.map((v) => (typeof v === 'number' ? v : String(v))).join('×');
}

interface ModelArchConfig {
  channels: number[];
  attention: string;
  use_bn: boolean;
  use_dropout: boolean;
  dropout_rate: number;
  use_residual: boolean;
  fc_hidden: number[];
  activation: string;
}

interface HyperparamConfig {
  learning_rate: number;
  batch_size: number;
  epochs: number;
  optimizer: string;
  val_split: number;
  loss_function: string;
  random_seed: number;
  scheduler_type: string;
  early_stopping: boolean;
  early_stopping_patience: number;
  use_amp: boolean;
}

const DEFAULT_COLORS: ChartColors = {
  trainLoss: '#e879f9', valLoss: '#fbbf24',
  trainAccuracy: '#4ade80', valAccuracy: '#60a5fa',
  learningRate: '#f472b6', gradientNorm: '#fbbf24',
  weightNorm: '#c084fc', confusionMatrix: '#4ade80',
  activationHistogram: ['#4ade80', '#fbbf24', '#e879f9', '#60a5fa'],
  cnnLayers: ['#4ade80', '#fbbf24', '#e879f9', '#60a5fa'],
  featureMaps: '#c084fc', success: '#4ade80', warning: '#fbbf24',
  danger: '#f87171', info: '#60a5fa', primary: '#4ade80',
  secondary: '#64748b', accent: '#e879f9',
};

const ATTENTION_OPTIONS = [
  { value: 'none', label: '无注意力' },
  { value: 'se', label: 'SE通道注意力' },
  { value: 'eca', label: 'ECA轻量通道注意力' },
  { value: 'cbam', label: 'CBAM注意力(通道+空间)' },
  { value: 'self_attention', label: 'Self-Attention(非局部)' },
  { value: 'mhsa', label: 'Multi-Head Self-Attention' },
  { value: 'gct', label: 'GCT门控通道变换' },
  { value: 'coord', label: 'Coordinate Attention' },
];

const OPTIMIZER_OPTIONS = [
  { value: 'adam', label: 'Adam' },
  { value: 'sgd', label: 'SGD' },
  { value: 'adamw', label: 'AdamW' },
  { value: 'rmsprop', label: 'RMSprop' },
];

const LOSS_FUNCTIONS = [
  { value: 'cross_entropy', label: 'CrossEntropyLoss' },
  { value: 'nll', label: 'NLLLoss' },
  { value: 'focal', label: 'FocalLoss (类别不平衡)' },
  { value: 'label_smoothing', label: 'LabelSmoothingCE (标签平滑)' },
  { value: 'dice', label: 'DiceLoss (Dice系数)' },
  { value: 'bce_with_logits', label: 'BCEWithLogitsLoss' },
  { value: 'mse', label: 'MSELoss' },
  { value: 'smooth_l1', label: 'SmoothL1Loss (Huber)' },
  { value: 'asymmetric', label: 'AsymmetricLoss (多标签)' },
];

const SCHEDULERS = [
  { value: 'none', label: '无调度器' },
  { value: 'step', label: 'StepLR' },
  { value: 'cosine', label: 'CosineAnnealingLR' },
  { value: 'reduce_on_plateau', label: 'ReduceLROnPlateau' },
];

const MODEL_IMPORT_FORMATS = [
  { value: 'pytorch', label: 'PyTorch (.pt/.pth)' },
  { value: 'onnx', label: 'ONNX (.onnx)' },
];

const DATASET_IMPORT_FORMATS = [
  { value: 'image_folder', label: '图像文件夹 (.zip)' },
  { value: 'csv', label: 'CSV表格 (.csv)' },
  { value: 'numpy', label: 'NumPy (.npy/.npz)' },
  { value: 'json', label: 'JSON (.json)' },
];

const ACTIVATIONS = [
  { value: 'relu', label: 'ReLU' },
  { value: 'leaky_relu', label: 'LeakyReLU' },
  { value: 'gelu', label: 'GELU' },
  { value: 'silu', label: 'SiLU' },
  { value: 'tanh', label: 'Tanh' },
];

const CHANNEL_PRESETS = [
  { label: '[16, 32]', value: [16, 32] },
  { label: '[32, 64]', value: [32, 64] },
  { label: '[32, 64, 128]', value: [32, 64, 128] },
  { label: '[64, 128, 256]', value: [64, 128, 256] },
  { label: '[64, 128, 256, 512]', value: [64, 128, 256, 512] },
];

export function AutoTrainingPanel() {
  const toast = useToast();
  const {
    data: trainingData, currentStep, isRealTraining,
    backendStatus, currentEpoch, totalEpochs, hasActiveTraining,
    trainingError, elapsedSeconds, trainingLogs,
    startRealTraining, stopRealTraining, reset,
  } = useGlobalTraining();

  const [datasets, setDatasets] = useState<DatasetOption[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<DatasetOption | null>(null);
  const [loadingDatasets, setLoadingDatasets] = useState(false);
  const [experimentId, setExperimentId] = useState<string | null>(null);
  const [isStarting, setIsStarting] = useState(false);
  const isStartingRef = useRef(false);

  // 新手引导：当前步骤 (1-4)
  const [showGuide, setShowGuide] = useState(true);

  // 数据集选择处理
  const handleSelectDataset = useCallback((ds: DatasetOption) => {
    setSelectedDataset(ds);
  }, []);

  // 模型导入
  const [modelImportFormat, setModelImportFormat] = useState('pytorch');
  const [importedModel, setImportedModel] = useState<{
    model_name: string;
    total_params: number;
    layer_count: number;
    model_path: string;
    model_format: string;
    model_class: string;
    weight_loaded: boolean;
    layers: any[];
  } | null>(null);
  const modelFileInputRef = useRef<HTMLInputElement>(null);

  // 数据集导入
  const [datasetImportFormat, setDatasetImportFormat] = useState('image_folder');
  const [importingDataset, setImportingDataset] = useState(false);
  const datasetFileInputRef = useRef<HTMLInputElement>(null);

  // 模型架构配置
  const [modelArch, setModelArch] = useState<ModelArchConfig>({
    channels: [32, 64],
    attention: 'none',
    use_bn: true,
    use_dropout: true,
    dropout_rate: 0.2,
    use_residual: false,
    fc_hidden: [128],
    activation: 'relu',
  });

  // 自定义通道输入
  const [customChannelsText, setCustomChannelsText] = useState('32,64');

  // 超参数配置
  const [hyperparams, setHyperparams] = useState<HyperparamConfig>({
    learning_rate: 0.001,
    batch_size: 64,
    epochs: 20,
    optimizer: 'adam',
    val_split: 0.2,
    loss_function: 'cross_entropy',
    random_seed: 42,
    scheduler_type: 'none',
    early_stopping: false,
    early_stopping_patience: 5,
    use_amp: false,
  });

  // 加载数据集列表
  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    setLoadingDatasets(true);
    try {
      const res = await apiService.listDatasets({ page: 1, page_size: 100 });
      if (res.data?.items) {
        setDatasets(res.data.items.filter((d: any) => d.status === 'ready') as unknown as DatasetOption[]);
      }
    } catch {
      // 静默失败
    } finally {
      setLoadingDatasets(false);
    }
  };

  // 当数据集列表更新时，验证当前选中的数据集是否仍然有效
  useEffect(() => {
    if (selectedDataset && datasets.length > 0) {
      const stillExists = datasets.some(ds => ds.dataset_id === selectedDataset.dataset_id);
      if (!stillExists) {
        setSelectedDataset(null);
      }
    }
  }, [datasets, selectedDataset]);

  const updateArch = useCallback((field: keyof ModelArchConfig, value: any) => {
    setModelArch(prev => ({ ...prev, [field]: value }));
  }, []);

  const updateHyperparam = useCallback((field: keyof HyperparamConfig, value: any) => {
    setHyperparams(prev => ({ ...prev, [field]: value }));
  }, []);

  const handleStartTraining = async () => {
    if (isStartingRef.current) return;
    if (!selectedDataset) {
      toast.showError('请先选择数据集');
      return;
    }

    isStartingRef.current = true;
    setIsStarting(true);

    try {
      const modelName = `AutoCNN_${modelArch.channels.join('x')}_${modelArch.attention}_${Date.now()}`;
      const expRes = await apiService.createExperiment({
        name: modelName,
        description: `自动训练 - 数据集: ${selectedDataset.name}`,
        model_id: `auto_${Date.now()}`,
        model_name: 'AutoCNN',
        model_architecture: {
          type: 'cnn',
          name: modelName,
          channels: modelArch.channels,
          attention: modelArch.attention,
          use_bn: modelArch.use_bn,
          use_dropout: modelArch.use_dropout,
          dropout_rate: modelArch.dropout_rate,
          use_residual: modelArch.use_residual,
          fc_hidden: modelArch.fc_hidden,
          activation: modelArch.activation,
        },
        total_params: 0,
        layer_count: 0,
        hyperparams: { ...hyperparams },
        config: {
          dataset_id: selectedDataset.dataset_id,
          dataset_name: selectedDataset.name,
          dataset_type: selectedDataset.dataset_type,
          feature_shape: selectedDataset.feature_shape,
          model_config: {
            channels: modelArch.channels,
            attention: modelArch.attention,
            use_attention: modelArch.attention !== 'none',
            use_bn: modelArch.use_bn,
            use_dropout: modelArch.use_dropout,
            dropout_rate: modelArch.dropout_rate,
            use_residual: modelArch.use_residual,
            fc_hidden: modelArch.fc_hidden,
            activation: modelArch.activation,
            num_classes: selectedDataset.num_classes || 10,
            in_channels: selectedDataset.feature_shape?.[0] || 1,
          },
        },
        tags: ['auto-training', selectedDataset.dataset_type || 'unknown'],
        status: 'running',
      });

      if (expRes.code !== 200 || !expRes.data) {
        throw new Error(expRes.message || '创建实验失败');
      }

      const expId = expRes.data.experiment_id;
      setExperimentId(expId);

      const success = await startRealTraining(expId, selectedDataset.dataset_id, { ...hyperparams }, {
        channels: modelArch.channels,
        attention: modelArch.attention,
        use_attention: modelArch.attention !== 'none',
        use_bn: modelArch.use_bn,
        use_dropout: modelArch.use_dropout,
        dropout_rate: modelArch.dropout_rate,
        use_residual: modelArch.use_residual,
        fc_hidden: modelArch.fc_hidden,
        activation: modelArch.activation,
        num_classes: selectedDataset.num_classes || 10,
        in_channels: selectedDataset.feature_shape?.[0] || 1,
      });

      if (success) {
        toast.showSuccess('训练已启动', `实验ID: ${expId.slice(0, 8)}...`);
      } else {
        toast.showError('训练启动失败', trainingError || '请检查后端服务');
      }
    } catch (err: any) {
      toast.showError('启动训练失败', err.message || '未知错误');
    } finally {
      isStartingRef.current = false;
      setIsStarting(false);
    }
  };

  const handleStopTraining = async () => {
    try {
      await stopRealTraining();
      toast.showInfo('训练已停止', '已保存的指标和模型权重将保留');
    } catch (err: any) {
      toast.showError('停止训练失败', err.message);
    }
  };

  const handleReset = () => {
    reset();
    setExperimentId(null);
  };

  // 模型导入处理
  const handleModelImport = () => {
    modelFileInputRef.current?.click();
  };

  const handleModelFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    try {
      const res = await apiService.importModel({
        model_name: file.name.replace(/\.(pt|pth|onnx)$/i, ''),
        model_format: modelImportFormat,
        dataset_type: selectedDataset?.dataset_type || 'image_folder',
        feature_shape: selectedDataset?.feature_shape ? formatFeatureShape(selectedDataset.feature_shape) : '1x28x28',
        num_classes: selectedDataset?.num_classes || 10,
      });
      if (res.code === 200 && res.data) {
        setImportedModel(res.data);
        toast.showSuccess('模型导入成功', `${res.data.total_params?.toLocaleString()} 参数, ${res.data.layer_count} 层`);
      } else {
        toast.showError('模型导入失败', res.message || '未知错误');
      }
    } catch (err: any) {
      toast.showError('模型导入失败', err.message || '未知错误');
    }
    // 重置 input 以便重新选择同一文件
    e.target.value = '';
  };

  // 数据集导入处理
  const handleDatasetImport = () => {
    datasetFileInputRef.current?.click();
  };

  const handleDatasetFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setImportingDataset(true);
    try {
      let res;
      const ext = file.name.split('.').pop()?.toLowerCase() || '';

      if (ext === 'zip') {
        // 压缩包格式走原有上传接口
        res = await apiService.uploadDataset(file, file.name.replace(/\.zip$/i, ''));
      } else {
        // CSV/JSON/NumPy 走直接上传接口
        res = await apiService.uploadDirectDataset(file, file.name);
      }

      if (res.code === 200 && res.data) {
        toast.showSuccess('数据集上传成功', `${res.data.name} - 正在后台解析`);
        // 刷新数据集列表
        await loadDatasets();
      } else {
        toast.showError('数据集上传失败', res.message || '未知错误');
      }
    } catch (err: any) {
      toast.showError('数据集上传失败', err.message || '未知错误');
    } finally {
      setImportingDataset(false);
      e.target.value = '';
    }
  };

  const isTraining = isRealTraining || hasActiveTraining;

  const currentGuideStep = !selectedDataset ? 1
    : !isTraining && trainingData.length === 0 ? 2
    : !isTraining ? 3
    : 4;

  const visibleData = trainingData.filter(d => d.step <= currentStep);

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-[#0a0c14] text-white">
      {/* 顶部栏 */}
      <div className="flex-shrink-0 border-b border-white/[0.06] bg-[#0c0e17]/95 backdrop-blur-sm">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-500 to-emerald-600 shadow-lg shadow-emerald-500/20">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-white">
                <path d="M5 3l14 9-14 9V3z" />
              </svg>
            </div>
            <div>
              <h1 className="text-lg font-bold">自动化训练</h1>
              <p className="text-xs text-muted-foreground">自定义模型架构，一键启动训练</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {isTraining && !isRealTraining && (
              <span className="flex items-center gap-1 rounded bg-blue-400/10 px-2 py-1 text-xs text-blue-400">
                <span className="h-2 w-2 rounded-full bg-blue-400 animate-pulse" />
                {backendStatus}
              </span>
            )}
            {backendStatus === 'completed' && trainingData.length > 0 && (
              <span className="rounded bg-emerald-400/10 px-2 py-1 text-xs text-emerald-400">
                训练完成
              </span>
            )}
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-6xl px-6 py-6">

          {/* 新手引导步骤条 */}
          {showGuide && !isTraining && (
            <div className="mb-5 rounded-xl border border-primary/15 bg-primary/[0.04] p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="flex items-center gap-2 text-sm font-bold text-primary">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>
                  </svg>
                  快速入门 - 三步完成训练
                </h3>
                <button onClick={() => setShowGuide(false)} className="text-muted-foreground/50 hover:text-muted-foreground transition-colors" title="关闭引导">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 6L6 18M6 6l12 12"/></svg>
                </button>
              </div>
              <div className="flex items-center gap-1">
                {[
                  { step: 1, label: '选择数据集', desc: '从列表中选择或上传新数据集', icon: 'M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zM14 2v6h6' },
                  { step: 2, label: '配置参数', desc: '调整模型架构和训练超参数', icon: 'M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2zM12 16v-4M12 8h.01' },
                  { step: 3, label: '开始训练', desc: '点击按钮启动训练，实时查看进度', icon: 'M5 3l14 9-14 9V3z' },
                ].map((s, i) => (
                  <Fragment key={s.step}>
                    {i > 0 && (
                      <div className={`flex-1 h-0.5 rounded-full transition-colors ${currentGuideStep > s.step - 1 ? 'bg-primary/40' : 'bg-white/[0.06]'}`} />
                    )}
                    <div className={`flex items-center gap-2 rounded-lg px-3 py-2 transition-all ${
                      currentGuideStep === s.step ? 'bg-primary/10 border border-primary/20' :
                      currentGuideStep > s.step ? 'bg-primary/[0.03]' : 'bg-transparent'
                    }`}>
                      <div className={`flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-[10px] font-bold transition-all ${
                        currentGuideStep > s.step ? 'bg-primary text-primary-foreground' :
                        currentGuideStep === s.step ? 'bg-primary/20 text-primary border border-primary/40' :
                        'bg-white/[0.04] text-muted-foreground border border-white/[0.06]'
                      }`}>
                        {currentGuideStep > s.step ? (
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3"><polyline points="20 6 9 17 4 12"/></svg>
                        ) : s.step}
                      </div>
                      <div>
                        <div className="text-[11px] font-medium text-foreground">{s.label}</div>
                        <div className="text-[9px] text-muted-foreground/70">{s.desc}</div>
                      </div>
                    </div>
                  </Fragment>
                ))}
              </div>
            </div>
          )}

          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {/* 数据集选择 */}
            <section className="rounded-xl border border-white/[0.06] bg-[#0f1119] p-5">
              <h2 className="mb-3 flex items-center gap-2 text-sm font-bold">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                  <path d="M14 2v6h6" />
                </svg>
                数据集
              </h2>
              {loadingDatasets ? (
                <p className="text-xs text-muted-foreground">加载中...</p>
              ) : datasets.length === 0 ? (
                <div className="flex flex-col items-center gap-2 py-4">
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-white/[0.03]">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-muted-foreground/40">
                      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zM14 2v6h6M12 18v-6M9 15h6"/>
                    </svg>
                  </div>
                  <p className="text-xs text-muted-foreground">暂无可用数据集</p>
                  <p className="text-[10px] text-muted-foreground/50">请先在下方「数据集导入」区域上传数据，或前往数据集页面导入</p>
                </div>
              ) : (
                <div className="grid grid-cols-1 gap-2 max-h-48 overflow-y-auto">
                  {datasets.map(ds => (
                    <button
                      key={ds.dataset_id}
                      onClick={() => handleSelectDataset(ds)}
                      className={`flex items-center justify-between rounded-lg border px-3 py-2 text-left text-xs transition-all ${
                        selectedDataset?.dataset_id === ds.dataset_id
                          ? 'border-primary/40 bg-primary/[0.06]'
                          : 'border-white/[0.06] bg-white/[0.02] hover:bg-white/[0.04]'
                      }`}
                    >
                      <div>
                        <span className="font-medium text-foreground">{ds.name}</span>
                        <span className="ml-2 text-muted-foreground/60">{ds.dataset_type}</span>
                      </div>
                      <div className="flex items-center gap-2 text-muted-foreground/50">
                        <span className="font-mono text-[10px]">{ds.num_classes}类</span>
                        <span className="font-mono text-[10px]">{formatFeatureShape(ds.feature_shape)}</span>
                      </div>
                    </button>
                  ))}
                </div>
              )}
              {selectedDataset && (
                <div className="mt-3 flex items-center gap-2 rounded-md bg-primary/10 px-3 py-2 text-xs text-primary">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  已选择: {selectedDataset.name} ({selectedDataset.num_classes}类, {formatFeatureShape(selectedDataset.feature_shape)})
                </div>
              )}
            </section>

            {/* 模型架构配置 */}
            <section className="rounded-xl border border-white/[0.06] bg-[#0f1119] p-5">
              <h2 className="mb-3 flex items-center gap-2 text-sm font-bold">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
                  <rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" />
                  <rect x="14" y="14" width="7" height="7" /><rect x="3" y="14" width="7" height="7" />
                </svg>
                模型架构
              </h2>
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div>
                  <label className="block text-[10px] font-medium text-muted-foreground mb-1">通道配置</label>
                  <div className="flex gap-1.5 flex-wrap">
                    {CHANNEL_PRESETS.map(p => (
                      <button
                        key={p.label}
                        onClick={() => {
                          updateArch('channels', p.value);
                          setCustomChannelsText(p.value.join(','));
                        }}
                        className={`rounded px-2 py-1 text-[10px] font-mono transition-all ${
                          modelArch.channels.join(',') === p.value.join(',')
                            ? 'bg-primary/15 text-primary border border-primary/30'
                            : 'bg-white/[0.03] text-muted-foreground border border-white/[0.06] hover:text-foreground'
                        }`}
                      >
                        {p.label}
                      </button>
                    ))}
                  </div>
                  <input
                    type="text"
                    value={customChannelsText}
                    onChange={e => {
                      setCustomChannelsText(e.target.value);
                      const val = e.target.value.trim();
                      if (!val) return;
                      try {
                        // 支持JSON格式嵌套数组：[[16,32],[32,64]]
                        if (val.startsWith('[') && val.includes('[')) {
                          const parsed = JSON.parse(val);
                          if (Array.isArray(parsed) && parsed.length > 0) {
                            updateArch('channels', parsed as any);
                          }
                        } else {
                          // 逗号分隔的扁平通道：16,32,64
                          const vals = val.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n > 0);
                          if (vals.length > 0) updateArch('channels', vals);
                        }
                      } catch {
                        // JSON解析失败，尝试逗号分隔
                        const vals = val.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n > 0);
                        if (vals.length > 0) updateArch('channels', vals);
                      }
                    }}
                    placeholder="如 32,64 或 [[16,32],[32,64]]"
                    className="mt-1.5 w-full rounded border border-white/[0.08] bg-white/[0.02] px-2 py-1.5 text-[10px] font-mono text-foreground focus:border-primary/40 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-[10px] font-medium text-muted-foreground mb-1">注意力机制</label>
                  <DarkSelect
                    options={ATTENTION_OPTIONS}
                    value={modelArch.attention}
                    onChange={(v) => updateArch('attention', v)}
                  />
                </div>
                <div>
                  <label className="block text-[10px] font-medium text-muted-foreground mb-1">激活函数</label>
                  <div className="flex gap-1 flex-wrap">
                    {ACTIVATIONS.map(a => (
                      <button
                        key={a.value}
                        onClick={() => updateArch('activation', a.value)}
                        className={`rounded px-2 py-1 text-[10px] transition-all ${
                          modelArch.activation === a.value
                            ? 'bg-primary/15 text-primary border border-primary/30'
                            : 'bg-white/[0.03] text-muted-foreground border border-white/[0.06] hover:text-foreground'
                        }`}
                      >
                        {a.label}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <label className="block text-[10px] font-medium text-muted-foreground mb-1">FC隐藏层</label>
                  <input
                    type="text"
                    value={modelArch.fc_hidden.join(',')}
                    onChange={e => {
                      const vals = e.target.value.split(',').map(s => parseInt(s.trim()) || 128);
                      updateArch('fc_hidden', vals.length > 0 ? vals : [128]);
                    }}
                    className="w-full rounded border border-white/[0.08] bg-white/[0.02] px-2 py-1.5 text-xs font-mono text-foreground focus:border-primary/40 focus:outline-none"
                    placeholder="128,64"
                  />
                </div>
              </div>
              <div className="mt-3 flex items-center gap-4">
                {([
                  { key: 'use_bn' as const, label: 'BatchNorm' },
                  { key: 'use_dropout' as const, label: 'Dropout' },
                  { key: 'use_residual' as const, label: '残差连接' },
                ]).map(item => (
                  <label key={item.key} className="flex items-center gap-1.5 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={modelArch[item.key]}
                      onChange={e => updateArch(item.key, e.target.checked)}
                      className="h-3.5 w-3.5 rounded border-white/20 bg-white/5 accent-primary"
                    />
                    <span className="text-[10px] text-muted-foreground">{item.label}</span>
                  </label>
                ))}
                {modelArch.use_dropout && (
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] text-muted-foreground">rate:</span>
                    <input
                      type="number"
                      value={modelArch.dropout_rate}
                      onChange={e => updateArch('dropout_rate', Math.min(0.9, Math.max(0, parseFloat(e.target.value) || 0.2)))}
                      className="w-14 rounded border border-white/[0.08] bg-white/[0.02] px-1.5 py-1 text-[10px] text-foreground focus:border-primary/40 focus:outline-none"
                      step={0.1}
                      min={0}
                      max={0.9}
                    />
                  </div>
                )}
              </div>
            </section>

            {/* 训练超参数 */}
            <section className="rounded-xl border border-white/[0.06] bg-[#0f1119] p-5 lg:col-span-2">
              <h2 className="mb-3 flex items-center gap-2 text-sm font-bold">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
                  <circle cx="12" cy="12" r="3" /><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
                </svg>
                训练超参数
              </h2>
              <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 lg:grid-cols-6">
                <div>
                  <label className="text-[10px] text-muted-foreground/60">学习率</label>
                  <input
                    type="number" step={0.0001} min={0.00001}
                    value={hyperparams.learning_rate}
                    onChange={e => updateHyperparam('learning_rate', parseFloat(e.target.value) || 0.001)}
                    className="w-full rounded border border-white/[0.08] bg-white/[0.02] px-2 py-1.5 text-xs text-foreground focus:border-primary/40 focus:outline-none font-mono"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-muted-foreground/60">Epochs</label>
                  <input
                    type="number" min={1}
                    value={hyperparams.epochs}
                    onChange={e => updateHyperparam('epochs', Math.max(1, parseInt(e.target.value) || 20))}
                    className="w-full rounded border border-white/[0.08] bg-white/[0.02] px-2 py-1.5 text-xs text-foreground focus:border-primary/40 focus:outline-none font-mono"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-muted-foreground/60">Batch Size</label>
                  <input
                    type="number" min={1}
                    value={hyperparams.batch_size}
                    onChange={e => updateHyperparam('batch_size', Math.max(1, parseInt(e.target.value) || 64))}
                    className="w-full rounded border border-white/[0.08] bg-white/[0.02] px-2 py-1.5 text-xs text-foreground focus:border-primary/40 focus:outline-none font-mono"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-muted-foreground/60">优化器</label>
                  <DarkSelect
                    options={OPTIMIZER_OPTIONS}
                    value={hyperparams.optimizer}
                    onChange={(v) => updateHyperparam('optimizer', v)}
                  />
                </div>
                <div>
                  <label className="text-[10px] text-muted-foreground/60">验证集比例</label>
                  <input
                    type="number" step={0.05} min={0.05} max={0.5}
                    value={hyperparams.val_split}
                    onChange={e => updateHyperparam('val_split', Math.min(0.5, Math.max(0.05, parseFloat(e.target.value) || 0.2)))}
                    className="w-full rounded border border-white/[0.08] bg-white/[0.02] px-2 py-1.5 text-xs text-foreground focus:border-primary/40 focus:outline-none font-mono"
                  />
                </div>
                <div>
                  <label className="text-[10px] text-muted-foreground/60">损失函数</label>
                  <DarkSelect
                    options={LOSS_FUNCTIONS}
                    value={hyperparams.loss_function}
                    onChange={(v) => updateHyperparam('loss_function', v)}
                  />
                </div>
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-3">
                <div className="flex items-center gap-2">
                  <label className="text-[10px] text-muted-foreground/60">调度器</label>
                  <DarkSelect
                    options={SCHEDULERS}
                    value={hyperparams.scheduler_type}
                    onChange={(v) => updateHyperparam('scheduler_type', v)}
                    className="w-32"
                  />
                </div>
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="checkbox" checked={hyperparams.early_stopping}
                    onChange={e => updateHyperparam('early_stopping', e.target.checked)}
                    className="h-3.5 w-3.5 rounded border-white/20 bg-white/5 accent-primary"
                  />
                  <span className="text-[10px] text-muted-foreground">早停</span>
                </label>
                {hyperparams.early_stopping && (
                  <div className="flex items-center gap-1">
                    <span className="text-[10px] text-muted-foreground/60">patience:</span>
                    <input
                      type="number" min={1} max={20}
                      value={hyperparams.early_stopping_patience}
                      onChange={e => updateHyperparam('early_stopping_patience', Math.max(1, parseInt(e.target.value) || 5))}
                      className="w-12 rounded border border-white/[0.08] bg-white/[0.02] px-1.5 py-1 text-[10px] text-foreground focus:border-primary/40 focus:outline-none"
                    />
                  </div>
                )}
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <input
                    type="checkbox" checked={hyperparams.use_amp}
                    onChange={e => updateHyperparam('use_amp', e.target.checked)}
                    className="h-3.5 w-3.5 rounded border-white/20 bg-white/5 accent-primary"
                  />
                  <span className="text-[10px] text-muted-foreground">AMP混合精度</span>
                </label>
                <div className="flex items-center gap-1">
                  <span className="text-[10px] text-muted-foreground/60">随机种子:</span>
                  <input
                    type="number" min={0}
                    value={hyperparams.random_seed}
                    onChange={e => updateHyperparam('random_seed', parseInt(e.target.value) || 42)}
                    className="w-14 rounded border border-white/[0.08] bg-white/[0.02] px-1.5 py-1 text-[10px] text-foreground focus:border-primary/40 focus:outline-none"
                  />
                </div>
              </div>
            </section>

            {/* 训练控制按钮 */}
            <section className="rounded-xl border border-white/[0.06] bg-[#0f1119] p-5 lg:col-span-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  {!isTraining ? (
                    <div className="flex items-center gap-3">
                      <button
                        onClick={handleStartTraining}
                        disabled={!selectedDataset || isStarting}
                        className="flex items-center gap-2 rounded-lg bg-emerald-500 px-6 py-2.5 text-sm font-bold text-white shadow-lg shadow-emerald-500/20 transition-all hover:bg-emerald-600 disabled:cursor-not-allowed disabled:opacity-50"
                      >
                        {isStarting ? (
                          <>
                            <div className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                            创建实验中...
                          </>
                        ) : (
                          <>
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                              <path d="M8 5v14l11-7z" />
                            </svg>
                            开始训练
                          </>
                        )}
                      </button>
                      {!selectedDataset && (
                        <span className="text-[10px] text-amber-400/80 flex items-center gap-1">
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg>
                          请先选择数据集
                        </span>
                      )}
                    </div>
                  ) : (
                    <button
                      onClick={handleStopTraining}
                      className="flex items-center gap-2 rounded-lg bg-red-500 px-6 py-2.5 text-sm font-bold text-white shadow-lg shadow-red-500/20 transition-all hover:bg-red-600"
                    >
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                        <rect x="6" y="4" width="4" height="16" /><rect x="14" y="4" width="4" height="16" />
                      </svg>
                      停止训练
                    </button>
                  )}
                  {trainingData.length > 0 && !isTraining && (
                    <button
                      onClick={handleReset}
                      className="flex items-center gap-1.5 rounded-lg border border-white/[0.08] bg-white/[0.02] px-4 py-2.5 text-sm text-muted-foreground transition-all hover:text-foreground"
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                        <path d="M3 3v5h5" />
                      </svg>
                      重置
                    </button>
                  )}
                </div>

                {/* 训练状态指示器 */}
                {isTraining && (
                  <div className="flex items-center gap-4 text-xs">
                    <div className="flex items-center gap-2">
                      <span className="h-2 w-2 rounded-full bg-blue-400 animate-pulse" />
                      <span className="text-muted-foreground">
                        Epoch {currentEpoch}/{totalEpochs > 0 ? totalEpochs : '?'}
                      </span>
                    </div>
                    <span className="text-muted-foreground/60">
                      {elapsedSeconds > 0 ? `${Math.floor(elapsedSeconds / 60)}分${elapsedSeconds % 60}秒` : ''}
                    </span>
                    {experimentId && (
                      <span className="font-mono text-muted-foreground/40">
                        ID: {experimentId.slice(0, 8)}
                      </span>
                    )}
                  </div>
                )}
                {!isTraining && backendStatus === 'completed' && trainingData.length > 0 && (
                  <div className="flex items-center gap-2 rounded-md bg-emerald-400/10 px-3 py-1.5 text-xs text-emerald-400">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><polyline points="20 6 9 17 4 12"/></svg>
                    训练完成！可查看下方曲线或导出结果
                  </div>
                )}
              </div>

              {/* 训练进度条 */}
              {isTraining && totalEpochs > 0 && (
                <div className="mt-3">
                  <div className="flex items-center justify-between mb-1 text-[10px] text-muted-foreground">
                    <span>训练进度</span>
                    <span className="font-mono">{Math.round((currentEpoch / totalEpochs) * 100)}%</span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-white/[0.04]">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-primary to-emerald-400 transition-all duration-500"
                      style={{ width: `${Math.min(100, (currentEpoch / totalEpochs) * 100)}%` }}
                    />
                  </div>
                </div>
              )}

              {trainingError && (
                <div className="mt-3 rounded-md bg-red-500/10 border border-red-500/20 px-3 py-2 text-xs text-red-400">
                  <span className="font-semibold">训练错误：</span>{trainingError}
                </div>
              )}
            </section>

            {/* 模型导入 */}
            <section className="rounded-xl border border-white/[0.06] bg-[#0f1119] p-5">
              <h2 className="mb-3 flex items-center gap-2 text-sm font-bold">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="7 10 12 15 17 10" />
                  <line x1="12" y1="15" x2="12" y2="3" />
                </svg>
                模型导入
              </h2>
              <div className="space-y-2 text-xs">
                <div className="flex gap-2">
                  <DarkSelect
                    options={MODEL_IMPORT_FORMATS}
                    value={modelImportFormat}
                    onChange={setModelImportFormat}
                    className="flex-1"
                  />
                  <button
                    onClick={handleModelImport}
                    disabled={isTraining}
                    className="rounded bg-primary/15 px-3 py-1.5 text-xs text-primary border border-primary/30 hover:bg-primary/20 disabled:opacity-40"
                  >
                    浏览模型
                  </button>
                </div>
                <input
                  ref={modelFileInputRef}
                  type="file"
                  accept=".pt,.pth,.onnx"
                  onChange={handleModelFileChange}
                  className="hidden"
                />
                {importedModel && (
                  <div className="mt-2 rounded-md bg-primary/5 px-3 py-2 text-xs text-primary border border-primary/10">
                    已导入: {importedModel.model_name} ({importedModel.total_params?.toLocaleString()} 参数, {importedModel.layer_count} 层)
                  </div>
                )}
                <p className="text-[10px] text-muted-foreground/50">
                  支持导入已训练的 PyTorch 模型或 ONNX 模型文件，系统将自动解析模型结构并加载权重
                </p>
              </div>
            </section>

            {/* 数据集导入 */}
            <section className="rounded-xl border border-white/[0.06] bg-[#0f1119] p-5">
              <h2 className="mb-3 flex items-center gap-2 text-sm font-bold">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                数据集导入
              </h2>
              <div className="space-y-2 text-xs">
                <div className="flex gap-2">
                  <DarkSelect
                    options={DATASET_IMPORT_FORMATS}
                    value={datasetImportFormat}
                    onChange={setDatasetImportFormat}
                    className="flex-1"
                  />
                  <button
                    onClick={handleDatasetImport}
                    disabled={isTraining}
                    className="rounded bg-primary/15 px-3 py-1.5 text-xs text-primary border border-primary/30 hover:bg-primary/20 disabled:opacity-40"
                  >
                    上传数据集
                  </button>
                </div>
                <input
                  ref={datasetFileInputRef}
                  type="file"
                  accept=".zip,.csv,.npy,.npz,.json"
                  onChange={handleDatasetFileChange}
                  className="hidden"
                />
                {importingDataset && (
                  <div className="mt-2 flex items-center gap-2 text-xs text-muted-foreground">
                    <div className="h-3 w-3 animate-spin rounded-full border border-primary border-t-transparent" />
                    正在上传数据集...
                  </div>
                )}
                <p className="text-[10px] text-muted-foreground/50">
                  支持导入图像数据集(.zip)、CSV表格、NumPy数组(.npy/.npz)和JSON格式数据，系统将自动解析并预处理
                </p>
              </div>
            </section>

            {/* 训练曲线可视化 */}
            <section className="rounded-xl border border-white/[0.06] bg-[#0f1119] p-5 lg:col-span-2">
              <TrainingCurvesPanel
                visibleData={visibleData}
                currentStep={currentStep}
                colors={DEFAULT_COLORS}
                hasActiveTraining={hasActiveTraining}
                onReset={handleReset}
              />
            </section>

            {/* 训练日志 */}
            {trainingLogs.length > 0 && (
              <section className="rounded-xl border border-white/[0.06] bg-[#0f1119] p-5 lg:col-span-2">
                <h2 className="mb-3 flex items-center gap-2 text-sm font-bold">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                    <path d="M14 2v6h6" />
                  </svg>
                  训练日志
                  <span className="text-[10px] text-muted-foreground">({trainingLogs.length} 条)</span>
                </h2>
                <div className="max-h-[300px] overflow-y-auto rounded-lg border border-white/[0.04] bg-[#0c0e17] p-3">
                  <div className="space-y-1 font-mono text-xs">
                    {trainingLogs.slice(-100).map((log, i) => {
                      const isError = /error|fail|失败|错误/i.test(log);
                      const isSuccess = /完成|best|新最佳|saved/i.test(log);
                      return (
                        <div key={i} className={`flex gap-2 border-b border-white/[0.02] py-1 last:border-0 ${
                          isError ? 'text-red-400' : isSuccess ? 'text-emerald-400' : 'text-muted-foreground/80'
                        }`}>
                          <span className="shrink-0 text-muted-foreground/40">{String(i + 1).padStart(3, '0')}</span>
                          <span>{log}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </section>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}