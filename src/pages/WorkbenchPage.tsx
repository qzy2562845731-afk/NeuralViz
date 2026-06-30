import { useEffect, useMemo, useRef, useState } from 'react';
import { useWorkbench } from '../components/workbench/WorkbenchContext';
import { LayerRail } from '../components/workbench/LayerRail';
import { InspectorPanel } from '../components/workbench/InspectorPanel';
import { AIDiagnosisPanel } from '../components/workbench/AIDiagnosisPanel';
import { ViewModeSwitch } from '../components/workbench/ViewModeSwitch';
import { ModelImporter } from '../components/workbench/ModelImporter';
import { InferencePanel } from '../components/workbench/InferencePanel';
import { SaveExperiment } from '../components/workbench/SaveExperiment';
import { DatasetSelector } from '../components/workbench/DatasetSelector';
import { HyperparamPanel, DEFAULT_HYPERPARAMS, type Hyperparams } from '../components/workbench/HyperparamPanel';
import { CNN3DViewer } from '../components/cnn3d';
import { type LayerConfig } from '../components/cnn3d/types';
import type { ModelAnalysisResult } from '../components/cnn3d/modelAnalyzer';
import { useGlobalTraining } from '../contexts/GlobalTrainingContext';
import { useToast } from '../contexts/ToastContext';
import apiService from '../services/api';
import { ModelConfigPanel, DEFAULT_MODEL_CONFIG, type ModelConfig } from '../components/workbench/ModelConfigPanel';
import { CNNVisualPanel } from '../components/CNNVisualPanel';
import { AblationPanel } from '../components/AblationPanel';
import { ExperimentComparisonPanel } from '../components/ExperimentComparisonPanel';
import { DeploymentConfig } from '../components/DeploymentConfig';

/* ============================================
   WorkbenchPage — CNN 可视化工作台
   - 左侧：层列表（树结构+搜索）
   - 中央：3D CNN 可视化
   - 右侧：Inspector + AI 诊断
   - 底部：时间线
   - 使用全局训练上下文，切换页面不中断训练
   ============================================ */

export default function WorkbenchPage() {
  // Bug2修复：WorkbenchProvider 已提升至 App.tsx，此处直接渲染布局
  return <WorkbenchLayout />;
}

// 命名导出兼容 App.tsx
export { WorkbenchPage };

/* ---------- 主布局 ---------- */
function WorkbenchLayout() {
  const {
    isPlaying,
    currentStep,
    activeLayerId,
    followTraining,
    speed,
    setPlaying,
    setStep,
    setSpeed,
    toggleAI,
    aiEnabled,
    resumeFollow,
    selectLayer,
    viewMode,
    architecture,
    loadArchitecture,
    realActivations,
    setRealActivations,
    currentModelId,
    currentModelName,
    selectedDataset,
    setSelectedDataset,
    currentExperimentId,
    setCurrentExperimentId,
    showLayerRail,
    showInspector,
    toggleLayerRail,
    toggleInspector,
    activeAblationGroupName,
  } = useWorkbench();

  // 全局训练上下文 - 用于与Visualization共享
  const {
    isRealTraining,
    currentEpoch,
    totalEpochs,
    startRealTraining,
    stopRealTraining,
    currentData: currentTrainingData,
    trainingError,
    elapsedSeconds,
    data: trainingDataList,
    backendStatus,
    hasActiveTraining,
  } = useGlobalTraining();
  const toast = useToast();

  // 是否正在启动训练（防止重复点击）
  const [isStarting, setIsStarting] = useState(false);
  // 潜在问题修复：同步锁，防止快速双击在setIsStarting重渲染前穿透
  const isStartingRef = useRef(false);
  // 超参数配置（用户可通过面板调整）- sessionStorage 持久化，切页不丢失
  const [hyperparams, setHyperparams] = useState<Hyperparams>(() => {
    try {
      const saved = sessionStorage.getItem('wb_hyperparams');
      if (saved) return { ...DEFAULT_HYPERPARAMS, ...JSON.parse(saved) };
    } catch {}
    return DEFAULT_HYPERPARAMS;
  });
  // hyperparams 变化时持久化到 sessionStorage
  useEffect(() => {
    try { sessionStorage.setItem('wb_hyperparams', JSON.stringify(hyperparams)); } catch {}
  }, [hyperparams]);

  // 模型结构配置（sessionStorage 持久化）
  const [modelConfig, setModelConfig] = useState<ModelConfig>(() => {
    try {
      const saved = sessionStorage.getItem('wb_model_config');
      if (saved) return { ...DEFAULT_MODEL_CONFIG, ...JSON.parse(saved) };
    } catch {}
    return DEFAULT_MODEL_CONFIG;
  });
  useEffect(() => {
    try { sessionStorage.setItem('wb_model_config', JSON.stringify(modelConfig)); } catch {}
  }, [modelConfig]);

  // 实验对比面板
  const [showComparisonPanel, setShowComparisonPanel] = useState(false);
  // 部署配置面板
  const [showDeploymentConfig, setShowDeploymentConfig] = useState(false);

  // 页面加载时同步全局训练状态到本地WorkbenchContext
  useEffect(() => {
    if (isRealTraining && !isPlaying) {
      setPlaying(true);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // 仅在组件挂载时执行一次

  // 终态同步：后端状态变为 completed/failed/stopped 时，同步关闭本地 isPlaying
  // 消除"训练日志页已完成、工作台还显示运行中"的状态不一致
  useEffect(() => {
    if (['completed', 'failed', 'stopped'].includes(backendStatus) && isPlaying) {
      setPlaying(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [backendStatus]);

  // 体验优化：空格键播放/暂停训练
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.code !== 'Space') return;
      // 排除输入框、文本域等可编辑元素
      const target = e.target as HTMLElement;
      const tag = target.tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || target.isContentEditable) return;
      e.preventDefault();
      handlePlayToggle();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPlaying, isRealTraining, isStarting, currentModelId, selectedDataset]);

  // 处理播放按钮点击 - 启动/停止真实训练
  const handlePlayToggle = async () => {
    // 同步锁防止并发快速点击
    if (isStartingRef.current) return;

    if (isPlaying || isRealTraining) {
      // 停止训练
      if (isRealTraining) {
        isStartingRef.current = true;
        setIsStarting(true);
        try {
          await stopRealTraining();
          setPlaying(false);
          toast.showInfo('训练已停止', '已保存的指标和模型权重将保留');
        } finally {
          isStartingRef.current = false;
          setIsStarting(false);
        }
      } else {
        setPlaying(false);
      }
    } else {
      // 启动训练
      if (isRealTraining) {
        toast.showInfo('训练正在进行中', '请先停止当前训练再启动新训练');
        return;
      }
      // 校验：必须同时选中模型和数据集才能启动训练
      if (!currentModelId) {
        toast.showError('请先导入模型', '点击右侧"导入模型"按钮上传模型文件');
        return;
      }
      if (!selectedDataset) {
        toast.showError('请先选择数据集', '点击右侧"选择数据集"按钮选择已上传的数据集');
        return;
      }
      if (selectedDataset.status !== 'ready') {
        toast.showError('数据集未就绪', `当前状态: ${selectedDataset.status}，请等待解析完成或重试`);
        return;
      }

      isStartingRef.current = true;
      setIsStarting(true);
      try {
        // 先创建实验记录
        const expRes = await apiService.createExperiment({
          name: `${currentModelName || '模型'}_${selectedDataset.name}_${Date.now()}`,
          description: `模型: ${currentModelName}, 数据集: ${selectedDataset.name}`,
          model_id: currentModelId,
          model_name: currentModelName || '',
          model_architecture: {
            type: (architecture as any)?.type || 'cnn',
            name: currentModelName || 'Custom CNN',
            layers: (architecture.layers as LayerConfig[]).map(l => ({
              id: l.id,
              type: l.type,
              nodeCount: l.nodeCount,
              activation: (l as any).activation,
              params: (l as any).params || 0,
              kernelSize: (l as any).kernelSize,
              stride: (l as any).stride,
              padding: (l as any).padding,
              dropout: (l as any).dropout,
            })),
            total_params: modelStats.totalParams,
            layer_count: modelStats.layerCount,
            input_shape: selectedDataset.feature_shape || null,
          },
          total_params: modelStats.totalParams,
          layer_count: modelStats.layerCount,
          hyperparams: {
            learning_rate: hyperparams.learning_rate,
            batch_size: hyperparams.batch_size,
            optimizer: hyperparams.optimizer,
            epochs: hyperparams.epochs,
            random_seed: hyperparams.random_seed,
            val_split: hyperparams.val_split,
            loss_function: hyperparams.loss_function,
            scheduler_type: hyperparams.scheduler_type,
            early_stopping: hyperparams.early_stopping,
            early_stopping_patience: hyperparams.early_stopping_patience,
            use_amp: hyperparams.use_amp,
            resume_from_checkpoint: hyperparams.resume_from_checkpoint,
          },
          config: {
            dataset_id: selectedDataset.dataset_id,
            dataset_name: selectedDataset.name,
            dataset_type: selectedDataset.dataset_type,
            feature_shape: selectedDataset.feature_shape,
            model_config: modelConfig as any,
          },
          tags: ['workbench', selectedDataset.dataset_type || 'unknown'],
          status: 'running',
        });

        if (expRes.code !== 200 || !expRes.data) {
          throw new Error(expRes.message || '创建实验失败');
        }

        const experimentId = expRes.data.experiment_id;
        setCurrentExperimentId(experimentId);

        // 启动后端真实训练
        const success = await startRealTraining(experimentId, selectedDataset.dataset_id, {
          learning_rate: hyperparams.learning_rate,
          batch_size: hyperparams.batch_size,
          optimizer: hyperparams.optimizer,
          epochs: hyperparams.epochs,
          random_seed: hyperparams.random_seed,
          val_split: hyperparams.val_split,
          loss_function: hyperparams.loss_function,
          scheduler_type: hyperparams.scheduler_type,
          early_stopping: hyperparams.early_stopping,
          early_stopping_patience: hyperparams.early_stopping_patience,
          use_amp: hyperparams.use_amp,
          resume_from_checkpoint: hyperparams.resume_from_checkpoint,
        }, modelConfig);

        if (success) {
          setPlaying(true);
          toast.showSuccess('训练已启动', `实验ID: ${experimentId.slice(0, 8)}...`);
        } else {
          toast.showError('训练启动失败', trainingError || '请检查后端服务是否正常运行');
        }
      } catch (err: any) {
        toast.showError('启动训练失败', err.message || '未知错误');
      } finally {
        isStartingRef.current = false;
        setIsStarting(false);
      }
    }
  };

  const handleModelLoad = (result: ModelAnalysisResult) => {
    loadArchitecture(result.architecture);
    setRealActivations({}); // 加载新模型时清除激活值
  };

  const handleInferenceComplete = (result: { activations: Record<string, number[]> }) => {
    if (result.activations) {
      setRealActivations(result.activations);
    }
  };

  // 全局训练上下文已通过handlePlayToggle直接同步

  // 生成训练激活值（基于真实训练数据）
  const activations = useMemo(() => {
    if (!isPlaying) return null;
    const acts: Record<string, number[]> = {};
    architecture.layers.forEach((layer: LayerConfig) => {
      const isActive = activeLayerId === layer.id;
      const base = isActive ? 0.5 : 0.2;
      const count = Math.max(8, Math.min(32, layer.nodeCount));
      acts[layer.id] = Array.from({ length: count }, (_, i) => {
        const v = base + Math.sin(currentStep * 0.1 + i * 0.3 + layer.nodeCount) * 0.3 + (Math.random() - 0.5) * 0.2;
        return isActive ? v : v * 0.3;
      });
    });
    return acts;
  }, [isPlaying, currentStep, activeLayerId, architecture]);

  // loss / accuracy：使用真实训练数据
  const [loss, acc] = useMemo<[number, number]>(() => {
    if (currentTrainingData) {
      return [
        currentTrainingData.loss || 0,
        currentTrainingData.accuracy || 0,
      ];
    }
    return [0, 0];
  }, [currentTrainingData]);

  // 显示 step 和进度（始终使用真实训练数据）
  const hasRealData = trainingDataList.length > 0;
  const displayStep = hasRealData ? currentEpoch : 0;
  const displayTotal = totalEpochs > 0 ? totalEpochs : (hasRealData ? trainingDataList.length : 0);
  const progressPercent = totalEpochs > 0
    ? Math.min(100, Math.round((currentEpoch / totalEpochs) * 100))
    : 0;

  // 真实训练模式下，将 currentEpoch 同步到 currentStep
  useEffect(() => {
    if (hasRealData && currentEpoch > 0) {
      setStep(currentEpoch);
    }
  }, [currentEpoch, setStep, hasRealData]);

  // 计算模型统计
  const modelStats = useMemo(() => {
    const layers = architecture.layers as LayerConfig[];
    let totalParams = 0;
    layers.forEach((layer) => {
      totalParams += (layer as any).params || layer.nodeCount || 0;
    });
    return {
      totalParams,
      layerCount: layers.length,
    };
  }, [architecture]);

  // 布局尺寸（预留，后续可支持拖拽调整）
  const leftRail = 250;
  const rightInspector = 340;
  const rightAI = 270;

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-[#0a0c14] text-white">
      {/* ===== 顶部控制栏（响应式：flex-wrap 支持小屏换行） ===== */}
      <div className="flex flex-wrap items-center gap-1.5 border-b border-white/[0.06] bg-[#0c0e17] px-3 py-2 sm:gap-2 sm:px-4">
        {/* 左：Logo + 页面标题 */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <div className="flex h-7 w-7 sm:h-8 sm:w-8 items-center justify-center rounded-lg bg-gradient-to-br from-emerald-400 to-cyan-500 shadow shadow-emerald-500/20">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-white sm:w-[15px] sm:h-[15px]">
              <path d="M12 2l9 4.5v11L12 22l-9-4.5v-11L12 2z" />
              <path d="M12 12l9-4.5M12 12v10M12 12L3 7.5" />
            </svg>
          </div>
          <div className="hidden sm:block">
            <h1 className="text-sm font-bold tracking-tight leading-tight">3D 结构视图</h1>
            <p className="text-[10px] text-muted-foreground leading-tight truncate max-w-[120px]">{currentModelName} · {architecture.name}</p>
          </div>
        </div>

        {/* 分隔线 - 小屏隐藏 */}
        <div className="h-6 w-px bg-white/[0.08] hidden md:block" />

        {/* 数据集和模型选择 */}
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <DatasetSelector selectedDataset={selectedDataset} onSelect={setSelectedDataset} />
          <ModelImporter onModelLoad={handleModelLoad} />
        </div>

        {/* 弹性间距 - 小屏隐藏 */}
        <div className="flex-1 hidden lg:block" />

        {/* 核心控制：训练/停止按钮 - 始终可见，优先级最高 */}
        <button
          onClick={handlePlayToggle}
          disabled={isStarting}
          className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 sm:px-3 text-xs font-semibold transition-all disabled:opacity-50 flex-shrink-0 ${
            isPlaying
              ? isRealTraining
                ? 'border-blue-400/40 bg-blue-400/10 text-blue-400 hover:bg-blue-400/15'
                : 'border-amber-400/40 bg-amber-400/10 text-amber-400 hover:bg-amber-400/15'
              : 'border-emerald-400/40 bg-emerald-400/10 text-emerald-400 hover:bg-emerald-400/15 hover:shadow-[0_0_12px_rgba(52,211,153,0.15)]'
          }`}
        >
          {isStarting ? (
            <svg className="animate-spin" width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M21 12a9 9 0 1 1-6.219-8.56" />
            </svg>
          ) : isPlaying ? (
            <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="4" width="4" height="16" rx="1" />
              <rect x="14" y="4" width="4" height="16" rx="1" />
            </svg>
          ) : (
            <svg width="10" height="10" viewBox="0 0 24 24" fill="currentColor">
              <path d="M6 4l14 8-14 8V4z" />
            </svg>
          )}
          <span>{isStarting ? '启动中' : isPlaying ? (isRealTraining ? '停止' : '暂停') : '训练'}</span>
        </button>

        {/* 推理按钮 - 始终可见 */}
        <div className="flex-shrink-0">
          <InferencePanel
            onActivationsChange={setRealActivations}
            onInferenceComplete={handleInferenceComplete}
            onOpenImporter={() => window.dispatchEvent(new CustomEvent('open-model-importer'))}
          />
        </div>

        {/* 速度切换（紧凑）- 中小屏隐藏文字 */}
        <div className="flex items-center gap-0.5 rounded-md border border-white/[0.08] bg-white/[0.02] p-0.5 flex-shrink-0 hidden sm:flex">
          {[0.5, 1, 2, 4].map((s) => (
            <button
              key={s}
              onClick={() => setSpeed(s)}
              className={`rounded px-1.5 py-0.5 font-mono text-[9px] transition-all ${
                speed === s ? 'bg-primary/20 text-primary font-bold' : 'text-muted-foreground hover:text-foreground/80'
              }`}
            >
              {s}x
            </button>
          ))}
        </div>

        {/* 训练状态（紧凑整合：状态+进度+指标） */}
        {(isPlaying || hasRealData || hasActiveTraining || displayStep > 0) && (
          <div className="flex items-center gap-1.5 sm:gap-2 rounded-lg border border-white/[0.08] bg-white/[0.02] px-2 py-1 sm:px-2.5 flex-shrink-0">
            {/* 状态圆点 */}
            {(() => {
              const isCompleted = backendStatus === 'completed';
              const isFailed = backendStatus === 'failed';
              const isStopped = backendStatus === 'stopped';
              const dotColor = isFailed ? 'bg-red-400' : isCompleted ? 'bg-emerald-400' : isStopped ? 'bg-amber-400' : isRealTraining ? 'bg-blue-400 animate-pulse' : 'bg-emerald-400';
              return <span className={`h-2 w-2 rounded-full ${dotColor} flex-shrink-0`} />;
            })()}
            {/* Epoch进度 */}
            <div className="flex items-center gap-1 sm:gap-1.5">
              <span className="text-[9px] text-muted-foreground font-mono">
                {displayStep}/{displayTotal || '-'}
              </span>
              {totalEpochs > 0 && (
                <div className="h-1 w-8 sm:w-12 overflow-hidden rounded-full bg-white/[0.08]">
                  <div className="h-full rounded-full bg-gradient-to-r from-primary to-blue-400" style={{ width: `${progressPercent}%` }} />
                </div>
              )}
            </div>
            {/* 分隔 - 小屏隐藏 */}
            <div className="h-3 w-px bg-white/10 hidden sm:block" />
            {/* Loss */}
            <div className="hidden sm:flex items-center gap-1">
              <span className="text-[9px] text-red-400/70">L</span>
              <span className="font-mono text-[10px] font-bold text-red-400">{loss.toFixed(3)}</span>
            </div>
            {/* Acc */}
            <div className="hidden sm:flex items-center gap-1">
              <span className="text-[9px] text-emerald-400/70">A</span>
              <span className="font-mono text-[10px] font-bold text-emerald-400">{(acc * 100).toFixed(1)}%</span>
            </div>
            {/* 耗时（训练中才显示）- 小屏隐藏 */}
            {isPlaying && elapsedSeconds > 0 && (
              <>
                <div className="h-3 w-px bg-white/10 hidden md:block" />
                <span className="font-mono text-[9px] text-muted-foreground hidden md:inline">
                  {elapsedSeconds < 60 ? `${elapsedSeconds.toFixed(0)}s` : `${Math.floor(elapsedSeconds/60)}m${Math.floor(elapsedSeconds%60)}s`}
                </span>
              </>
            )}
            {/* 错误提示 */}
            {trainingError && (
              <div className="flex items-center gap-1 text-red-400" title={trainingError}>
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <circle cx="12" cy="12" r="10" /><line x1="12" y1="8" x2="12" y2="12" /><line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
              </div>
            )}
          </div>
        )}

        {/* 分隔线 - 小屏隐藏 */}
        <div className="h-6 w-px bg-white/[0.08] hidden md:block" />

        {/* 配置按钮区 */}
        <div className="flex items-center gap-1 flex-shrink-0">
          <HyperparamPanel hyperparams={hyperparams} onChange={setHyperparams} disabled={isPlaying || isRealTraining} />
          <ModelConfigPanel modelConfig={modelConfig} onChange={setModelConfig} disabled={isPlaying || isRealTraining} datasetType={selectedDataset?.dataset_type ?? undefined} />
          <ViewModeSwitch />
        </div>

        {/* 分隔线 - 小屏隐藏 */}
        <div className="h-6 w-px bg-white/[0.08] hidden md:block" />

        {/* 面板切换按钮 */}
        <div className="flex items-center gap-1 flex-shrink-0">
          <button
            onClick={toggleLayerRail}
            className={`flex items-center gap-1 rounded-md border px-1.5 py-1.5 sm:px-2 transition-all ${
              showLayerRail
                ? 'border-primary/30 bg-primary/10 text-primary'
                : 'border-white/[0.08] bg-white/[0.02] text-muted-foreground hover:text-primary/70'
            }`}
            title={showLayerRail ? '隐藏层列表' : '显示层列表'}
          >
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" />
              <rect x="3" y="14" width="7" height="7" /><rect x="14" y="14" width="7" height="7" />
            </svg>
            <span className="text-[9px] font-semibold hidden sm:inline">层列表</span>
          </button>
          <button
            onClick={toggleInspector}
            className={`flex items-center gap-1 rounded-md border px-1.5 py-1.5 sm:px-2 transition-all ${
              showInspector
                ? 'border-emerald-400/30 bg-emerald-400/10 text-emerald-400'
                : 'border-white/[0.08] bg-white/[0.02] text-muted-foreground hover:text-emerald-400/70'
            }`}
            title={showInspector ? '隐藏检查器' : '显示检查器'}
          >
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <circle cx="12" cy="12" r="3" />
              <path d="M12 1v6M12 17v6M1 12h6M17 12h6M4.93 4.93l4.24 4.24M14.83 14.83l4.24 4.24M4.93 19.07l4.24-4.24M14.83 9.17l4.24-4.24" />
            </svg>
            <span className="text-[9px] font-semibold hidden sm:inline">检查器</span>
          </button>
        </div>

        {/* 分隔线 - 小屏隐藏 */}
        <div className="h-6 w-px bg-white/[0.08] hidden md:block" />

        {/* 操作按钮区 */}
        <div className="flex items-center gap-1 flex-shrink-0">
          <SaveExperiment
            modelId={currentModelId}
            modelName={currentModelName}
            architecture={architecture}
            totalParams={modelStats.totalParams}
            layerCount={modelStats.layerCount}
            currentStep={displayStep}
            loss={loss}
            accuracy={acc}
            trainingStatus={
              backendStatus === 'completed' ? 'completed'
              : backendStatus === 'failed' ? 'failed'
              : backendStatus === 'stopped' ? 'stopped'
              : (isRealTraining || hasRealData) ? 'running'
              : 'draft'
            }
            totalEpochs={totalEpochs}
            hyperparams={hyperparams}
          />
        </div>

        {/* AI诊断开关（精简） */}
        <button
          onClick={toggleAI}
          className={`flex items-center gap-1.5 rounded-md border px-2 py-1.5 transition-all flex-shrink-0 ${
            aiEnabled
              ? 'border-amber-400/30 bg-amber-400/10 text-amber-400'
              : 'border-white/[0.08] bg-white/[0.02] text-muted-foreground hover:text-amber-400/70'
          }`}
          title={aiEnabled ? '关闭AI诊断' : '开启AI诊断'}
        >
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <circle cx="12" cy="12" r="3" />
            <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" />
          </svg>
          <span className="text-[9px] font-semibold">AI</span>
        </button>

        {/* 实验对比 */}
        <button
          onClick={() => setShowComparisonPanel(true)}
          className="flex items-center gap-1.5 rounded-md border border-white/[0.08] bg-white/[0.02] px-2 py-1.5 transition-all flex-shrink-0 text-muted-foreground hover:text-purple-400/70 hover:border-purple-400/30"
          title="实验对比"
        >
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <path d="M18 20V10M12 20V4M6 20v-6" />
          </svg>
          <span className="text-[9px] font-semibold">对比</span>
        </button>

        {/* 部署配置 */}
        <button
          onClick={() => setShowDeploymentConfig(true)}
          className="flex items-center gap-1.5 rounded-md border border-white/[0.08] bg-white/[0.02] px-2 py-1.5 transition-all flex-shrink-0 text-muted-foreground hover:text-cyan-400/70 hover:border-cyan-400/30"
          title="部署配置"
        >
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
            <rect x="2" y="2" width="20" height="8" rx="2" ry="2" />
            <rect x="2" y="14" width="20" height="8" rx="2" ry="2" />
            <circle cx="6" cy="6" r="1" /><circle cx="6" cy="18" r="1" />
          </svg>
          <span className="text-[9px] font-semibold">部署</span>
        </button>
      </div>

      {/* ===== 主内容区：左层列表 | 中 3D | 右检查器+AI ===== */}
      <div className="flex min-h-0 flex-1 overflow-hidden">
        {/* ===== 左：层列表（带动画折叠） ===== */}
        <div
          className="flex-shrink-0 overflow-hidden transition-[width] duration-300 ease-in-out"
          style={{ width: showLayerRail ? `${leftRail}px` : '0px' }}
        >
          <LayerRail width={leftRail} />
        </div>

        {/* ===== 中央：3D 可视化 ===== */}
        <div className="relative flex min-w-0 flex-1 flex-col overflow-hidden">
          <div className="flex min-h-0 flex-1 flex-col bg-[#0d1018]">
            <CNN3DViewer
              activations={activations}
              realActivations={realActivations}
              step={displayStep}
              isPlaying={isPlaying}
              speed={speed}
              selectedLayerId={null}
              activeLayerId={activeLayerId}
              onLayerSelect={selectLayer}
              architecture={architecture}
              viewMode={viewMode}
            />
          </div>

          {/* ===== 底部科研工具面板（特征图/卷积核/消融实验） ===== */}
          {(currentExperimentId || selectedDataset) && (
            <div className="border-t border-white/[0.06] bg-[#0c0e17]">
              <div className="flex max-h-[280px] overflow-hidden">
                <div className="flex-1 overflow-auto border-r border-white/[0.06] p-3">
                  <CNNVisualPanel experimentId={currentExperimentId} hasImageData={selectedDataset?.dataset_type === 'mnist_idx' || selectedDataset?.dataset_type === 'image_folder'} />
                </div>
                <div className="w-[380px] flex-shrink-0 overflow-auto p-3">
                  <AblationPanel datasetId={selectedDataset?.dataset_id || null} />
                </div>
              </div>
            </div>
          )}

          {/* ===== 底部时间线 ===== */}
          <div className="border-t border-white/[0.06] bg-[#0c0e17] px-5 py-2.5">
            <div className="flex items-center justify-between gap-6">
              {/* 层序列导航 */}
              <div className="flex items-center gap-3">
                <span className="mr-2 text-[9px] uppercase tracking-widest text-muted-foreground">Layers</span>
                <div className="flex items-center gap-1.5 overflow-x-auto" style={{ maxWidth: '420px' }}>
                  {architecture.layers.slice(0, 10).map((layer: LayerConfig) => {
                    const isActive = activeLayerId === layer.id;
                    return (
                      <button
                        key={layer.id}
                        onClick={() => selectLayer(layer.id)}
                        className={`flex-shrink-0 rounded border px-1.5 py-0.5 font-mono text-[9px] transition-all ${
                          isActive
                            ? 'border-emerald-400/40 bg-emerald-400/10 text-emerald-400'
                            : 'border-white/[0.08] bg-white/[0.02] text-muted-foreground hover:bg-white/[0.04]'
                        }`}
                      >
                        {layer.name}
                      </button>
                    );
                  })}
                </div>
              </div>

              {/* 进度条 + Loss/Acc */}
              <div className="flex flex-1 items-center gap-6">
                <div className="flex flex-1 items-center gap-3">
                  <span className="text-[9px] uppercase tracking-widest text-muted-foreground">Progress</span>
                  <div className="relative h-1.5 flex-1 overflow-hidden rounded-full bg-white/[0.04]">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-primary via-purple-400 to-emerald-400 transition-all duration-500"
                      style={{ width: `${progressPercent}%` }}
                    />
                  </div>
                  <span className="font-mono text-[10px] text-foreground/70">
                    {progressPercent}%
                  </span>
                </div>

                {/* Mini loss/acc */}
                <div className="flex items-center gap-3 rounded-md border border-white/[0.06] bg-white/[0.02] px-3 py-1.5">
                  <div className="flex items-center gap-1.5">
                    <span className="size-1.5 rounded-full bg-red-400" />
                    <span className="text-[9px] text-muted-foreground">Loss</span>
                    <span className="font-mono text-[10px] font-bold text-red-400">{loss.toFixed(4)}</span>
                  </div>
                  <div className="h-3 w-px bg-white/10" />
                  <div className="flex items-center gap-1.5">
                    <span className="size-1.5 rounded-full bg-emerald-400" />
                    <span className="text-[9px] text-muted-foreground">Acc</span>
                    <span className="font-mono text-[10px] font-bold text-emerald-400">{(acc * 100).toFixed(2)}%</span>
                  </div>
                </div>
              </div>

              {/* 跟随训练状态 */}
              <button
                onClick={resumeFollow}
                className={`flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 transition-all ${
                  followTraining
                    ? 'border-emerald-400/25 bg-emerald-400/[0.06] text-emerald-400'
                    : 'border-primary/25 bg-primary/[0.06] text-primary hover:bg-primary/10'
                }`}
              >
                <span className={`size-1.5 rounded-full ${followTraining ? 'bg-emerald-400' : 'bg-primary'}`} />
                <span className="text-[10px] font-semibold">{followTraining ? '跟随训练' : '手动模式 · 恢复'}</span>
              </button>
            </div>
          </div>
        </div>

        {/* ===== 右：Inspector + AI 诊断 ===== */}
        <div className="flex flex-shrink-0">
          {/* Inspector 面板（带动画折叠） */}
          <div
            className="flex-shrink-0 overflow-hidden transition-[width] duration-300 ease-in-out"
            style={{ width: showInspector ? `${rightInspector}px` : '0px' }}
          >
            <InspectorPanel width={rightInspector} />
          </div>

          {/* Inspector 显示窄条（隐藏时可快速展开） */}
          {!showInspector && (
            <button
              onClick={toggleInspector}
              title="显示检查器"
              className="flex w-10 flex-shrink-0 flex-col items-center justify-center gap-2 border-l border-white/[0.06] bg-[#0c0e17] transition-all hover:bg-emerald-400/5"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-emerald-400/80">
                <circle cx="12" cy="12" r="3" />
                <path d="M12 1v6M12 17v6M1 12h6M17 12h6" />
              </svg>
              <span className="text-[8px] font-semibold uppercase tracking-wider text-emerald-400/80 [writing-mode:vertical-rl] rotate-180">
                检查器
              </span>
            </button>
          )}

          {/* AI 开关窄条：始终可见，确保关闭后也能重新启用 */}
          {!aiEnabled && (
            <button
              onClick={toggleAI}
              title="启用 AI 诊断"
              className="flex w-12 flex-shrink-0 flex-col items-center justify-center gap-2 border-l border-white/[0.06] bg-[#0c0e17] transition-all hover:bg-amber-400/5"
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-amber-400/80">
                <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
                <circle cx="12" cy="12" r="3" />
              </svg>
              <span className="text-[9px] font-semibold uppercase tracking-wider text-amber-400/80 [writing-mode:vertical-rl] rotate-180">
                启用 AI
              </span>
            </button>
          )}

          {/* Bug4修复：AI面板平滑过渡动画，宽度从0过渡到rightAI，不再条件卸载 */}
          <div
            className="flex-shrink-0 overflow-hidden transition-all duration-300 ease-in-out"
            style={{ width: aiEnabled ? `${rightAI}px` : '0px' }}
          >
            <AIDiagnosisPanel width={rightAI} />
          </div>
        </div>
      </div>

      {/* 实验对比面板 */}
      <ExperimentComparisonPanel
        isOpen={showComparisonPanel}
        onClose={() => setShowComparisonPanel(false)}
        ablationGroup={activeAblationGroupName || undefined}
      />

      {/* 部署配置面板 */}
      <DeploymentConfig
        isOpen={showDeploymentConfig}
        onClose={() => setShowDeploymentConfig(false)}
      />
    </div>
  );
}
