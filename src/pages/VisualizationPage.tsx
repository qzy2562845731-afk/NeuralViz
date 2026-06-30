import { useState, useMemo } from 'react';
import { useGlobalTraining } from '../contexts/GlobalTrainingContext';
import { useToast } from '../contexts/ToastContext';
import { useColorConfig } from '../hooks/useColorConfig';
import { TrainingCurvesPanel } from '../components/TrainingCurvesPanel';
import { ConfusionMatrix } from '../components/ConfusionMatrix';
import { ActivationHistogram } from '../components/ActivationHistogram';
import { GradientWeightChart } from '../components/GradientWeightChart';
import { FeatureMapViewer } from '../components/FeatureMapViewer';
import { PredictionDistribution } from '../components/PredictionDistribution';
import { ModelMetricsPanel } from '../components/ModelMetricsPanel';

/* ============================================
   VisualizationPage — 数据可视化中心
   - 使用全局训练上下文，支持跨页面实时可视化
   - 展示混淆矩阵、热力图、训练曲线、激活分布、梯度权重、预测分布
   ============================================ */

export default function VisualizationPage() {
  const { colors } = useColorConfig();
  const {
    data: trainingData,
    currentStep,
    maxStep,
    displayTotalSteps,
    currentData,
    goToStep,
    isPlaying,
    play,
    pause,
    reset,
    hasActiveTraining,
    isRealTraining,
    backendStatus,
  } = useGlobalTraining();
  const toast = useToast();

  const [showHeatmapOverlay, setShowHeatmapOverlay] = useState(false);

  // 训练运行中状态（用于状态指示与按钮禁用）
  const isTrainingRunning = isRealTraining && backendStatus === 'running';

  const handleStepChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    goToStep(Number(e.target.value));
  };

  // 重置画布：清空当前数据
  const handleResetCanvas = () => {
    reset();
    toast.showSuccess('画布已重置', '历史训练数据已清空');
  };

  // 计算可见数据（用于图表）
  const visibleData = useMemo(() => {
    if (trainingData.length === 0) return [];
    return trainingData.slice(0, currentStep + 1);
  }, [trainingData, currentStep]);

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
            <h1 className="text-[13px] font-bold tracking-tight">数据可视化中心</h1>
            <p className="text-[10px] text-muted-foreground">混淆矩阵 · 热力图 · 训练曲线 · 激活分布</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {isRealTraining && (
            <span
              className={`flex h-8 items-center gap-1.5 rounded-lg border px-3 text-[11px] font-semibold ${
                isTrainingRunning
                  ? 'border-emerald-400/30 bg-emerald-400/10 text-emerald-400'
                  : 'border-amber-400/30 bg-amber-400/10 text-amber-400'
              }`}
            >
              <span
                className={`size-2 rounded-full ${
                  isTrainingRunning ? 'animate-pulse bg-emerald-400' : 'bg-amber-400'
                }`}
              />
              {isTrainingRunning ? '训练中' : backendStatus}
            </span>
          )}
          <button
            onClick={() => isPlaying ? pause() : play()}
            disabled={isRealTraining}
            className={`flex h-8 items-center gap-1.5 rounded-lg border px-3 text-[11px] font-semibold transition-all ${
              isRealTraining
                ? 'cursor-not-allowed border-white/[0.04] bg-white/[0.01] text-white/30'
                : isPlaying
                  ? 'border-emerald-400/30 bg-emerald-400/10 text-emerald-400 hover:bg-emerald-400/15'
                  : 'border-white/[0.08] bg-white/[0.02] text-foreground/85 hover:bg-white/[0.05]'
            }`}
          >
            {isPlaying ? '暂停' : '播放'}
          </button>
          <button
            onClick={reset}
            className="flex h-8 items-center gap-1.5 rounded-lg border border-white/[0.08] bg-white/[0.02] px-3 text-[11px] font-semibold text-foreground/85 transition-all hover:bg-white/[0.05]"
          >
            重置
          </button>
          <button
            onClick={() => setShowHeatmapOverlay((v) => !v)}
            className={`flex h-8 items-center gap-1.5 rounded-lg border px-3 text-[11px] font-semibold transition-all ${
              showHeatmapOverlay
                ? 'border-primary/30 bg-primary/10 text-primary'
                : 'border-white/[0.08] bg-white/[0.02] text-foreground/85 hover:bg-white/[0.05]'
            }`}
          >
            热力图叠加
          </button>
        </div>
      </header>

      {/* Step 滑块 */}
      <div className="flex items-center gap-3 border-b border-white/[0.06] bg-[#0c0e17]/80 px-5 py-2 backdrop-blur-sm">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Step</span>
        <input
          type="range"
          min={0}
          max={maxStep}
          value={Math.min(currentStep, maxStep)}
          onChange={handleStepChange}
          disabled={isRealTraining}
          className={`h-1 flex-1 appearance-none rounded-full bg-white/10 accent-primary outline-none ${
            isRealTraining ? 'cursor-not-allowed opacity-60' : 'cursor-pointer'
          }`}
        />
        <span className="w-20 text-right font-mono text-[11px] font-bold text-primary">
          {(currentStep + 1).toString().padStart(3, '0')} / {displayTotalSteps}
        </span>
      </div>

      {/* 主内容 */}
      <main className="flex-1 overflow-y-auto p-5">
        <div className="grid grid-cols-12 gap-4">
          {/* 左侧：训练曲线 */}
          <div className="col-span-12 lg:col-span-8">
            <TrainingCurvesPanel
              visibleData={visibleData}
              currentStep={currentStep}
              colors={colors}
              hasActiveTraining={hasActiveTraining}
              onReset={handleResetCanvas}
            />
          </div>

          {/* 右上：指标卡 */}
          <div className="col-span-12 lg:col-span-4">
            <ModelMetricsPanel data={currentData} visibleData={visibleData} colors={colors} />
          </div>

          {/* 混淆矩阵 */}
          <div className="col-span-12 md:col-span-6 lg:col-span-4">
            <ConfusionMatrix data={currentData} colors={colors} />
          </div>

          {/* 激活分布 */}
          <div className="col-span-12 md:col-span-6 lg:col-span-4">
            <ActivationHistogram data={currentData} colors={colors} />
          </div>

          {/* 预测分布 */}
          <div className="col-span-12 md:col-span-6 lg:col-span-4">
            <PredictionDistribution data={currentData} />
          </div>

          {/* 梯度权重 */}
          <div className="col-span-12 lg:col-span-6">
            <GradientWeightChart visibleData={visibleData} currentStep={currentStep} colors={colors} />
          </div>

          {/* 特征图 */}
          <div className="col-span-12 lg:col-span-6">
            <FeatureMapViewer featureMaps={currentData?.featureMaps} step={currentStep} />
          </div>
        </div>

        {/* 底部留白 */}
        <div className="h-8" />
      </main>
    </div>
  );
}
