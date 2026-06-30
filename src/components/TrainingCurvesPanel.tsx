import { useMemo, useRef, useState } from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption } from 'echarts';
import type { TrainingStepData } from '../types/training';
import type { ChartColors } from '../hooks/useColorConfig';

interface TrainingCurvesPanelProps {
  visibleData: TrainingStepData[];
  currentStep: number;
  colors: ChartColors;
  hasActiveTraining?: boolean;
  onReset?: () => void;
}

const ALL_SERIES = ['Train Loss', 'Val Loss', 'Train Acc', 'Val Acc', 'Learning Rate'] as const;

export function TrainingCurvesPanel({ visibleData, currentStep, colors, onReset }: TrainingCurvesPanelProps) {
  const chartRef = useRef<ReactECharts>(null);
  const [hiddenSeries, setHiddenSeries] = useState<Set<string>>(new Set());

  // 图例点击切换显示/隐藏
  const toggleSeries = (name: string) => {
    setHiddenSeries(prev => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  // 下载图表截图
  const handleDownloadScreenshot = () => {
    const chartInstance = chartRef.current?.getEchartsInstance();
    if (!chartInstance) return;
    try {
      const url = chartInstance.getDataURL({
        type: 'png',
        pixelRatio: 2,
        backgroundColor: '#0c0e17',
      });
      const link = document.createElement('a');
      link.href = url;
      link.download = `training-curves-step-${currentStep}-${Date.now()}.png`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    } catch (err) {
      console.error('下载截图失败:', err);
    }
  };

  // 重置视图（清空隐藏的图例）
  const handleResetView = () => {
    setHiddenSeries(new Set());
    const chartInstance = chartRef.current?.getEchartsInstance();
    if (chartInstance) {
      chartInstance.dispatchAction({ type: 'restore' });
    }
  };

  const chartOption = useMemo((): EChartsOption => {
    if (visibleData.length === 0) {
      return {};
    }

    const steps = visibleData.map((d) => d.step);
    const trainLoss = visibleData.map((d) => d.trainLoss);
    const valLoss = visibleData.map((d) => d.valLoss);
    const trainAcc = visibleData.map((d) => d.trainAccuracy * 100);
    const valAcc = visibleData.map((d) => d.valAccuracy * 100);
    const learningRate = visibleData.map((d) => d.learningRate);

    const maxLoss = Math.max(...trainLoss, ...valLoss, 0.1);

    // 自动计算刻度间隔，保证刻度文字完整显示
    const stepCount = steps.length;
    const labelInterval = stepCount > 50 ? Math.ceil(stepCount / 8) : stepCount > 20 ? Math.ceil(stepCount / 10) : 0;

    const tooltipFormatter = (params: unknown): string => {
      const arr = params as Array<{ seriesName: string; value: number | string; dataIndex: number }>;
      if (!arr || arr.length === 0) return '';
      const step = arr[0].dataIndex;

      const lines = arr.map(p => {
        let color = '#888';
        let formattedValue = String(p.value);

        if (p.seriesName.includes('Loss')) {
          color = p.seriesName.includes('Train') ? colors.trainLoss : colors.valLoss;
          formattedValue = (p.value as number).toFixed(4);
        } else if (p.seriesName.includes('Acc')) {
          color = p.seriesName.includes('Train') ? colors.trainAccuracy : colors.valAccuracy;
          formattedValue = `${(p.value as number).toFixed(2)}%`;
        } else if (p.seriesName === 'Learning Rate') {
          color = colors.learningRate;
          // fe10修复：学习率使用 toFixed(4) 而非 toExponential，避免 0.00e+0
          formattedValue = (p.value as number) > 0 ? (p.value as number).toFixed(4) : '—';
        }

        return `<span style="color:${color}">${p.seriesName}: ${formattedValue}</span>`;
      });

      return `<div style="padding: 10px; min-width: 180px;">
        <div style="font-weight:600; margin-bottom:8px; border-bottom:1px solid #333; padding-bottom:6px;">
          Step ${step}
        </div>
        ${lines.join('<br/>')}
      </div>`;
    };

    const buildSeries = (name: string, data: number[], color: string, yAxisIndex: number, opts?: { dashed?: boolean; area?: boolean; dotted?: boolean }) => {
      if (hiddenSeries.has(name)) {
        return { name, type: 'line' as const, data: [], lineStyle: { opacity: 0 }, itemStyle: { opacity: 0 } };
      }
      return {
        name,
        type: 'line' as const,
        yAxisIndex,
        data,
        smooth: 0.4,
        symbol: 'none',
        lineStyle: {
          color,
          width: opts?.dotted ? 1.5 : 2.5,
          ...(opts?.dashed ? { type: 'dashed' as const } : {}),
          ...(opts?.dotted ? { type: 'dotted' as const } : {}),
        },
        ...(opts?.area ? {
          areaStyle: {
            color: {
              type: 'linear' as const,
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: color + '4d' },
                { offset: 1, color: color + '05' }
              ]
            }
          }
        } : {}),
        z: opts?.dotted ? -1 : 0,
      };
    };

    return {
      backgroundColor: 'transparent',
      grid: { left: '8%', right: '8%', top: '15%', bottom: '12%', containLabel: true },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(15, 17, 25, 0.95)',
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1,
        textStyle: { color: '#e0e0e0', fontSize: 12 },
        formatter: tooltipFormatter,
        appendToBody: true,
        axisPointer: {
          type: 'cross',
          crossStyle: { color: '#555' },
          lineStyle: { color: '#555', type: 'dashed' },
        },
      },
      xAxis: {
        type: 'category',
        data: steps,
        name: 'Step',
        nameLocation: 'middle',
        nameGap: 28,
        nameTextStyle: { color: '#666', fontSize: 11 },
        axisLine: { lineStyle: { color: '#333' } },
        axisLabel: {
          color: '#666',
          fontSize: 10,
          interval: labelInterval,
          hideOverlap: true,
        },
        axisTick: { show: false },
        splitLine: { show: false },
      },
      yAxis: [
        {
          type: 'value',
          name: 'Loss',
          nameTextStyle: { color: colors.trainLoss, fontSize: 11 },
          position: 'left',
          min: 0,
          max: maxLoss * 1.1,
          axisLine: { show: true, lineStyle: { color: colors.trainLoss } },
          axisLabel: { color: '#666', fontSize: 10, formatter: (val: number) => val.toFixed(2) },
          splitLine: { lineStyle: { color: '#222', type: 'dashed' } },
        },
        {
          type: 'value',
          name: 'Accuracy (%)',
          nameTextStyle: { color: colors.trainAccuracy, fontSize: 11 },
          position: 'right',
          min: 0,
          max: 100,
          axisLine: { show: true, lineStyle: { color: colors.trainAccuracy } },
          axisLabel: { color: '#666', fontSize: 10, formatter: (val: number) => `${val.toFixed(0)}%` },
          splitLine: { show: false },
        }
      ],
      series: [
        buildSeries('Train Loss', trainLoss, colors.trainLoss, 0, { area: true }),
        buildSeries('Val Loss', valLoss, colors.valLoss, 0, { dashed: true }),
        buildSeries('Train Acc', trainAcc, colors.trainAccuracy, 1, { area: true }),
        buildSeries('Val Acc', valAcc, colors.valAccuracy, 1, { dashed: true }),
        buildSeries('Learning Rate', learningRate, colors.learningRate, 0, { dotted: true }),
      ],
      animation: true,
      animationDuration: 300,
      animationEasing: 'cubicOut',
    };
  }, [visibleData, colors, hiddenSeries]);

  // 空状态：无训练数据（训练完成后仍需显示历史曲线，不判断 hasActiveTraining）
  if (visibleData.length === 0) {
    return (
      <div className="rounded-xl border border-white/[0.06] bg-[#0c0e17] p-4">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-bold">训练曲线</h2>
          {onReset && (
            <button
              onClick={onReset}
              className="flex items-center gap-1 rounded-md border border-white/[0.08] bg-white/[0.02] px-2 py-1 text-[10px] text-muted-foreground transition hover:text-foreground"
            >
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                <path d="M3 3v5h5" />
              </svg>
              重置画布
            </button>
          )}
        </div>
        <div className="flex h-[320px] flex-col items-center justify-center">
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="mb-3 text-muted-foreground/30">
            <path d="M3 3v18h18" />
            <path d="M7 16l4-6 4 3 5-8" />
          </svg>
          <p className="text-[12px] font-medium text-muted-foreground/70">暂无训练数据</p>
          <p className="mt-1 text-[10px] text-muted-foreground/50">请先在 3D 工作台启动训练</p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-white/[0.06] bg-[#0c0e17] p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h2 className="text-sm font-bold">训练曲线</h2>
          <span className="text-[10px] text-muted-foreground">
            当前训练：第 {currentStep} 步 / 共 {visibleData.length} 步
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleDownloadScreenshot}
            className="flex items-center gap-1 rounded-md border border-white/[0.08] bg-white/[0.02] px-2 py-1 text-[10px] text-muted-foreground transition hover:text-foreground"
            title="下载截图"
          >
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="7 10 12 15 17 10" />
              <line x1="12" y1="15" x2="12" y2="3" />
            </svg>
            下载截图
          </button>
          <button
            onClick={handleResetView}
            className="flex items-center gap-1 rounded-md border border-white/[0.08] bg-white/[0.02] px-2 py-1 text-[10px] text-muted-foreground transition hover:text-foreground"
            title="重置视图"
          >
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
              <path d="M3 3v5h5" />
            </svg>
            重置视图
          </button>
          {onReset && (
            <button
              onClick={onReset}
              className="flex items-center gap-1 rounded-md border border-white/[0.08] bg-white/[0.02] px-2 py-1 text-[10px] text-muted-foreground transition hover:text-foreground"
              title="重置画布"
            >
              <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                <path d="M3 3v5h5" />
              </svg>
              重置画布
            </button>
          )}
        </div>
      </div>

      {/* 可点击图例 */}
      <div className="mb-2 flex flex-wrap items-center gap-2">
        {ALL_SERIES.map((name) => {
          const isHidden = hiddenSeries.has(name);
          const color = name === 'Train Loss' ? colors.trainLoss
            : name === 'Val Loss' ? colors.valLoss
            : name === 'Train Acc' ? colors.trainAccuracy
            : name === 'Val Acc' ? colors.valAccuracy
            : colors.learningRate;
          return (
            <button
              key={name}
              onClick={() => toggleSeries(name)}
              className={`flex items-center gap-1.5 rounded-md border px-2 py-0.5 text-[10px] font-medium transition-all ${
                isHidden
                  ? 'border-white/[0.04] bg-white/[0.01] text-muted-foreground/40'
                  : 'border-white/[0.08] bg-white/[0.03] text-foreground/80 hover:bg-white/[0.05]'
              }`}
            >
              <span
                className="h-2 w-2 rounded-full"
                style={{ backgroundColor: isHidden ? '#444' : color, opacity: isHidden ? 0.4 : 1 }}
              />
              {name}
            </button>
          );
        })}
      </div>

      <div style={{ height: 300 }}>
        <ReactECharts
          ref={chartRef}
          option={chartOption}
          style={{ height: '100%', width: '100%' }}
          opts={{ renderer: 'canvas' }}
        />
      </div>
    </div>
  );
}
