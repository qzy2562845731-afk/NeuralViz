import { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption } from 'echarts';
import type { TrainingStepData } from '../types/training';

import type { ChartColors } from '../hooks/useColorConfig';

interface ActivationHistogramProps {
  data: TrainingStepData | null;
  colors: ChartColors;
}

const BIN_RANGES = [
  '0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0'
];

function createBins(arr: number[]): number[] {
  const bins = [0, 0, 0, 0, 0];
  arr.forEach(v => {
    if (v < 0.2) bins[0]++;
    else if (v < 0.4) bins[1]++;
    else if (v < 0.6) bins[2]++;
    else if (v < 0.8) bins[3]++;
    else bins[4]++;
  });
  return bins;
}

export function ActivationHistogram({ data, colors }: ActivationHistogramProps) {
  // fe11修复：无激活值数据时显示空状态，不渲染全零图表
  // 可选链保护：layerActivations 可能为 undefined 时不崩溃
  const hasData = data && (
    (data?.layerActivations?.input && data.layerActivations.input.length > 0) ||
    (data?.layerActivations?.conv1 && data.layerActivations.conv1.length > 0) ||
    (data?.layerActivations?.fc && data.layerActivations.fc.length > 0)
  );

  const chartOption = useMemo((): EChartsOption => {
    if (!hasData) {
      return {};
    }

    const inputBins = createBins(data!.layerActivations.input);
    const conv1Bins = createBins(data!.layerActivations.conv1);
    const conv2Bins = data!.layerActivations.conv2 ? createBins(data!.layerActivations.conv2) : [];
    const fcBins = createBins(data!.layerActivations.fc);

    const seriesData = [
      { name: 'Input', data: inputBins, color: colors.activationHistogram[0] ?? '#4285f4' },
      { name: 'Conv1', data: conv1Bins, color: colors.activationHistogram[1] ?? '#34a853' },
      { name: 'Conv2', data: conv2Bins.length > 0 ? conv2Bins : null, color: colors.activationHistogram[2] ?? '#fbbc04' },
      { name: 'FC', data: fcBins, color: colors.activationHistogram[3] ?? '#ea4335' }
    ].filter((s): s is { name: string; data: number[]; color: string } => s.data !== null);

    const tooltipFormatter = (params: unknown): string => {
      const arr = params as Array<{ seriesName: string; name: string; value: number }>;
      if (!arr || arr.length === 0) return '';
      
      const items = arr.map(p => {
        const color = seriesData.find(s => s.name === p.seriesName)?.color ?? '#888';
        return `<span style="color:${color}">${p.seriesName} [${p.name}]: <b>${p.value}</b></span>`;
      }).join('<br/>');
      
      return `<div style="padding: 8px;">
        <div style="font-weight: 600; margin-bottom: 6px;">激活值范围</div>
        ${items}
      </div>`;
    };

    return {
      backgroundColor: 'transparent',
      grid: {
        left: '12%',
        right: '8%',
        top: '18%',
        bottom: '15%'
      },
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        backgroundColor: 'rgba(26, 26, 46, 0.95)',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        textStyle: { color: '#e0e0e0', fontSize: 12 },
        appendToBody: true,
        formatter: tooltipFormatter
      },
      legend: {
        data: seriesData.map(s => s.name),
        textStyle: { color: '#888', fontSize: 10 },
        top: 0,
        right: '5%'
      },
      xAxis: {
        type: 'category',
        data: BIN_RANGES,
        name: 'Activation Range',
        nameLocation: 'middle',
        nameGap: 30,
        nameTextStyle: { color: '#666', fontSize: 11 },
        axisLine: { lineStyle: { color: '#333' } },
        axisLabel: { color: '#888', fontSize: 10, rotate: 30 },
        axisTick: { show: false }
      },
      yAxis: {
        type: 'value',
        name: 'Count',
        nameTextStyle: { color: '#666', fontSize: 11 },
        axisLine: { lineStyle: { color: '#333' } },
        axisLabel: { color: '#888', fontSize: 10 },
        splitLine: { lineStyle: { color: '#222', type: 'dashed' } }
      },
      series: seriesData.map(s => ({
        name: s.name,
        type: 'bar',
        data: s.data,
        itemStyle: {
          color: {
            type: 'linear',
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: s.color },
              { offset: 1, color: `${s.color}55` }
            ]
          },
          borderRadius: [3, 3, 0, 0]
        },
        barWidth: '25%'
      }))
    };
  }, [data, colors]);

  return (
    <div className="activation-histogram card">
      <div className="card-header">
        <h2 className="card-title">激活分布</h2>
        <span className="card-subtitle">各层激活值分桶统计</span>
      </div>
      <div className="chart-wrapper-medium">
        {hasData ? (
          <ReactECharts option={chartOption} style={{ height: '100%', width: '100%' }} />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
            等待训练数据...
          </div>
        )}
      </div>
    </div>
  );
}
