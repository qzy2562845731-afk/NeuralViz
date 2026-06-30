import { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption } from 'echarts';
import type { TrainingStepData } from '../types/training';

interface PredictionDistributionProps {
  data: TrainingStepData | null;
}

export function PredictionDistribution({ data }: PredictionDistributionProps) {
  // fe11修复：无预测分布数据时显示空状态
  const hasData = data && data.predictionDistribution && data.predictionDistribution.length > 0;

  const chartOption = useMemo((): EChartsOption => {
    if (!hasData) return {};

    const values = data!.predictionDistribution;
    const categories = values.map((_, i) => `Class ${i}`);

    return {
      backgroundColor: 'transparent',
      grid: {
        left: '12%',
        right: '8%',
        top: '18%',
        bottom: '12%',
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(26, 26, 46, 0.95)',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        textStyle: { color: '#e0e0e0', fontSize: 12 },
        appendToBody: true,
        formatter: (params: unknown) => {
          const arr = params as Array<{ name: string; value: number }>;
          const p = arr[0];
          return `<div style="padding: 8px;"><b>${p.name}</b><br/>Probability: ${(p.value * 100).toFixed(1)}%</div>`;
        },
      },
      xAxis: {
        type: 'category',
        data: categories,
        axisLine: { lineStyle: { color: '#333' } },
        axisLabel: { color: '#888', fontSize: 10 },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value',
        min: 0,
        max: 1,
        axisLine: { lineStyle: { color: '#333' } },
        axisLabel: { color: '#888', fontSize: 10, formatter: (val: number) => `${(val * 100).toFixed(0)}%` },
        splitLine: { lineStyle: { color: '#222', type: 'dashed' } },
      },
      series: [
        {
          type: 'bar',
          data: values,
          barWidth: '55%',
          itemStyle: {
            borderRadius: [4, 4, 0, 0],
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: '#a78bfa' },
                { offset: 1, color: '#4c1d95' },
              ],
            },
          },
        },
      ],
    };
  }, [data]);

  return (
    <div className="prediction-distribution card">
      <div className="card-header">
        <h2 className="card-title">预测分布</h2>
        <span className="card-subtitle">当前样本各类别概率</span>
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
