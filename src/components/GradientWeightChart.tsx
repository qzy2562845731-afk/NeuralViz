import { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption } from 'echarts';
import type { TrainingStepData } from '../types/training';
import type { ChartColors } from '../hooks/useColorConfig';

interface GradientWeightChartProps {
  visibleData: TrainingStepData[];
  currentStep: number;
  colors: ChartColors;
}

export function GradientWeightChart({ visibleData, currentStep, colors }: GradientWeightChartProps) {
  const { chartOption, riskStatus } = useMemo((): { chartOption: EChartsOption; riskStatus: string | null } => {
    if (visibleData.length === 0) {
      return { chartOption: {}, riskStatus: null };
    }

    const steps = visibleData.map(d => d.step);
    const gradientNorm = visibleData.map(d => d.gradientNorm);
    const weightNorm = visibleData.map(d => d.weightNorm);

    const currentGrad = gradientNorm[gradientNorm.length - 1];
    
    let riskStatus: string | null = null;
    if (currentGrad > 5) {
      riskStatus = '梯度爆炸风险';
    } else if (currentGrad < 0.05 && currentStep > 30) {
      riskStatus = '梯度消失风险';
    }

    const maxGrad = Math.max(...gradientNorm, 1);

    const tooltipFormatter = (params: unknown): string => {
      const arr = params as Array<{ seriesName: string; value: number; dataIndex: number }>;
      if (!arr || arr.length === 0) return '';
      
      const step = arr[0].dataIndex;
      const items = arr.map(p => {
        const color = p.seriesName === 'Gradient Norm' ? colors.gradientNorm : colors.weightNorm;
        return `<span style="color:${color}">${p.seriesName}: <b>${(p.value as number).toFixed(4)}</b></span>`;
      }).join('<br/>');
      
      return `<div style="padding: 8px;">
        <div style="font-weight: 600; margin-bottom: 6px;">Step ${step}</div>
        ${items}
      </div>`;
    };

    const chartOption: EChartsOption = {
      backgroundColor: 'transparent',
      grid: {
        left: '12%',
        right: '8%',
        top: riskStatus ? '18%' : '15%',
        bottom: '14%'
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(26, 26, 46, 0.95)',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        textStyle: { color: '#e0e0e0', fontSize: 12 },
        appendToBody: true,
        formatter: tooltipFormatter,
        axisPointer: {
          type: 'cross',
          label: { backgroundColor: '#1a1a2e' }
        }
      },
      legend: {
        data: ['Gradient Norm', 'Weight Norm'],
        textStyle: { color: '#888', fontSize: 10 },
        top: 0
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
          color: '#888',
          fontSize: 10,
          interval: Math.max(Math.floor(steps.length / 8), 1)
        },
        axisTick: { show: false }
      },
      yAxis: [
        {
          type: 'value',
          name: 'Grad Norm',
          nameTextStyle: { color: colors.gradientNorm, fontSize: 11 },
          max: maxGrad * 1.2,
          axisLine: { lineStyle: { color: colors.gradientNorm } },
          axisLabel: {
            color: '#888',
            fontSize: 10,
            formatter: (val: number) => val.toFixed(2)
          },
          splitLine: { lineStyle: { color: '#222', type: 'dashed' } }
        },
        {
          type: 'value',
          name: 'Weight Norm',
          nameTextStyle: { color: colors.weightNorm, fontSize: 11 },
          axisLine: { lineStyle: { color: colors.weightNorm } },
          axisLabel: {
            color: '#888',
            fontSize: 10,
            formatter: (val: number) => val.toFixed(2)
          },
          splitLine: { show: false }
        }
      ],
      series: [
        {
          name: 'Gradient Norm',
          type: 'line',
          data: gradientNorm,
          smooth: 0.4,
          symbol: 'none',
          lineStyle: { color: colors.gradientNorm, width: 2.5 },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0, y: 0, x2: 0, y2: 1,
              colorStops: [
                { offset: 0, color: colors.gradientNorm + '4d' },
                { offset: 1, color: colors.gradientNorm + '05' }
              ]
            }
          },
          markLine: currentGrad > 5 ? {
            silent: true,
            symbol: 'none',
            lineStyle: { color: '#f44336', type: 'dashed', width: 2 },
            data: [{ yAxis: 5, name: 'Explode Threshold' }]
          } : undefined
        },
        {
          name: 'Weight Norm',
          type: 'line',
          yAxisIndex: 1,
          data: weightNorm,
          smooth: 0.4,
          symbol: 'none',
          lineStyle: { color: colors.weightNorm, width: 2 }
        }
      ]
    };

    return { chartOption, riskStatus };
  }, [visibleData, currentStep, colors]);

  return (
    <div className="gradient-weight-chart card">
      <div className="card-header">
        <h2 className="card-title">梯度与权重</h2>
        <span className="chart-info">Step: {currentStep}</span>
      </div>
      {riskStatus && (
        <div className="risk-banner warning">
          <span className="risk-icon">!</span>
          <span className="risk-text">{riskStatus}</span>
        </div>
      )}
      <div className="chart-wrapper-medium">
        <ReactECharts option={chartOption} style={{ height: '100%', width: '100%' }} />
      </div>
    </div>
  );
}
