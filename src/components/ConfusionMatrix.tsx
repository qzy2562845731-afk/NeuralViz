import { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption } from 'echarts';
import type { TrainingStepData } from '../types/training';
import type { ChartColors } from '../hooks/useColorConfig';

interface ConfusionMatrixProps {
  data: TrainingStepData | null;
  colors: ChartColors;
}

export function ConfusionMatrix({ data, colors }: ConfusionMatrixProps) {
  const chartOption = useMemo((): EChartsOption => {
    // 可选链保护：confusionMatrix 可能为 undefined 时不崩溃
    if (!data || !data?.confusionMatrix || data.confusionMatrix.length === 0) {
      return {};
    }

    const matrix = data.confusionMatrix;
    const numClasses = matrix.length;
    const xAxisData = Array.from({ length: numClasses }, (_, i) => `C${i}`);
    const yAxisData = Array.from({ length: numClasses }, (_, i) => `C${i}`);

    const heatmapData: [number, number, number][] = [];
    matrix.forEach((row, i) => {
      row.forEach((val, j) => {
        heatmapData.push([j, i, val]);
      });
    });

    return {
      backgroundColor: 'transparent',
      grid: {
        left: '12%',
        right: '10%',
        top: '8%',
        bottom: '12%'
      },
      tooltip: {
        position: 'top',
        backgroundColor: 'rgba(26, 26, 46, 0.95)',
        borderColor: 'rgba(255, 255, 255, 0.1)',
        textStyle: { color: '#e0e0e0', fontSize: 12 },
        appendToBody: true,
        formatter: (params: unknown) => {
          const p = params as { data: [number, number, number] };
          const [x, y, val] = p.data;
          const isCorrect = x === y;
          return `
            <div style="padding: 8px;">
              <div style="font-weight: 600; margin-bottom: 4px;">
                ${isCorrect ? '✓ 正确' : '✗ 错误'}
              </div>
              <div>Actual: <span style="color: #4fd1c5;">Class ${y}</span></div>
              <div>Predicted: <span style="color: #f093fb;">Class ${x}</span></div>
              <div>Count: <span style="font-weight: 600;">${val}</span></div>
            </div>
          `;
        }
      },
      xAxis: {
        type: 'category',
        data: xAxisData,
        name: 'Predicted',
        nameLocation: 'middle',
        nameGap: 30,
        nameTextStyle: { color: '#666', fontSize: 11 },
        axisLine: { lineStyle: { color: '#333' } },
        axisLabel: { color: '#888', fontSize: 10 },
        axisTick: { show: false },
        splitArea: { show: false }
      },
      yAxis: {
        type: 'category',
        data: yAxisData,
        name: 'Actual',
        nameLocation: 'middle',
        nameGap: 35,
        nameTextStyle: { color: '#666', fontSize: 11 },
        axisLine: { lineStyle: { color: '#333' } },
        axisLabel: { color: '#888', fontSize: 10 },
        axisTick: { show: false },
        splitArea: { show: false }
      },
      visualMap: {
        show: true,
        orient: 'horizontal',
        left: 'center',
        bottom: '0%',
        min: 0,
        max: Math.max(...matrix.flat()),
        calculable: false,
        inRange: {
          color: ['#0a0a14', '#1a1a2e', colors.confusionMatrix]
        },
        textStyle: {
          color: '#666',
          fontSize: 10
        }
      },
      series: [{
        type: 'heatmap',
        data: heatmapData,
        label: {
          show: true,
          fontSize: 9,
          color: '#fff',
          formatter: (params: unknown) => {
            const p = params as { value: [number, number, number] };
            return p.value[2] > 0 ? `${p.value[2]}` : '';
          }
        },
        emphasis: {
          itemStyle: {
            borderColor: '#fff',
            borderWidth: 2
          }
        },
        itemStyle: {
          borderColor: 'rgba(255, 255, 255, 0.05)',
          borderWidth: 1
        }
      }]
    };
  }, [data, colors]);

  // 可选链保护：confusionMatrix 可能为 undefined 时显示空状态
  if (!data || !data?.confusionMatrix || data.confusionMatrix.length === 0) {
    return (
      <div className="confusion-matrix card">
        <div className="card-header">
          <h2 className="card-title">混淆矩阵</h2>
        </div>
        <div className="empty-state">
          <span>等待数据加载...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="confusion-matrix card">
      <div className="card-header">
        <h2 className="card-title">混淆矩阵</h2>
        <span className="card-subtitle">10 类分类结果</span>
      </div>
      <div className="chart-wrapper-medium">
        <ReactECharts option={chartOption} style={{ height: '100%', width: '100%' }} />
      </div>
    </div>
  );
}
