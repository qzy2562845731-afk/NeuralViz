import React, { useCallback, useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import type { EChartsOption } from 'echarts';
import type { TrainingStepData } from '../types/training';

interface LossChartProps {
  visibleData: TrainingStepData[];
  currentStep: number;
}

export const LossChart: React.FC<LossChartProps> = ({ visibleData, currentStep }) => {
  const getChartOption = useCallback((): EChartsOption => {
    const steps = visibleData.map((d) => d.step);
    const losses = visibleData.map((d) => d.loss);
    const accuracies = visibleData.map((d) => d.accuracy);

    const maxLoss = Math.max(...losses, 0.1);
    const minLoss = Math.min(...losses, 0);

    return {
      backgroundColor: 'transparent',
      grid: {
        left: '12%',
        right: '8%',
        top: '18%',
        bottom: '12%'
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(26, 26, 46, 0.95)',
        borderColor: '#333',
        borderWidth: 1,
        textStyle: {
          color: '#e0e0e0'
        },
        appendToBody: true,
        formatter: (params: unknown) => {
          const arr = params as Array<{ seriesIndex: number; value: number; dataIndex: number }>;
          if (!arr || arr.length === 0) return '';
          const step = arr[0].dataIndex;
          const lossVal = arr.find((p) => p.seriesIndex === 0)?.value ?? '-';
          const accVal = arr.find((p) => p.seriesIndex === 1)?.value ?? '-';
          return `
            <div style="padding: 8px;">
              <div style="font-weight: 600; margin-bottom: 8px; font-size: 14px;">Step ${step}</div>
              <div style="color: #f093fb; margin-bottom: 4px;">Loss: ${(lossVal as number).toFixed(4)}</div>
              <div style="color: #4fd1c5;">Accuracy: ${((accVal as number) * 100).toFixed(2)}%</div>
            </div>
          `;
        }
      },
      legend: {
        data: ['Loss', 'Accuracy'],
        textStyle: {
          color: '#888'
        },
        top: 0,
        right: 0
      },
      xAxis: {
        type: 'category',
        data: steps,
        name: 'Step',
        nameLocation: 'middle',
        nameGap: 32,
        nameTextStyle: {
          color: '#666'
        },
        axisLine: {
          lineStyle: {
            color: '#333'
          }
        },
        axisLabel: {
          color: '#666',
          interval: Math.max(Math.floor(steps.length / 10), 1)
        },
        axisTick: {
          show: false
        }
      },
      yAxis: [
        {
          type: 'value',
          name: 'Loss',
          nameTextStyle: {
            color: '#f093fb'
          },
          min: Math.max(0, minLoss - 0.1),
          max: maxLoss + 0.1,
          axisLine: {
            lineStyle: {
              color: '#f093fb'
            }
          },
          axisLabel: {
            color: '#666',
            formatter: (val: number) => val.toFixed(1)
          },
          splitLine: {
            lineStyle: {
              color: '#222',
              type: 'dashed'
            }
          }
        },
        {
          type: 'value',
          name: 'Accuracy',
          nameTextStyle: {
            color: '#4fd1c5'
          },
          min: 0,
          max: 1,
          axisLine: {
            lineStyle: {
              color: '#4fd1c5'
            }
          },
          axisLabel: {
            color: '#666',
            formatter: (val: number) => `${(val * 100).toFixed(0)}%`
          },
          splitLine: {
            show: false
          }
        }
      ],
      series: [
        {
          name: 'Loss',
          type: 'line',
          data: losses,
          smooth: true,
          symbol: 'circle',
          symbolSize: 6,
          lineStyle: {
            color: '#f093fb',
            width: 2
          },
          itemStyle: {
            color: '#f093fb',
            borderWidth: 2,
            borderColor: '#fff'
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(240, 147, 251, 0.3)' },
                { offset: 1, color: 'rgba(240, 147, 251, 0.02)' }
              ]
            }
          }
        },
        {
          name: 'Accuracy',
          type: 'line',
          yAxisIndex: 1,
          data: accuracies,
          smooth: true,
          symbol: 'circle',
          symbolSize: 6,
          lineStyle: {
            color: '#4fd1c5',
            width: 2
          },
          itemStyle: {
            color: '#4fd1c5',
            borderWidth: 2,
            borderColor: '#fff'
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(79, 209, 197, 0.25)' },
                { offset: 1, color: 'rgba(79, 209, 197, 0.02)' }
              ]
            }
          }
        }
      ],
      animationDuration: 150,
      animationEasing: 'cubicOut'
    };
  }, [visibleData]);

  const chartStyle = useMemo(() => ({ height: '100%', width: '100%' }), []);

  return (
    <div className="chart-panel">
      <div className="panel-header">
        <h2>Loss 曲线</h2>
        <span className="chart-status">当前 Step: {currentStep}</span>
      </div>
      <div className="chart-container">
        <ReactECharts option={getChartOption()} style={chartStyle} />
      </div>
    </div>
  );
};
