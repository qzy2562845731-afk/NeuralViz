import type { EChartsOption } from 'echarts';

export const CHART_COLORS = {
  trainLoss: '#f093fb',
  valLoss: '#ff9800',
  trainAccuracy: '#4fd1c5',
  valAccuracy: '#66bb6a',
  learningRate: '#667eea',
  gradientNorm: '#ff9800',
  weightNorm: '#9c27b0',
  precision: '#ab47bc',
  recall: '#26a69a',
  f1Score: '#42a5f5',
  background: 'transparent',
  grid: '#222',
  text: '#666',
  axisLine: '#333',
  tooltip: 'rgba(26, 26, 46, 0.95)',
  tooltipBorder: 'rgba(255, 255, 255, 0.1)'
} as const;

export interface GridConfig {
  left?: string | number;
  right?: string | number;
  top?: string | number;
  bottom?: string | number;
}

export interface AxisConfig {
  name?: string;
  nameTextStyle?: { color?: string; fontSize?: number };
  axisLine?: { lineStyle?: { color?: string } };
  axisLabel?: { color?: string; formatter?: (value: number) => string };
  splitLine?: { lineStyle?: { color?: string; type?: 'solid' | 'dashed' | 'dotted' } };
  min?: number | string;
  max?: number | string;
}

export function createGridConfig(config?: GridConfig): EChartsOption['grid'] {
  return {
    left: config?.left ?? '12%',
    right: config?.right ?? '8%',
    top: config?.top ?? '15%',
    bottom: config?.bottom ?? '12%',
    containLabel: true
  };
}

export function createXAxisConfig(name?: string): EChartsOption['xAxis'] {
  return {
    type: 'category' as const,
    name: name,
    nameLocation: 'middle' as const,
    nameGap: 30,
    nameTextStyle: {
      color: CHART_COLORS.text,
      fontSize: 12
    },
    axisLine: {
      lineStyle: {
        color: CHART_COLORS.axisLine
      }
    },
    axisLabel: {
      color: CHART_COLORS.text,
      fontSize: 11,
      interval: 'auto'
    },
    axisTick: {
      show: false
    },
    splitLine: {
      show: false
    }
  };
}

export function createYAxisConfig(config?: AxisConfig & { position?: 'left' | 'right' }): EChartsOption['yAxis'] {
  const color = config?.nameTextStyle?.color ?? CHART_COLORS.text;
  
  return {
    type: 'value' as const,
    name: config?.name,
    nameTextStyle: {
      color: config?.nameTextStyle?.color ?? CHART_COLORS.text,
      fontSize: 11
    },
    position: config?.position ?? 'left',
    min: config?.min,
    max: config?.max,
    axisLine: {
      show: true,
      lineStyle: {
        color: color
      }
    },
    axisLabel: {
      color: CHART_COLORS.text,
      fontSize: 10,
      formatter: config?.axisLabel?.formatter
    },
    splitLine: {
      lineStyle: {
        color: CHART_COLORS.grid,
        type: 'dashed' as const
      }
    }
  };
}

export function createTooltipConfig(
  formatter?: (params: unknown) => string
): EChartsOption['tooltip'] {
  return {
    trigger: 'axis' as const,
    backgroundColor: CHART_COLORS.tooltip,
    borderColor: CHART_COLORS.tooltipBorder,
    borderWidth: 1,
    textStyle: {
      color: '#e0e0e0',
      fontSize: 12
    },
    formatter: formatter,
    appendToBody: true,
    axisPointer: {
      type: 'cross' as const,
      crossStyle: {
        color: '#888'
      },
      lineStyle: {
        color: '#555',
        type: 'dashed' as const
      }
    }
  };
}

export function createLegendConfig(
  data: string[],
  top?: number | string
): EChartsOption['legend'] {
  return {
    data: data,
    textStyle: {
      color: CHART_COLORS.text,
      fontSize: 11
    },
    top: top ?? 0,
    right: '5%',
    itemWidth: 14,
    itemHeight: 8
  };
}

export function createBaseChartOption(partial?: Partial<EChartsOption>): EChartsOption {
  return {
    backgroundColor: CHART_COLORS.background,
    animation: true,
    animationDuration: 200,
    animationEasing: 'cubicOut' as const,
    ...partial
  };
}

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(2)}%`;
}

export function formatLoss(value: number): string {
  return value.toFixed(4);
}

export function formatScientific(value: number): string {
  if (!value || value === 0) return '0.0000';
  return value.toFixed(4);
}

export function formatNorm(value: number): string {
  return value.toFixed(3);
}

export function getColorForSeries(seriesName: string): string {
  const colorMap: Record<string, string> = {
    'Train Loss': CHART_COLORS.trainLoss,
    'Val Loss': CHART_COLORS.valLoss,
    'Train Accuracy': CHART_COLORS.trainAccuracy,
    'Val Accuracy': CHART_COLORS.valAccuracy,
    'Learning Rate': CHART_COLORS.learningRate,
    'Gradient Norm': CHART_COLORS.gradientNorm,
    'Weight Norm': CHART_COLORS.weightNorm,
    'Precision': CHART_COLORS.precision,
    'Recall': CHART_COLORS.recall,
    'F1 Score': CHART_COLORS.f1Score,
    'Input': '#4285f4',
    'Conv1': '#34a853',
    'Conv2': '#fbbc04',
    'FC': '#ea4335'
  };
  return colorMap[seriesName] ?? '#888';
}

export function createSeriesStyle(
  seriesName: string,
  options?: {
    smooth?: boolean;
    areaStyle?: boolean;
    dashed?: boolean;
    yAxisIndex?: number;
    symbolSize?: number;
  }
): Record<string, unknown> {
  const color = getColorForSeries(seriesName);
  const lineStyle: Record<string, unknown> = {
    width: 2
  };
  
  if (options?.dashed) {
    lineStyle.type = 'dashed';
  }
  
  return {
    type: 'line' as const,
    smooth: options?.smooth ?? true,
    symbol: 'circle' as const,
    symbolSize: options?.symbolSize ?? 4,
    yAxisIndex: options?.yAxisIndex ?? 0,
    lineStyle: lineStyle,
    itemStyle: {
      color: color,
      borderWidth: 1,
      borderColor: '#fff'
    },
    areaStyle: options?.areaStyle ? {
      color: {
        type: 'linear' as const,
        x: 0, y: 0, x2: 0, y2: 1,
        colorStops: [
          { offset: 0, color: color.replace(')', ', 0.3)').replace('rgb', 'rgba') },
          { offset: 1, color: color.replace(')', ', 0.02)').replace('rgb', 'rgba') }
        ]
      }
    } : undefined
  };
}
