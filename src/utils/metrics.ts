import type { LayerActivations, LayerStats, ActivationStats, FeatureMaps, FeatureMap } from '../types/training';

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function getAverage(arr: number[]): number {
  if (arr.length === 0) return 0;
  const sum = arr.reduce((acc, val) => acc + val, 0);
  return sum / arr.length;
}

export function getMin(arr: number[]): number {
  if (arr.length === 0) return 0;
  return Math.min(...arr);
}

export function getMax(arr: number[]): number {
  if (arr.length === 0) return 0;
  return Math.max(...arr);
}

export function getVariance(arr: number[]): number {
  if (arr.length === 0) return 0;
  const avg = getAverage(arr);
  const squaredDiffs = arr.map(v => Math.pow(v - avg, 2));
  return getAverage(squaredDiffs);
}

export function getSparsity(arr: number[], threshold: number = 0.01): number {
  if (arr.length === 0) return 0;
  const count = arr.filter(v => Math.abs(v) < threshold).length;
  return count / arr.length;
}

export function activationToColor(activation: number, baseColor: { r: number; g: number; b: number }): string {
  const alpha = 0.3 + activation * 0.7;
  return `rgba(${baseColor.r}, ${baseColor.g}, ${baseColor.b}, ${alpha})`;
}

export function activationToRadius(activation: number): number {
  return Math.max(10, Math.min(20, 12 + activation * 10));
}

export function flattenActivations(activations: LayerActivations): number[] {
  const result: number[] = [];
  result.push(...activations.input);
  result.push(...activations.conv1);
  if (activations.conv2) {
    result.push(...activations.conv2);
  }
  result.push(...activations.fc);
  return result;
}

export function calculateLayerStats(activations: number[], layer: string): LayerStats {
  return {
    layer: layer as LayerStats['layer'],
    avg: getAverage(activations),
    max: getMax(activations),
    min: getMin(activations),
    sparsity: getSparsity(activations),
    variance: getVariance(activations)
  };
}

export function createActivationStats(layerActivations: LayerActivations): ActivationStats {
  return {
    input: calculateLayerStats(layerActivations.input, 'input'),
    conv1: calculateLayerStats(layerActivations.conv1, 'conv1'),
    conv2: layerActivations.conv2 ? calculateLayerStats(layerActivations.conv2, 'conv2') : undefined,
    fc: calculateLayerStats(layerActivations.fc, 'fc')
  };
}

export function createHistogramBins(data: number[], bins: number = 20): number[] {
  if (data.length === 0) return Array(bins).fill(0);
  
  const min = getMin(data);
  const max = getMax(data);
  const range = max - min || 1;
  
  const result = Array(bins).fill(0);
  data.forEach(v => {
    const binIndex = Math.min(bins - 1, Math.floor(((v - min) / range) * bins));
    result[binIndex]++;
  });
  
  return result;
}

export function createConfusionMatrix(numClasses: number = 10): number[][] {
  const matrix: number[][] = Array(numClasses).fill(null).map(() => Array(numClasses).fill(0));
  return matrix;
}

export function generateFeatureMap(size: number = 6): FeatureMap {
  const map: FeatureMap = [];
  for (let i = 0; i < size; i++) {
    const row: number[] = [];
    for (let j = 0; j < size; j++) {
      row.push(Math.random());
    }
    map.push(row);
  }
  return map;
}

export function createFeatureMaps(conv1Count: number = 4, conv2Count?: number): FeatureMaps {
  return {
    conv1: Array(conv1Count).fill(null).map(() => generateFeatureMap()),
    conv2: conv2Count ? Array(conv2Count).fill(null).map(() => generateFeatureMap()) : undefined
  };
}

export function smoothFeatureMap(map: FeatureMap, smoothness: number): FeatureMap {
  const size = map.length;
  const result: FeatureMap = [];
  
  for (let i = 0; i < size; i++) {
    const row: number[] = [];
    for (let j = 0; j < size; j++) {
      let sum = map[i][j];
      let count = 1;
      
      if (i > 0) {
        sum += map[i - 1][j] * smoothness;
        count += smoothness;
      }
      if (i < size - 1) {
        sum += map[i + 1][j] * smoothness;
        count += smoothness;
      }
      if (j > 0) {
        sum += map[i][j - 1] * smoothness;
        count += smoothness;
      }
      if (j < size - 1) {
        sum += map[i][j + 1] * smoothness;
        count += smoothness;
      }
      
      row.push(sum / count);
    }
    result.push(row);
  }
  
  return result;
}

export function normalizeFeatureMap(map: FeatureMap): FeatureMap {
  let maxVal = 0;
  for (const row of map) {
    const rowMax = Math.max(...row);
    if (rowMax > maxVal) maxVal = rowMax;
  }
  
  if (maxVal === 0) return map;
  
  return map.map(row => row.map(v => v / maxVal));
}
