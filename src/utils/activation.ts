import type { LayerActivations } from '../types/training';

export const LAYER_COLORS = {
  input: { r: 66, g: 135, b: 245 },
  conv1: { r: 78, g: 205, b: 196 },
  conv2: { r: 119, g: 182, b: 129 },
  fc: { r: 255, g: 107, b: 107 }
} as const;

export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

export function activationToColor(activation: number, layerType: keyof LayerActivations): string {
  const color = LAYER_COLORS[layerType];
  const alpha = 0.3 + activation * 0.7;
  return `rgba(${color.r}, ${color.g}, ${color.b}, ${alpha})`;
}

export function activationToStrokeColor(activation: number, layerType: keyof LayerActivations): string {
  const color = LAYER_COLORS[layerType];
  const alpha = 0.6 + activation * 0.4;
  return `rgba(${color.r}, ${color.g}, ${color.b}, ${alpha})`;
}

export function activationToOpacity(activation: number): number {
  return Math.max(0.3, Math.min(1, 0.3 + activation * 0.7));
}

export function activationToRadius(activation: number): number {
  return Math.max(10, Math.min(20, 12 + activation * 10));
}

export function calculateAverage(arr: number[]): number {
  if (arr.length === 0) return 0;
  const sum = arr.reduce((acc, val) => acc + val, 0);
  return sum / arr.length;
}

export function getActivationSummary(activations: number[]): {
  avg: number;
  max: number;
  min: number;
} {
  if (activations.length === 0) {
    return { avg: 0, max: 0, min: 0 };
  }
  return {
    avg: calculateAverage(activations),
    max: Math.max(...activations),
    min: Math.min(...activations)
  };
}

export function getSparsity(arr: number[], threshold: number = 0.01): number {
  if (arr.length === 0) return 0;
  const count = arr.filter(v => Math.abs(v) < threshold).length;
  return count / arr.length;
}

export function getVariance(arr: number[]): number {
  if (arr.length === 0) return 0;
  const avg = calculateAverage(arr);
  const squaredDiffs = arr.map(v => Math.pow(v - avg, 2));
  return calculateAverage(squaredDiffs);
}
