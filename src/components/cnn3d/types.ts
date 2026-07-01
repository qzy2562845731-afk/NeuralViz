/* ============================================
   CNN 3D Viewer 类型定义
   支持动态层架构、层分组、颜色编码
   ============================================ */

import * as THREE from 'three';

/* ---------- 层类型 ---------- */
export type LayerCategory = 'input' | 'conv' | 'pool' | 'fc' | 'output' | 'norm' | 'dropout';

/* ---------- 单个层配置 ---------- */
export interface LayerConfig {
  id: string;           // 唯一 ID
  name: string;         // 显示名称
  type: LayerCategory;  // 层类型
  nodeCount: number;    // 节点数 / 通道数
  inputShape: number[]; // 输入形状
  outputShape: number[];
  params: number;       // 参数数量
  activation?: string;   // 激活函数名
  kernelSize?: number;    // 卷积核大小
  group?: string;       // 分组名（用于视觉分组
}

/* ---------- 动态网络架构配置 ---------- */
export interface NetworkArchitecture {
  name: string;
  layers: LayerConfig[];
}

/* ---------- 3D 视图模式 ---------- */
export type ViewMode = 'structure' | 'activation' | 'parameter' | 'feature';

/* ---------- 视图模式说明 ---------- */
export const VIEW_MODES: Array<{ id: ViewMode; name: string; description: string; icon: string
}> = [
  { id: 'structure', name: '结构视图', description: '显示网络结构和形状', icon: '◈' },
  { id: 'activation', name: '激活视图', description: '显示激活强度热力图', icon: '✦' },
  { id: 'parameter', name: '参数视图', description: '显示参数分布', icon: '◉' },
  { id: 'feature', name: '特征图视图', description: '显示CNN特征图/卷积核/注意力', icon: '◈' },
];

/* ---------- 各层类型的颜色编码 ---------- */
export const LAYER_COLORS: Record<LayerCategory, THREE.Color> = {
  input:   new THREE.Color(0x76b900),   // 青绿色
  conv:    new THREE.Color(0x7c3aed),   // 紫色
  pool:    new THREE.Color(0x06b6d4),   // 青色
  fc:      new THREE.Color(0xf97316),     // 橙色
  output:  new THREE.Color(0xef4444),         // 红色
  norm:    new THREE.Color(0xeab308),     // 金黄色
  dropout: new THREE.Color(0x9ca3af),     // 灰色
};

export const LAYER_COLORS_HEX: Record<LayerCategory, string> = {
  input: '#76b900',
  conv: '#7c3aed',
  pool: '#06b6d4',
  fc: '#f97316',
  output: '#ef4444',
  norm: '#eab308',
  dropout: '#9ca3af',
};

/* ---------- 默认 CNN 架构 ---------- */
export const DEFAULT_ARCHITECTURE: NetworkArchitecture = {
  name: 'SimpleCNN',
  layers: [
    { id: 'input',    name: 'Input',    type: 'input',  nodeCount: 1,  inputShape: [28, 28, 1],  outputShape: [28, 28, 1],  params: 0 },
    { id: 'conv1',    name: 'Conv1',    type: 'conv',   nodeCount: 32, inputShape: [28, 28, 1],  outputShape: [26, 26, 32], params: 320,      kernelSize: 3, activation: 'ReLU' },
    { id: 'pool1',    name: 'Pool1',    type: 'pool',   nodeCount: 32, inputShape: [26, 26, 32], outputShape: [13, 13, 32], params: 0,         activation: 'MaxPool' },
    { id: 'conv2',    name: 'Conv2',    type: 'conv',   nodeCount: 64, inputShape: [13, 13, 32], outputShape: [11, 11, 64], params: 18496,   kernelSize: 3, activation: 'ReLU' },
    { id: 'pool2',    name: 'Pool2',    type: 'pool',   nodeCount: 64, inputShape: [11, 11, 64], outputShape: [5, 5, 64],  params: 0,         activation: 'MaxPool' },
    { id: 'fc',       name: 'FC',         type: 'fc',     nodeCount: 8, inputShape: [5 * 5 * 64, 128], outputShape: [128], params: 204928, activation: 'ReLU' },
    { id: 'output',   name: 'Output',     type: 'output', nodeCount: 10, inputShape: [128], outputShape: [10], params: 1290, activation: 'Softmax' },
  ],
};

/* ---------- 层形状格式化工具 ---------- */
/**
 * 将形状数组格式化为显示字符串，兼容任意维度
 * - [28, 28, 1] → "28×28×1"
 * - [128] → "128"
 * - [10] → "10"
 * - [] 或 undefined → "-"
 */
export function formatLayerShape(shape: number[] | undefined | null): string {
  if (!shape || !Array.isArray(shape) || shape.length === 0) {
    return '-';
  }
  const valid = shape.filter((n) => typeof n === 'number' && !isNaN(n) && n > 0);
  if (valid.length === 0) {
    return '-';
  }
  return valid.join('×');
}
