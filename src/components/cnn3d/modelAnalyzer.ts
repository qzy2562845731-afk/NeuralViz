import type { NetworkArchitecture, LayerConfig, LayerCategory } from './types';

export interface ModelAnalysisResult {
  architecture: NetworkArchitecture;
  summary: ModelSummary;
  warnings: ModelWarning[];
}

export interface ModelSummary {
  totalLayers: number;
  totalParams: number;
  inputShape: number[];
  outputShape: number[];
  layerTypes: Record<LayerCategory, number>;
  depth: number;
}

export interface ModelWarning {
  id: string;
  severity: 'info' | 'warning' | 'error';
  message: string;
  layerId?: string;
  suggestion?: string;
}

export interface ParsedLayer {
  id: string;
  name: string;
  type: LayerCategory;
  nodeCount: number;
  inputShape: number[];
  outputShape: number[];
  params: number;
  activation?: string;
  kernelSize?: number;
  group?: string;
}

export function analyzeModel(jsonData: unknown): ModelAnalysisResult {
  const warnings: ModelWarning[] = [];
  const layers: ParsedLayer[] = [];

  try {
    if (typeof jsonData !== 'object' || jsonData === null) {
      throw new Error('Invalid model data: expected object');
    }

    const data = jsonData as Record<string, unknown>;
    
    // 尝试从多种格式解析模型
    const modelLayers = parseLayers(data);
    
    if (modelLayers.length === 0) {
      warnings.push({
        id: 'no-layers',
        severity: 'warning',
        message: '未检测到网络层信息',
        suggestion: '请确保模型JSON包含layers或类似字段',
      });
      return {
        architecture: createDefaultArchitecture(),
        summary: createSummary([]),
        warnings,
      };
    }

    // 验证并转换层
    modelLayers.forEach((layer, index) => {
      const validatedLayer = validateAndConvertLayer(layer, index);
      layers.push(validatedLayer);
      
      // 添加分析警告
      if (layer.params > 500000) {
        warnings.push({
          id: `large-layer-${layer.id}`,
          severity: 'info',
          message: `${layer.name} 参数数量较大 (${formatNumber(layer.params)})`,
          layerId: layer.id,
          suggestion: '考虑使用更小的滤波器或减少通道数',
        });
      }
      
      if (layer.type === 'fc' && layer.nodeCount > 1000) {
        warnings.push({
          id: `large-fc-${layer.id}`,
          severity: 'info',
          message: `${layer.name} 神经元数量较多`,
          layerId: layer.id,
          suggestion: '考虑添加Dropout层防止过拟合',
        });
      }
    });

    // 分析网络深度
    if (layers.length > 20) {
      warnings.push({
        id: 'deep-network',
        severity: 'info',
        message: `网络层数较多 (${layers.length}层)`,
        suggestion: '注意梯度消失问题，考虑使用残差连接',
      });
    }

    const architecture = createArchitecture(layers);
    const summary = createSummary(layers);

    return { architecture, summary, warnings };
    
  } catch (error) {
    warnings.push({
      id: 'parse-error',
      severity: 'error',
      message: `模型解析失败: ${error instanceof Error ? error.message : '未知错误'}`,
      suggestion: '请检查模型JSON格式是否正确',
    });
    
    return {
      architecture: createDefaultArchitecture(),
      summary: createSummary([]),
      warnings,
    };
  }
}

function parseLayers(data: Record<string, unknown>): ParsedLayer[] {
  const layers: ParsedLayer[] = [];
  
  // 尝试多种格式
  const getLayers = (key: string) => {
    const value = data[key];
    return value && typeof value === 'object'
      ? (value as Record<string, unknown>)['layers']
      : undefined;
  };

  const possibleSources = [
    data['layers'],
    getLayers('model'),
    getLayers('config'),
    getLayers('network'),
  ];

  for (const source of possibleSources) {
    if (Array.isArray(source)) {
      source.forEach((item, index) => {
        const layer = parseLayerItem(item, index);
        if (layer) layers.push(layer);
      });
      return layers;
    }
  }

  return layers;
}

function parseLayerItem(item: unknown, index: number): ParsedLayer | null {
  if (typeof item !== 'object' || item === null) return null;
  
  const layer = item as Record<string, unknown>;
  const name = String(layer['name'] || `layer_${index}`);
  const typeStr = String(layer['type'] || layer['class_name'] || 'conv').toLowerCase();
  
  const type = mapLayerType(typeStr);

  // fe9修复：当 JSON 未提供 params 时，根据层类型和形状自动计算参数量（含偏置）
  let params = Number(layer['params'] || layer['param_count'] || 0);
  if (params === 0) {
    params = calcLayerParams(layer, type);
  }

  return {
    id: String(layer['id'] || `layer_${index}`),
    name,
    type,
    nodeCount: Number(layer['nodeCount'] || layer['units'] || layer['filters'] || layer['channels'] || layer['out_features'] || layer['outFeatures'] || 32),
    inputShape: parseShape(layer['inputShape'] || layer['input_shape'] || [28, 28, 1]),
    outputShape: parseShape(layer['outputShape'] || layer['output_shape'] || [28, 28, 32]),
    params,
    activation: String(layer['activation'] || ''),
    kernelSize: parseKernelSize(layer['kernelSize'] || layer['kernel_size']),
    group: String(layer['group'] || ''),
  };
}

/**
 * fe9修复：根据层类型和输入/输出形状自动计算参数量（含偏置）
 * 仅在 JSON 未提供 params 字段时作为兜底使用
 */
function calcLayerParams(layer: any, type: LayerCategory): number {
  try {
    const inShape = parseShape(layer['inputShape'] || layer['input_shape'] || []);
    const outShape = parseShape(layer['outputShape'] || layer['output_shape'] || []);
    const kernelSize = parseKernelSize(layer['kernelSize'] || layer['kernel_size']);
    const hasBias = layer['bias'] !== false; // 默认有偏置

    if (type === 'conv' && kernelSize > 0 && inShape.length >= 3 && outShape.length >= 3) {
      // 卷积层: out_channels * in_channels * kh * kw + out_channels
      const inChannels = inShape[inShape.length - 1] || inShape[0] || 1;
      const outChannels = outShape[outShape.length - 1] || outShape[0] || 1;
      return outChannels * inChannels * kernelSize * kernelSize + (hasBias ? outChannels : 0);
    }

    if (type === 'fc' && inShape.length >= 1 && outShape.length >= 1) {
      // 全连接层: in_features * out_features + out_features
      const inFeatures = inShape.reduce((a: number, b: number) => a * b, 1);
      const outFeatures = outShape.reduce((a: number, b: number) => a * b, 1);
      return inFeatures * outFeatures + (hasBias ? outFeatures : 0);
    }

    // input/pool/dropout/output 等层无参数
    return 0;
  } catch {
    return 0;
  }
}

function mapLayerType(typeStr: string): LayerCategory {
  const typeMap: Record<string, LayerCategory> = {
    // 基础类型
    'input': 'input',
    'inputlayer': 'input',
    'conv': 'conv',
    'convolutional': 'conv',
    'conv2d': 'conv',
    'convolution2d': 'conv',
    'conv1d': 'conv',
    'conv3d': 'conv',
    'depthwiseconv': 'conv',
    'separableconv': 'conv',
    // ONNX 类型
    'convtranspose': 'conv',
    'convtranspose2d': 'conv',
    'gemm': 'fc',
    'matmul': 'fc',
    'flatten': 'conv',
    'reshape': 'conv',
    'relu': 'conv',
    'leakyrelu': 'conv',
    'prelu': 'conv',
    'elu': 'conv',
    'sigmoid': 'conv',
    'tanh': 'conv',
    'gelu': 'conv',
    'silu': 'conv',
    'add': 'conv',
    'mul': 'conv',
    'concat': 'conv',
    'split': 'conv',
    'transpose': 'conv',
    // PyTorch 类型
    'linear': 'fc',
    'bilinear': 'fc',
    'maxpool2d': 'pool',
    'maxpool1d': 'pool',
    'maxpool3d': 'pool',
    'avgpool2d': 'pool',
    'avgpool1d': 'pool',
    'avgpool3d': 'pool',
    'adaptivemaxpool2d': 'pool',
    'adaptiveavgpool2d': 'pool',
    'adaptivemaxpool': 'pool',
    'adaptiveavgpool': 'pool',
    'pool': 'pool',
    'pooling': 'pool',
    'maxpool': 'pool',
    'maxpooling': 'pool',
    'avgpool': 'pool',
    'avgpooling': 'pool',
    'fc': 'fc',
    'dense': 'fc',
    'fullyconnected': 'fc',
    'output': 'output',
    'softmax': 'output',
    'classifier': 'output',
    'logsoftmax': 'output',
    'norm': 'norm',
    'batchnorm': 'norm',
    'batchnorm2d': 'norm',
    'batchnorm1d': 'norm',
    'batchnormalization': 'norm',
    'layernorm': 'norm',
    'groupnorm': 'norm',
    'instancenorm': 'norm',
    'dropout': 'dropout',
    'drop': 'dropout',
    'dropout2d': 'dropout',
    'dropout1d': 'dropout',
    'flattenlayer': 'conv',
    'identity': 'conv',
    'relu6': 'conv',
    'hardsigmoid': 'conv',
    'hardswish': 'conv',
    'mish': 'conv',
    'selu': 'conv',
    'batchnorm3d': 'norm',
  };
  
  return typeMap[typeStr] || 'conv';
}

function parseShape(shape: unknown): number[] {
  if (Array.isArray(shape)) {
    return shape.map(s => Number(s)).filter(n => !isNaN(n));
  }
  if (typeof shape === 'string') {
    try {
      const parsed = JSON.parse(shape);
      if (Array.isArray(parsed)) {
        return parsed.map(s => Number(s)).filter(n => !isNaN(n));
      }
    } catch {
      // ignore
    }
  }
  return [28, 28, 1];
}

function parseKernelSize(kernelSize: unknown): number {
  if (typeof kernelSize === 'number') {
    return kernelSize;
  }
  if (Array.isArray(kernelSize) && kernelSize.length > 0) {
    const first = Number(kernelSize[0]);
    return isNaN(first) ? 3 : first;
  }
  if (typeof kernelSize === 'string') {
    const num = parseInt(kernelSize, 10);
    return isNaN(num) ? 3 : num;
  }
  return 3;
}

function validateAndConvertLayer(layer: ParsedLayer, index: number): ParsedLayer {
  let { type } = layer;
  
  // 确保类型有效
  const validTypes: LayerCategory[] = ['input', 'conv', 'pool', 'fc', 'output', 'norm', 'dropout'];
  if (!validTypes.includes(type)) {
    type = 'conv';
  }
  
  // 确保形状有效
  const ensureShape = (shape: number[]): number[] => {
    while (shape.length < 3) shape.push(1);
    return shape.slice(0, 3);
  };
  
  return {
    ...layer,
    id: layer.id || `layer_${index}`,
    type,
    inputShape: ensureShape(layer.inputShape),
    outputShape: ensureShape(layer.outputShape),
    nodeCount: Math.max(1, layer.nodeCount),
  };
}

function createArchitecture(layers: ParsedLayer[]): NetworkArchitecture {
  const configLayers: LayerConfig[] = layers.map((layer): LayerConfig => ({
    id: layer.id,
    name: layer.name,
    type: layer.type,
    nodeCount: layer.nodeCount,
    inputShape: layer.inputShape,
    outputShape: layer.outputShape,
    params: layer.params,
    activation: layer.activation || undefined,
    kernelSize: layer.kernelSize || undefined,
    group: layer.group || undefined,
  }));

  return {
    name: 'Custom Model',
    layers: configLayers,
  };
}

function createSummary(layers: ParsedLayer[]): ModelSummary {
  const layerTypes: Record<LayerCategory, number> = {
    input: 0, conv: 0, pool: 0, fc: 0, output: 0, norm: 0, dropout: 0,
  };
  
  layers.forEach(layer => {
    layerTypes[layer.type]++;
  });

  return {
    totalLayers: layers.length,
    totalParams: layers.reduce((sum, l) => sum + l.params, 0),
    inputShape: layers[0]?.inputShape || [0, 0, 0],
    outputShape: layers[layers.length - 1]?.outputShape || [0, 0, 0],
    layerTypes,
    depth: layers.length,
  };
}

function createDefaultArchitecture(): NetworkArchitecture {
  return {
    name: 'Default CNN',
    layers: [
      { id: 'input', name: 'Input', type: 'input', nodeCount: 1, inputShape: [28, 28, 1], outputShape: [28, 28, 1], params: 0 },
      { id: 'conv1', name: 'Conv1', type: 'conv', nodeCount: 32, inputShape: [28, 28, 1], outputShape: [26, 26, 32], params: 320, kernelSize: 3, activation: 'ReLU' },
      { id: 'pool1', name: 'Pool1', type: 'pool', nodeCount: 32, inputShape: [26, 26, 32], outputShape: [13, 13, 32], params: 0 },
      { id: 'conv2', name: 'Conv2', type: 'conv', nodeCount: 64, inputShape: [13, 13, 32], outputShape: [11, 11, 64], params: 18496, kernelSize: 3, activation: 'ReLU' },
      { id: 'pool2', name: 'Pool2', type: 'pool', nodeCount: 64, inputShape: [11, 11, 64], outputShape: [5, 5, 64], params: 0 },
      { id: 'fc', name: 'FC', type: 'fc', nodeCount: 128, inputShape: [5 * 5 * 64], outputShape: [128], params: 204928, activation: 'ReLU' },
      { id: 'output', name: 'Output', type: 'output', nodeCount: 10, inputShape: [128], outputShape: [10], params: 1290, activation: 'Softmax' },
    ],
  };
}

function formatNumber(num: number): string {
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`;
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`;
  return num.toString();
}

export function generateSampleModel(): ModelAnalysisResult {
  const layers: ParsedLayer[] = [
    { id: 'input', name: 'Input', type: 'input', nodeCount: 1, inputShape: [28, 28, 1], outputShape: [28, 28, 1], params: 0 },
    { id: 'conv1', name: 'Conv1', type: 'conv', nodeCount: 32, inputShape: [28, 28, 1], outputShape: [26, 26, 32], params: 320, kernelSize: 3, activation: 'ReLU' },
    { id: 'bn1', name: 'BN1', type: 'norm', nodeCount: 32, inputShape: [26, 26, 32], outputShape: [26, 26, 32], params: 128 },
    { id: 'pool1', name: 'Pool1', type: 'pool', nodeCount: 32, inputShape: [26, 26, 32], outputShape: [13, 13, 32], params: 0 },
    { id: 'conv2', name: 'Conv2', type: 'conv', nodeCount: 64, inputShape: [13, 13, 32], outputShape: [11, 11, 64], params: 18496, kernelSize: 3, activation: 'ReLU' },
    { id: 'bn2', name: 'BN2', type: 'norm', nodeCount: 64, inputShape: [11, 11, 64], outputShape: [11, 11, 64], params: 256 },
    { id: 'pool2', name: 'Pool2', type: 'pool', nodeCount: 64, inputShape: [11, 11, 64], outputShape: [5, 5, 64], params: 0 },
    { id: 'conv3', name: 'Conv3', type: 'conv', nodeCount: 128, inputShape: [5, 5, 64], outputShape: [3, 3, 128], params: 73856, kernelSize: 3, activation: 'ReLU' },
    { id: 'dropout1', name: 'Dropout', type: 'dropout', nodeCount: 128, inputShape: [3, 3, 128], outputShape: [3, 3, 128], params: 0 },
    { id: 'fc1', name: 'FC1', type: 'fc', nodeCount: 256, inputShape: [3 * 3 * 128], outputShape: [256], params: 295168, activation: 'ReLU' },
    { id: 'fc2', name: 'FC2', type: 'fc', nodeCount: 128, inputShape: [256], outputShape: [128], params: 32896, activation: 'ReLU' },
    { id: 'output', name: 'Output', type: 'output', nodeCount: 10, inputShape: [128], outputShape: [10], params: 1290, activation: 'Softmax' },
  ];

  return {
    architecture: createArchitecture(layers),
    summary: createSummary(layers),
    warnings: [],
  };
}