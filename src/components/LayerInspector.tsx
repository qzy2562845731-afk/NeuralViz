import type { TrainingStepData, LayerName } from '../types/training';

export interface LayerStats {
  name: string;
  type: 'input' | 'conv' | 'fc' | 'output';
  activationMean: number;
  activationMax: number;
  activationMin: number;
  sparsity: number;
  weightNorm: number;
  gradientNorm: number;
  nodeCount: number;
}

interface LayerInspectorProps {
  selectedLayer: { layerId: string; layerInfo: LayerStats } | null;
  currentStep: number;
  currentData: TrainingStepData | null;
  onLayerSelect?: (selected: { layerId: string; layerInfo: LayerStats } | null) => void;
}

// 网络架构配置
const NETWORK_ARCHITECTURE: Array<{
  name: string;
  type: LayerName;
  inputShape: string;
  outputShape: string;
  params: string;
  activation: string;
  description: string;
  role: string;
}> = [
  { 
    name: 'Input', 
    type: 'input', 
    inputShape: '-',
    outputShape: '28×28×1',
    params: '0',
    activation: '-',
    description: 'Grayscale image input',
    role: '接收原始输入图像,将 28×28 像素的灰度图像转换为网络可处理的数据格式'
  },
  { 
    name: 'Conv1', 
    type: 'conv1', 
    inputShape: '28×28×1',
    outputShape: '26×26×32',
    params: '320',
    activation: 'ReLU',
    description: '3×3 conv, 32 filters',
    role: '第一层卷积,使用 32 个 3×3 卷积核提取低级特征(边缘、纹理等),输出通道数 32'
  },
  { 
    name: 'Conv2', 
    type: 'conv2', 
    inputShape: '13×13×32',
    outputShape: '11×11×64',
    params: '18,496',
    activation: 'ReLU',
    description: '3×3 conv, 64 filters',
    role: '第二层卷积,使用 64 个 3×3 卷积核提取中级特征(形状、模式等),通过池化降维后输入'
  },
  { 
    name: 'Features', 
    type: 'conv2', 
    inputShape: '11×11×64',
    outputShape: '6×6×16',
    params: '9,232',
    activation: 'ReLU',
    description: '3×3 conv, 16 filters',
    role: '特征提取层,使用 16 个 3×3 卷积核进一步提取高级语义特征,经过池化后维度降低'
  },
  { 
    name: 'FC', 
    type: 'fc', 
    inputShape: '11×11×64',
    outputShape: '128',
    params: '984,320',
    activation: 'ReLU',
    description: 'Fully connected layer',
    role: '全连接层,将卷积特征展平后映射到 128 维特征空间,进行高层语义特征组合'
  },
  { 
    name: 'Output', 
    type: 'fc', 
    inputShape: '128',
    outputShape: '10',
    params: '1,290',
    activation: 'Softmax',
    description: '10-class classification',
    role: '输出层,使用 Softmax 激活函数输出 10 个类别的概率分布,用于最终分类预测'
  },
];

function getLayerHealth(stats: LayerStats): 'healthy' | 'warning' | 'critical' {
  if (stats.sparsity > 0.8) return 'critical';
  if (stats.gradientNorm > 5 || stats.gradientNorm < 0.01) return 'warning';
  if (stats.sparsity > 0.5) return 'warning';
  return 'healthy';
}

function getHealthColor(health: 'healthy' | 'warning' | 'critical'): string {
  switch (health) {
    case 'healthy': return 'oklch(0.68 0.16 155)';
    case 'warning': return 'oklch(0.72 0.16 85)';
    case 'critical': return 'oklch(0.60 0.20 25)';
  }
}

export function LayerInspector({ selectedLayer, currentStep, currentData, onLayerSelect }: LayerInspectorProps) {
  // 选中具体层时的详情视图
  if (selectedLayer) {
    const { layerInfo } = selectedLayer;
    const health = getLayerHealth(layerInfo);
    const healthColor = getHealthColor(health);
    const gradientRatio = layerInfo.weightNorm > 0
      ? layerInfo.gradientNorm / layerInfo.weightNorm
      : 0;

    // 查找对应的架构信息
    const archInfo = NETWORK_ARCHITECTURE.find(arch => arch.name === layerInfo.name);
    
    // 生成状态描述
    const getStatusDescription = () => {
      if (health === 'critical') return '该层激活值稀疏度较高，可能存在神经元死亡';
      if (health === 'warning') return '该层梯度或激活值分布需要关注';
      return '该层运行状态正常，激活值分布合理';
    };

    return (
      <div className="flex h-full flex-col">
        <div className="flex flex-shrink-0 items-center justify-between border-b border-white/10 pb-3">
          <div className="flex items-center gap-2">
            <h3 className="text-base font-semibold tracking-tight">{layerInfo.name}</h3>
            <span className={`badge badge-${layerInfo.type}`}>
              {layerInfo.type.toUpperCase()}
            </span>
          </div>
          <div className="inspector-health-dot" style={{ backgroundColor: healthColor }} />
        </div>

        {/* 层作用说明 */}
        {archInfo && (
          <div className="inspector-role-section">
            <div className="inspector-role-text">{archInfo.role}</div>
          </div>
        )}

        {/* 架构信息 */}
        {archInfo && (
          <div className="inspector-arch-section">
            <div className="inspector-section-label">Architecture</div>
            <div className="inspector-arch-grid">
              <div className="inspector-arch-item">
                <span className="inspector-arch-label">Input Shape</span>
                <span className="inspector-arch-value">{archInfo.inputShape}</span>
              </div>
              <div className="inspector-arch-item">
                <span className="inspector-arch-label">Output Shape</span>
                <span className="inspector-arch-value">{archInfo.outputShape}</span>
              </div>
              <div className="inspector-arch-item">
                <span className="inspector-arch-label">Parameters</span>
                <span className="inspector-arch-value">{archInfo.params}</span>
              </div>
              <div className="inspector-arch-item">
                <span className="inspector-arch-label">Activation</span>
                <span className="inspector-arch-value">{archInfo.activation}</span>
              </div>
            </div>
          </div>
        )}

        {/* 摘要区 */}
        <div className="inspector-summary">
          <div className="inspector-summary-item">
            <span className="inspector-summary-label">Mean Activation</span>
            <span className="inspector-summary-value">{layerInfo.activationMean.toFixed(4)}</span>
          </div>
          <div className="inspector-summary-item">
            <span className="inspector-summary-label">Nodes</span>
            <span className="inspector-summary-value">{layerInfo.nodeCount}</span>
          </div>
          <div className="inspector-summary-item">
            <span className="inspector-summary-label">Sparsity</span>
            <span className="inspector-summary-value">{(layerInfo.sparsity * 100).toFixed(1)}%</span>
          </div>
        </div>

        {/* 激活值分布条 */}
        <div className="inspector-activation-section">
          <div className="inspector-section-label">Activation Range</div>
          <div className="inspector-range-bar">
            <div
              className="inspector-range-fill"
              style={{
                left: `${Math.max(0, (layerInfo.activationMin + 1) * 50)}%`,
                width: `${Math.min(100, (layerInfo.activationMax - layerInfo.activationMin) * 50)}%`,
                backgroundColor: healthColor,
              }}
            />
          </div>
          <div className="inspector-range-labels">
            <span>{layerInfo.activationMin.toFixed(3)}</span>
            <span>0</span>
            <span>{layerInfo.activationMax.toFixed(3)}</span>
          </div>
        </div>

        {/* 详细指标 */}
        <div className="inspector-detail-section">
          <div className="inspector-section-label">Training Metrics</div>
          <div className="inspector-detail-grid">
            <div className="inspector-detail-item">
              <span className="inspector-detail-label">Weight Norm</span>
              <span className="inspector-detail-value">{layerInfo.weightNorm.toFixed(4)}</span>
            </div>
            <div className="inspector-detail-item">
              <span className="inspector-detail-label">Gradient Norm</span>
              <span className="inspector-detail-value">{layerInfo.gradientNorm.toFixed(4)}</span>
            </div>
            <div className="inspector-detail-item">
              <span className="inspector-detail-label">Grad/Weight</span>
              <span className="inspector-detail-value">{gradientRatio.toFixed(6)}</span>
            </div>
          </div>
        </div>

        {/* 状态描述 */}
        <div className="inspector-status-section">
          <div className="inspector-section-label">Status</div>
          <div className="inspector-status-text">{getStatusDescription()}</div>
        </div>

        {/* Step 指示 */}
        <div className="inspector-step-indicator">
          <span className="inspector-step-label">Current Step</span>
          <span className="inspector-step-value">{currentStep}</span>
        </div>
      </div>
    );
  }

  // 模型概览（未选中时）
  const trainLoss = currentData?.trainLoss ?? 0;
  const valLoss = currentData?.valLoss ?? 0;
  const lossGap = valLoss - trainLoss;
  const isOverfitting = lossGap > trainLoss * 0.2;

  // 计算总参数量
  const totalParams = NETWORK_ARCHITECTURE.reduce((sum, layer) => {
    const paramStr = layer.params.replace(/,/g, '');
    const paramNum = parseInt(paramStr, 10);
    return sum + (isNaN(paramNum) ? 0 : paramNum);
  }, 0);

  // 添加一个辅助函数来映射 layer name 到 layerId
  const nameToId: Record<string, string> = {
    'Input': 'input',
    'Conv1': 'conv1', 
    'Conv2': 'conv2',
    'Features': 'features',
    'FC': 'fc',
    'Output': 'output',
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex flex-shrink-0 items-center justify-between border-b border-white/10 pb-3">
        <h3 className="text-base font-semibold tracking-tight">Layer Inspector</h3>
        <span className="rounded-full border border-green-500/20 bg-green-500/10 px-2.5 py-1 font-mono text-xs text-green-400">Step {currentStep}</span>
      </div>

      {/* 模型摘要 */}
      <div className="inspector-model-summary">
        <div className="inspector-section-label">Model Overview</div>
        <div className="inspector-model-stats">
          <div className="inspector-model-stat">
            <span className="inspector-model-stat-label">Layers</span>
            <span className="inspector-model-stat-value">{NETWORK_ARCHITECTURE.length}</span>
          </div>
          <div className="inspector-model-stat">
            <span className="inspector-model-stat-label">Parameters</span>
            <span className="inspector-model-stat-value">{(totalParams / 1000).toFixed(1)}K</span>
          </div>
        </div>
      </div>

      {/* 训练状态摘要 */}
      {currentData && (
        <div className="inspector-overview">
          <div className="inspector-section-label">Training Status</div>
          <div className="inspector-overview-row">
            <div className="inspector-overview-item">
              <span className="inspector-overview-label">Loss</span>
              <span className="inspector-overview-value loss">{trainLoss.toFixed(4)}</span>
            </div>
            <div className="inspector-overview-item">
              <span className="inspector-overview-label">Accuracy</span>
              <span className="inspector-overview-value accuracy">
                {(currentData.trainAccuracy * 100).toFixed(1)}%
              </span>
            </div>
            <div className="inspector-overview-item">
              <span className="inspector-overview-label">F1</span>
              <span className="inspector-overview-value">{currentData.f1Score.toFixed(3)}</span>
            </div>
          </div>
          {isOverfitting && (
            <div className="inspector-overfit-warning">
              <span className="inspector-warning-icon">!</span>
              <span>Train-Val loss gap: {lossGap.toFixed(4)} — possible overfitting</span>
            </div>
          )}
        </div>
      )}

      {/* 网络架构列表 */}
      <div className="inspector-architecture">
        <div className="inspector-section-label">Network Architecture</div>
        <div className="inspector-layer-list">
          {NETWORK_ARCHITECTURE.map((layer, idx) => (
            <div 
              key={idx} 
              className="inspector-layer-row"
              onClick={() => {
                if (onLayerSelect) {
                  const layerId = nameToId[layer.name] || layer.name.toLowerCase();
                  onLayerSelect({
                    layerId,
                    layerInfo: {
                      name: layer.name,
                      type: layer.type === 'conv1' || layer.type === 'conv2' ? 'conv' : layer.type as 'input' | 'conv' | 'fc' | 'output',
                      activationMean: 0,
                      activationMax: 0,
                      activationMin: 0,
                      sparsity: 0,
                      weightNorm: currentData?.weightNorm ?? 0,
                      gradientNorm: currentData?.gradientNorm ?? 0,
                      nodeCount: 32,
                    },
                  });
                }
              }}
            >
              <div className="inspector-layer-index">{idx + 1}</div>
              <div className="inspector-layer-info">
                <span className="inspector-layer-name">{layer.name}</span>
                <span className="inspector-layer-desc">{layer.description}</span>
                <span className="inspector-layer-meta">
                  {layer.inputShape} → {layer.outputShape} | {layer.params} params | {layer.activation}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className="inspector-footer-hint">
        Click a layer in the 3D view to inspect details
      </div>
    </div>
  );
}
