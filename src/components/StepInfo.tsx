import type { TrainingStepData, ActivationStats } from '../types/training';

interface StepInfoProps {
  data: TrainingStepData | null;
  step: number;
}

interface LayerStatDisplayProps {
  name: string;
  stats: ActivationStats['input'] | ActivationStats['conv1'] | ActivationStats['fc'];
  color: string;
}

function LayerStatDisplay({ name, stats, color }: LayerStatDisplayProps) {
  return (
    <div className="layer-stat-card">
      <div className="layer-stat-header">
        <span className="layer-indicator" style={{ backgroundColor: color }} />
        <span className="layer-name">{name}</span>
      </div>
      <div className="layer-stat-grid">
        <div className="stat-item">
          <span className="stat-label">Mean</span>
          <span className="stat-value">{stats.avg.toFixed(3)}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Max</span>
          <span className="stat-value">{stats.max.toFixed(3)}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Min</span>
          <span className="stat-value">{stats.min.toFixed(3)}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">Sparse</span>
          <span className="stat-value">{(stats.sparsity * 100).toFixed(1)}%</span>
        </div>
      </div>
    </div>
  );
}

export function StepInfo({ data, step }: StepInfoProps) {
  if (!data) {
    return (
      <div className="step-info card">
        <div className="card-header">
          <h2 className="card-title">训练状态</h2>
        </div>
        <div className="empty-state">
          <span>等待数据加载...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="step-info card">
      <div className="card-header">
        <h2 className="card-title">训练状态</h2>
        <span className="step-badge">Step {step}</span>
      </div>

      <div className="metrics-grid">
        <div className="metric-card loss">
          <span className="metric-label">Train Loss</span>
          <span className="metric-value">{data.trainLoss.toFixed(4)}</span>
        </div>
        <div className="metric-card loss-val">
          <span className="metric-label">Val Loss</span>
          <span className="metric-value">{data.valLoss.toFixed(4)}</span>
        </div>
        <div className="metric-card accuracy">
          <span className="metric-label">Train Acc</span>
          <span className="metric-value">{(data.trainAccuracy * 100).toFixed(2)}%</span>
        </div>
        <div className="metric-card accuracy-val">
          <span className="metric-label">Val Acc</span>
          <span className="metric-value">{(data.valAccuracy * 100).toFixed(2)}%</span>
        </div>
        <div className="metric-card lr">
          <span className="metric-label">Learning Rate</span>
          <span className="metric-value lr-value">{data.learningRate ? data.learningRate.toFixed(4) : '0.0000'}</span>
        </div>
      </div>

      <div className="activation-section">
        <h3 className="section-title">层激活统计</h3>
        <div className="activation-list">
          <LayerStatDisplay
            name="输入层 (Input)"
            stats={data.activationStats.input}
            color="#4285f4"
          />
          <LayerStatDisplay
            name="卷积层 1 (Conv1)"
            stats={data.activationStats.conv1}
            color="#34a853"
          />
          {data.activationStats.conv2 && (
            <LayerStatDisplay
              name="卷积层 2 (Conv2)"
              stats={data.activationStats.conv2}
              color="#fbbc04"
            />
          )}
          <LayerStatDisplay
            name="全连接层 (FC)"
            stats={data.activationStats.fc}
            color="#ea4335"
          />
        </div>
      </div>
    </div>
  );
}
