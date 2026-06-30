import { useMemo } from 'react';
import type { TrainingStepData } from '../types/training';

import type { ChartColors } from '../hooks/useColorConfig';

interface ModelMetricsPanelProps {
  data: TrainingStepData | null;
  visibleData?: TrainingStepData[];
  colors: ChartColors;
}

interface MetricHint {
  text: string;
  type: 'good' | 'warning' | 'info';
}

export function ModelMetricsPanel({ data, visibleData = [], colors: _colors }: ModelMetricsPanelProps) {
  const hints = useMemo((): MetricHint[] => {
    if (!data || visibleData.length < 2) return [];
    
    const hintsList: MetricHint[] = [];
    const recentWindow = visibleData.slice(-10);
    
    const recentLossTrend = recentWindow[recentWindow.length - 1].trainLoss - recentWindow[0].trainLoss;
    const recentF1Trend = recentWindow[recentWindow.length - 1].f1Score - recentWindow[0].f1Score;
    
    if (data.valLoss > data.trainLoss * 1.2) {
      hintsList.push({
        text: '验证损失高于训练损失，可能存在过拟合',
        type: 'warning'
      });
    }
    
    if (data.trainAccuracy - data.valAccuracy > 0.1) {
      hintsList.push({
        text: '训练-验证准确率差距较大，泛化能力待提升',
        type: 'warning'
      });
    }
    
    if (recentF1Trend > 0.02 && data.step > 20) {
      hintsList.push({
        text: 'F1 分数持续改善，分类性能正在提升',
        type: 'good'
      });
    }
    
    if (data.learningRate < 0.0001 && data.step > 50) {
      hintsList.push({
        text: '学习率已较低，模型进入精调阶段',
        type: 'info'
      });
    }
    
    if (data.gradientNorm > 5) {
      hintsList.push({
        text: '梯度范数过大，存在梯度爆炸风险',
        type: 'warning'
      });
    }
    
    if (data.gradientNorm < 0.05) {
      hintsList.push({
        text: '梯度范数过小，可能出现梯度消失',
        type: 'warning'
      });
    }
    
    if (recentLossTrend > 0 && data.step > 30) {
      hintsList.push({
        text: '训练损失未下降，训练可能陷入停滞',
        type: 'warning'
      });
    }
    
    // 可选链保护：activationStats / conv1 可能为 undefined 时不崩溃
    if (data?.activationStats?.conv1?.sparsity != null && data.activationStats.conv1.sparsity > 0.7) {
      hintsList.push({
        text: '卷积层激活稀疏度过高，可能存在死神经元',
        type: 'info'
      });
    }
    
    return hintsList;
  }, [data, visibleData]);

  if (!data) {
    return (
      <div className="model-metrics-panel card">
        <div className="card-header">
          <h2 className="card-title">模型指标</h2>
        </div>
        <div className="empty-state">
          <span>等待数据加载...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="model-metrics-panel card">
      <div className="card-header">
        <h2 className="card-title">模型指标</h2>
        <span className="card-subtitle">Step {data.step}</span>
      </div>
      <div className="metrics-grid-full">
        <div className="metric-section">
          <div className="metric-section-title">损失函数</div>
          <div className="metric-row">
            <span className="metric-name">Train Loss</span>
            <span className="metric-number loss">{data.trainLoss.toFixed(4)}</span>
          </div>
          <div className="metric-row">
            <span className="metric-name">Val Loss</span>
            <span className="metric-number loss-val">{data.valLoss.toFixed(4)}</span>
          </div>
        </div>
        
        <div className="metric-section">
          <div className="metric-section-title">准确率</div>
          <div className="metric-row">
            <span className="metric-name">Train Acc</span>
            <span className="metric-number accuracy">{(data.trainAccuracy * 100).toFixed(2)}%</span>
          </div>
          <div className="metric-row">
            <span className="metric-name">Val Acc</span>
            <span className="metric-number accuracy-val">{(data.valAccuracy * 100).toFixed(2)}%</span>
          </div>
        </div>
        
        <div className="metric-divider" />
        
        <div className="metric-section">
          <div className="metric-section-title">分类指标</div>
          <div className="metric-row">
            <span className="metric-name">Precision</span>
            <span className="metric-number precision">{(data.precision * 100).toFixed(2)}%</span>
          </div>
          <div className="metric-row">
            <span className="metric-name">Recall</span>
            <span className="metric-number recall">{(data.recall * 100).toFixed(2)}%</span>
          </div>
          <div className="metric-row">
            <span className="metric-name">F1 Score</span>
            <span className="metric-number f1">{(data.f1Score * 100).toFixed(2)}%</span>
          </div>
        </div>
        
        <div className="metric-divider" />
        
        <div className="metric-section">
          <div className="metric-section-title">训练参数</div>
          <div className="metric-row">
            <span className="metric-name">Learning Rate</span>
            <span className="metric-number lr">{data.learningRate > 0 ? data.learningRate.toFixed(4) : '—'}</span>
          </div>
          <div className="metric-row">
            <span className="metric-name">Gradient Norm</span>
            <span className={`metric-number grad ${data.gradientNorm > 5 ? 'warning' : data.gradientNorm < 0.1 ? 'info' : ''}`}>
              {data.gradientNorm.toFixed(3)}
            </span>
          </div>
          <div className="metric-row">
            <span className="metric-name">Weight Norm</span>
            <span className="metric-number weight">{data.weightNorm.toFixed(3)}</span>
          </div>
        </div>
        
        {hints.length > 0 && (
          <>
            <div className="metric-divider" />
            <div className="metric-section">
              <div className="metric-section-title">质量提示</div>
              {hints.map((hint, idx) => (
                <div key={idx} className={`hint-item ${hint.type}`}>
                  <span className="hint-icon">{hint.type === 'good' ? '✓' : hint.type === 'warning' ? '!' : 'i'}</span>
                  <span className="hint-text">{hint.text}</span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
