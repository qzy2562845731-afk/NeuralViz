import { useMemo } from 'react';
import type { TrainingStepData } from '../types/training';
import type { ChartColors } from '../hooks/useColorConfig';
import { diagnoseTraining, type DiagnosisResult, type TrainingDiagnosis } from '../utils/diagnosis';

interface DiagnosisPanelProps {
  currentData: TrainingStepData | null;
  visibleData: TrainingStepData[];
  colors: ChartColors;
}

function DiagnosisBadge({ result }: { result: DiagnosisResult }) {
  const severityClass = result.severity;
  
  return (
    <div className={`diagnosis-item ${severityClass}`}>
      <div className="diagnosis-header">
        <span className="diagnosis-icon">
          {result.severity === 'danger' ? '✗' : 
           result.severity === 'warning' ? '!' : 
           result.severity === 'good' ? '✓' : 'i'}
        </span>
        <span className="diagnosis-label">{result.label}</span>
        {result.value && <span className="diagnosis-value">{result.value}</span>}
      </div>
      <p className="diagnosis-description">{result.description}</p>
    </div>
  );
}

function getOverallLabel(overall: TrainingDiagnosis['overall']): string {
  switch (overall) {
    case 'good': return '优秀';
    case 'healthy': return '正常';
    case 'warning': return '注意';
    case 'concerning': return '警告';
    default: return '未知';
  }
}

function getOverallDescription(overall: TrainingDiagnosis['overall']): string {
  switch (overall) {
    case 'good': return '各项指标表现良好';
    case 'healthy': return '各项指标处于正常范围';
    case 'warning': return '部分指标需要关注';
    case 'concerning': return '存在严重问题，需要处理';
    default: return '';
  }
}

export function DiagnosisPanel({ currentData, visibleData, colors: _colors }: DiagnosisPanelProps) {
  const diagnosis = useMemo((): TrainingDiagnosis | null => {
    if (!currentData) return null;
    
    const windowSize = Math.min(10, visibleData.length);
    const windowData = visibleData.slice(-windowSize).map(d => ({
      trainLoss: d.trainLoss,
      valLoss: d.valLoss,
      f1Score: d.f1Score
    }));
    
    return diagnoseTraining(
      {
        step: currentData.step,
        trainLoss: currentData.trainLoss,
        valLoss: currentData.valLoss,
        trainAccuracy: currentData.trainAccuracy,
        valAccuracy: currentData.valAccuracy,
        gradientNorm: currentData.gradientNorm,
        f1Score: currentData.f1Score,
        activationStats: {
          input: { 
            sparsity: currentData.activationStats.input.sparsity,
            max: currentData.activationStats.input.max
          },
          conv1: { 
            sparsity: currentData.activationStats.conv1.sparsity,
            max: currentData.activationStats.conv1.max
          },
          conv2: currentData.activationStats.conv2 ? {
            sparsity: currentData.activationStats.conv2.sparsity,
            max: currentData.activationStats.conv2.max
          } : undefined,
          fc: { 
            sparsity: currentData.activationStats.fc.sparsity,
            max: currentData.activationStats.fc.max
          }
        }
      },
      windowData
    );
  }, [currentData, visibleData]);

  if (!currentData || !diagnosis) {
    return (
      <div className="diagnosis-panel card">
        <div className="card-header">
          <h2 className="card-title">训练诊断</h2>
        </div>
        <div className="empty-state">
          <span>等待数据加载...</span>
        </div>
      </div>
    );
  }

  return (
    <div className="diagnosis-panel card">
      <div className="card-header">
        <h2 className="card-title">训练诊断</h2>
        <span className={`overall-badge ${diagnosis.overall}`}>
          {getOverallLabel(diagnosis.overall)}
        </span>
      </div>
      
      <div className="diagnosis-summary">
        <span className={`summary-indicator ${diagnosis.overall}`}></span>
        <span className="summary-text">{getOverallDescription(diagnosis.overall)}</span>
      </div>
      
      <div className="diagnosis-list">
        {diagnosis.results.length === 0 ? (
          <div className="diagnosis-item good">
            <div className="diagnosis-header">
              <span className="diagnosis-icon">✓</span>
              <span className="diagnosis-label">训练状态良好</span>
            </div>
            <p className="diagnosis-description">各项指标均处于正常范围内</p>
          </div>
        ) : (
          diagnosis.results.map((result, idx) => (
            <DiagnosisBadge key={idx} result={result} />
          ))
        )}
      </div>
      
      <div className="diagnosis-footer">
        <span className="footer-info">
          共 {diagnosis.results.length} 项诊断
          {diagnosis.overall === 'concerning' && ' · 建议检查训练配置'}
          {diagnosis.overall === 'good' && ' · 训练效果优秀'}
        </span>
      </div>
    </div>
  );
}
