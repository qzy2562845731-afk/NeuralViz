import type { TrainingStepData } from '../types/training';

export interface LearningInsight {
  phase: 'initial' | 'learning' | 'refining' | 'converged' | 'overfitting';
  title: string;
  description: string;
  observations: string[];
}

export function generateLearningInsight(current: TrainingStepData, previous: TrainingStepData | null): LearningInsight {
  const observations: string[] = [];
  let phase: LearningInsight['phase'] = 'learning';
  let title = '持续学习中';

  if (current.step <= 10) {
    phase = 'initial';
    title = '初始化阶段';
    observations.push('模型刚开始接收梯度更新，权重随机初始化后的首次调整。');
    observations.push('损失值通常较高，准确率接近随机猜测水平。');
  } else if (current.trainAccuracy > 0.95 && current.valAccuracy > 0.9) {
    phase = 'converged';
    title = '趋于收敛';
    observations.push('训练准确率已接近饱和，模型对训练集拟合充分。');
    observations.push('如果验证指标同步稳定，说明模型学到了可泛化的特征。');
  } else if (current.valLoss > current.trainLoss * 1.25 && current.trainAccuracy - current.valAccuracy > 0.08) {
    phase = 'overfitting';
    title = '过拟合风险';
    observations.push('验证损失明显高于训练损失，模型可能开始记忆训练样本。');
    observations.push('建议引入 Dropout、数据增强或提前停止。');
  } else if (current.learningRate < 0.0002 && current.step > 60) {
    phase = 'refining';
    title = '精细调整';
    observations.push('学习率已降至较低水平，模型在做最后的边界优化。');
    observations.push('此时权重变化较小，损失曲线趋于平缓。');
  }

  // CNN-specific observations
  const conv1Sparsity = current.activationStats.conv1.sparsity;
  if (conv1Sparsity > 0.75) {
    observations.push('Conv1 激活稀疏度过高，部分卷积核可能处于“死亡”状态。');
  } else if (conv1Sparsity < 0.3) {
    observations.push('Conv1 激活较密集，底层特征响应丰富。');
  }

  if (current.gradientNorm > 5) {
    observations.push('梯度范数偏大，当前 step 更新幅度剧烈，需留意梯度爆炸。');
  } else if (current.gradientNorm < 0.05 && current.step > 30) {
    observations.push('梯度范数过小，模型可能进入局部最优或梯度消失区域。');
  }

  if (previous) {
    const lossDelta = previous.trainLoss - current.trainLoss;
    const accDelta = current.trainAccuracy - previous.trainAccuracy;
    if (lossDelta > 0.01) {
      observations.push(`本 step 训练损失下降 ${lossDelta.toFixed(4)}，模型正在有效学习。`);
    } else if (lossDelta < -0.01) {
      observations.push(`本 step 训练损失上升 ${Math.abs(lossDelta).toFixed(4)}，可能受噪声批次影响。`);
    }
    if (accDelta > 0.01) {
      observations.push(`训练准确率提升 ${(accDelta * 100).toFixed(1)}%，分类边界更清晰。`);
    }
  }

  const matrix = current.confusionMatrix;
  if (matrix.length > 0) {
    const diagonalSum = matrix.reduce((sum, row, i) => sum + row[i], 0);
    const total = matrix.flat().reduce((a, b) => a + b, 0);
    const diagonalRatio = diagonalSum / total;
    if (diagonalRatio > 0.8) {
      observations.push(`混淆矩阵对角线占比 ${(diagonalRatio * 100).toFixed(1)}%，类内判别能力较强。`);
    } else {
      observations.push(`混淆矩阵对角线占比 ${(diagonalRatio * 100).toFixed(1)}%，仍有类别间混淆。`);
    }
  }

  // Keep observations unique and limited
  const uniqueObservations = Array.from(new Set(observations)).slice(0, 5);

  return {
    phase,
    title,
    description: `Step ${current.step} 阶段判断：${title}`,
    observations: uniqueObservations,
  };
}

export function summarizeTraining(data: TrainingStepData[]): LearningInsight {
  if (data.length === 0) {
    return {
      phase: 'initial',
      title: '暂无训练数据',
      description: '等待训练开始',
      observations: [],
    };
  }

  const first = data[0];
  const last = data[data.length - 1];
  const bestValAcc = Math.max(...data.map((d) => d.valAccuracy));
  const bestValStep = data.find((d) => d.valAccuracy === bestValAcc)?.step ?? last.step;

  const lossDrop = first.trainLoss - last.trainLoss;
  const accGain = last.trainAccuracy - first.trainAccuracy;
  const overfitGap = last.trainAccuracy - last.valAccuracy;

  const observations: string[] = [];
  observations.push(`训练共 ${data.length} 个 step，训练损失从 ${first.trainLoss.toFixed(3)} 降至 ${last.trainLoss.toFixed(3)}。`);
  observations.push(`训练准确率从 ${(first.trainAccuracy * 100).toFixed(1)}% 提升至 ${(last.trainAccuracy * 100).toFixed(1)}%。`);
  observations.push(`最佳验证准确率 ${(bestValAcc * 100).toFixed(1)}% 出现在 Step ${bestValStep}。`);

  if (overfitGap > 0.1) {
    observations.push(`训练-验证准确率差距 ${(overfitGap * 100).toFixed(1)}%，存在过拟合迹象。`);
  } else if (overfitGap < 0.03) {
    observations.push('训练与验证准确率接近，模型泛化能力良好。');
  }

  if (lossDrop > 1.5) {
    observations.push('损失下降显著，模型有效学习了数据分布。');
  }

  if (accGain > 0.8) {
    observations.push('准确率大幅提升，分类任务基本掌握。');
  }

  let phase: LearningInsight['phase'] = 'learning';
  if (overfitGap > 0.1) phase = 'overfitting';
  else if (last.trainAccuracy > 0.95 && last.valAccuracy > 0.9) phase = 'converged';
  else if (last.learningRate < 0.0002) phase = 'refining';

  return {
    phase,
    title: phase === 'converged' ? '训练完成并收敛' : phase === 'overfitting' ? '训练完成但过拟合' : '训练总结',
    description: `整体训练趋势：${phase === 'converged' ? '已收敛' : phase === 'overfitting' ? '过拟合' : '持续优化'}`,
    observations,
  };
}

export function downloadJSON(data: TrainingStepData[], filename = 'training-log.json'): void {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function downloadCSV(data: TrainingStepData[], filename = 'training-log.csv'): void {
  const headers = [
    'step',
    'loss',
    'accuracy',
    'trainLoss',
    'valLoss',
    'trainAccuracy',
    'valAccuracy',
    'learningRate',
    'precision',
    'recall',
    'f1Score',
    'gradientNorm',
    'weightNorm',
  ];
  const rows = data.map((d) =>
    [
      d.step,
      d.loss,
      d.accuracy,
      d.trainLoss,
      d.valLoss,
      d.trainAccuracy,
      d.valAccuracy,
      d.learningRate,
      d.precision,
      d.recall,
      d.f1Score,
      d.gradientNorm,
      d.weightNorm,
    ].join(',')
  );
  const csv = [headers.join(','), ...rows].join('\n');
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
