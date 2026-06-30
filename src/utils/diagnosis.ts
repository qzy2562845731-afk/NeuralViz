export type DiagnosisType = 
  | 'overfit' 
  | 'generalization_gap' 
  | 'gradient_exploding' 
  | 'gradient_vanishing' 
  | 'converging' 
  | 'sparse_activation' 
  | 'feature_learning' 
  | 'loss_increasing' 
  | 'healthy';

export interface DiagnosisResult {
  type: DiagnosisType;
  label: string;
  description: string;
  severity: 'good' | 'info' | 'warning' | 'danger';
  value?: string;
}

export interface TrainingDiagnosis {
  results: DiagnosisResult[];
  overall: 'good' | 'healthy' | 'warning' | 'concerning';
}

export function diagnoseOverfitting(
  trainLoss: number, 
  valLoss: number, 
  step: number
): DiagnosisResult | null {
  const gap = valLoss - trainLoss;
  
  if (gap > 0.5 && step > 50) {
    return {
      type: 'overfit',
      label: '严重过拟合',
      description: `验证损失显著高于训练损失，模型过拟合严重`,
      severity: 'danger',
      value: `gap: ${gap.toFixed(4)}`
    };
  }
  
  if (gap > 0.2 && step > 30) {
    return {
      type: 'overfit',
      label: '可能过拟合',
      description: `训练-验证损失差距较大，建议增加正则化或数据增强`,
      severity: 'warning',
      value: `gap: ${gap.toFixed(4)}`
    };
  }
  
  return null;
}

export function diagnoseGeneralizationGap(
  trainAcc: number, 
  valAcc: number
): DiagnosisResult | null {
  const gap = trainAcc - valAcc;
  
  if (gap > 0.2) {
    return {
      type: 'generalization_gap',
      label: '泛化差距严重',
      description: `训练准确率与验证准确率差距过大`,
      severity: 'danger',
      value: `${(gap * 100).toFixed(1)}%`
    };
  }
  
  if (gap > 0.1) {
    return {
      type: 'generalization_gap',
      label: '泛化差距较大',
      description: `模型泛化能力有待提升`,
      severity: 'warning',
      value: `${(gap * 100).toFixed(1)}%`
    };
  }
  
  if (gap < 0.02) {
    return {
      type: 'generalization_gap',
      label: '泛化能力良好',
      description: `训练与验证表现一致性好`,
      severity: 'good'
    };
  }
  
  return null;
}

export function diagnoseGradientExploding(
  gradientNorm: number
): DiagnosisResult | null {
  if (gradientNorm > 10) {
    return {
      type: 'gradient_exploding',
      label: '梯度爆炸',
      description: `梯度范数极大，训练严重不稳定，建议降低学习率或使用梯度裁剪`,
      severity: 'danger',
      value: gradientNorm.toFixed(3)
    };
  }
  
  if (gradientNorm > 5) {
    return {
      type: 'gradient_exploding',
      label: '梯度过大',
      description: `梯度范数偏大，训练可能出现振荡`,
      severity: 'warning',
      value: gradientNorm.toFixed(3)
    };
  }
  
  return null;
}

export function diagnoseGradientVanishing(
  gradientNorm: number,
  step: number
): DiagnosisResult | null {
  if (gradientNorm < 0.001 && step > 20) {
    return {
      type: 'gradient_vanishing',
      label: '梯度消失',
      description: `梯度范数极小，训练几乎停滞，建议使用残差连接或预训练`,
      severity: 'danger',
      value: gradientNorm.toFixed(5)
    };
  }
  
  if (gradientNorm < 0.01 && step > 30) {
    return {
      type: 'gradient_vanishing',
      label: '梯度衰减',
      description: `梯度范数较小，训练速度放缓`,
      severity: 'warning',
      value: gradientNorm.toFixed(4)
    };
  }
  
  return null;
}

export function diagnoseConvergence(
  trainLoss: number,
  valLoss: number,
  step: number,
  windowData?: { trainLoss: number; valLoss: number; f1Score: number }[]
): DiagnosisResult | null {
  if (step < 15 || !windowData || windowData.length < 5) return null;
  
  const recentLossDelta = trainLoss - windowData[0].trainLoss;
  const recentValLossDelta = valLoss - windowData[0].valLoss;
  
  if (recentLossDelta < -0.05 && recentValLossDelta < -0.05) {
    return {
      type: 'converging',
      label: '快速收敛中',
      description: `损失正在快速下降，模型训练效果良好`,
      severity: 'good'
    };
  }
  
  if (Math.abs(recentLossDelta) < 0.01 && Math.abs(recentValLossDelta) < 0.01 && step > 50) {
    return {
      type: 'converging',
      label: '趋于收敛',
      description: `损失下降趋于平缓，模型可能已收敛`,
      severity: 'info'
    };
  }
  
  return null;
}

export function diagnoseSparseActivation(
  layerName: string,
  sparsity: number
): DiagnosisResult | null {
  if (sparsity > 0.9) {
    return {
      type: 'sparse_activation',
      label: `${layerName} 激活极度稀疏`,
      description: `激活稀疏度过高，可能存在大量死神经元`,
      severity: 'warning',
      value: `${(sparsity * 100).toFixed(1)}%`
    };
  }
  
  if (sparsity > 0.7) {
    return {
      type: 'sparse_activation',
      label: `${layerName} 激活稀疏`,
      description: `部分神经元激活较弱`,
      severity: 'info',
      value: `${(sparsity * 100).toFixed(1)}%`
    };
  }
  
  return null;
}

export function diagnoseFeatureLearning(
  conv1Sparsity: number,
  conv1Max: number
): DiagnosisResult | null {
  if (conv1Sparsity > 0.6 && conv1Max < 0.5) {
    return {
      type: 'feature_learning',
      label: '特征学习不足',
      description: `卷积层响应较弱，模型可能需要更多训练或调整架构`,
      severity: 'warning'
    };
  }
  
  if (conv1Max > 0.7 && conv1Sparsity < 0.5) {
    return {
      type: 'feature_learning',
      label: '特征学习活跃',
      description: `卷积层响应良好，特征提取有效`,
      severity: 'good'
    };
  }
  
  return null;
}

export function diagnoseLossIncreasing(
  recentLossTrend: number,
  step: number
): DiagnosisResult | null {
  if (recentLossTrend > 0.1 && step > 30) {
    return {
      type: 'loss_increasing',
      label: '损失上升',
      description: `训练损失持续上升，可能学习率过大或数据问题`,
      severity: 'danger',
      value: `+${recentLossTrend.toFixed(4)}`
    };
  }
  
  return null;
}

export function diagnoseTraining(
  currentData: {
    step: number;
    trainLoss: number;
    valLoss: number;
    trainAccuracy: number;
    valAccuracy: number;
    gradientNorm: number;
    f1Score: number;
    activationStats: {
      input: { sparsity: number; max: number };
      conv1: { sparsity: number; max: number };
      conv2?: { sparsity: number; max: number };
      fc: { sparsity: number; max: number };
    };
  },
  windowData?: {
    trainLoss: number;
    valLoss: number;
    f1Score: number;
  }[]
): TrainingDiagnosis {
  const results: DiagnosisResult[] = [];
  
  const overfitResult = diagnoseOverfitting(
    currentData.trainLoss,
    currentData.valLoss,
    currentData.step
  );
  if (overfitResult) results.push(overfitResult);
  
  const gapResult = diagnoseGeneralizationGap(
    currentData.trainAccuracy,
    currentData.valAccuracy
  );
  if (gapResult) results.push(gapResult);
  
  const explodingResult = diagnoseGradientExploding(currentData.gradientNorm);
  if (explodingResult) results.push(explodingResult);
  
  const vanishingResult = diagnoseGradientVanishing(
    currentData.gradientNorm,
    currentData.step
  );
  if (vanishingResult) results.push(vanishingResult);
  
  const convergenceResult = diagnoseConvergence(
    currentData.trainLoss,
    currentData.valLoss,
    currentData.step,
    windowData
  );
  if (convergenceResult) results.push(convergenceResult);
  
  const inputSparse = diagnoseSparseActivation('Input', currentData.activationStats.input.sparsity);
  if (inputSparse) results.push(inputSparse);
  
  const convSparse = diagnoseSparseActivation('Conv1', currentData.activationStats.conv1.sparsity);
  if (convSparse) results.push(convSparse);
  
  if (currentData.activationStats.conv2) {
    const conv2Sparse = diagnoseSparseActivation('Conv2', currentData.activationStats.conv2.sparsity);
    if (conv2Sparse) results.push(conv2Sparse);
  }
  
  const fcSparse = diagnoseSparseActivation('FC', currentData.activationStats.fc.sparsity);
  if (fcSparse) results.push(fcSparse);
  
  const featureResult = diagnoseFeatureLearning(
    currentData.activationStats.conv1.sparsity,
    currentData.activationStats.conv1.max
  );
  if (featureResult) results.push(featureResult);
  
  if (windowData && windowData.length >= 5) {
    const recentTrend = currentData.trainLoss - windowData[0].trainLoss;
    const lossIncreasingResult = diagnoseLossIncreasing(recentTrend, currentData.step);
    if (lossIncreasingResult) results.push(lossIncreasingResult);
  }
  
  const dangerCount = results.filter(r => r.severity === 'danger').length;
  const warningCount = results.filter(r => r.severity === 'warning').length;
  const goodCount = results.filter(r => r.severity === 'good').length;
  
  let overall: TrainingDiagnosis['overall'] = 'healthy';
  if (dangerCount > 0) {
    overall = 'concerning';
  } else if (warningCount > 2) {
    overall = 'warning';
  } else if (goodCount > 2 && warningCount === 0) {
    overall = 'good';
  }
  
  return { results, overall };
}
