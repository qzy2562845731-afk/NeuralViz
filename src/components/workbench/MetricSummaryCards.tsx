import type { TrainingStepData } from '../../types/training';

interface MetricSummaryCardsProps {
  currentData: TrainingStepData | null;
}

export function MetricSummaryCards({ currentData }: MetricSummaryCardsProps) {
  const metrics = [
    { label: 'Loss', value: currentData ? currentData.loss.toFixed(4) : '0.0000', color: 'text-red-400' },
    { label: 'Accuracy', value: currentData ? `${(currentData.accuracy * 100).toFixed(1)}%` : '0.0%', color: 'text-green-400' },
    { label: 'F1 Score', value: currentData ? currentData.f1Score.toFixed(3) : '0.000', color: 'text-blue-400' },
    { label: 'Learning Rate', value: currentData ? (currentData.learningRate ? currentData.learningRate.toFixed(4) : '0.0000') : '0.0000', color: 'text-yellow-400' },
    { label: 'Epoch', value: currentData ? `${currentData.step}/100` : '0/100', color: 'text-purple-400' },
    { label: 'Batch Size', value: '32', color: 'text-gray-400' },
  ];

  return (
    <div className="grid grid-cols-6 gap-4 border-b border-white/10 bg-[#0f1119] p-4">
      {metrics.map((metric) => (
        <div
          key={metric.label}
          className="rounded-lg border border-white/10 bg-white/5 p-3"
        >
          <div className="mb-1 text-xs text-gray-500">{metric.label}</div>
          <div className={`text-lg font-semibold ${metric.color}`}>
            {metric.value}
          </div>
        </div>
      ))}
    </div>
  );
}
