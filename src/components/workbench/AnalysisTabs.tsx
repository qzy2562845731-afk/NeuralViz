import { useState } from 'react';
import { TrainingCurvesPanel } from '../TrainingCurvesPanel';
import { FeatureMapViewer } from '../FeatureMapViewer';
import { ConfusionMatrix } from '../ConfusionMatrix';
import { ActivationHistogram } from '../ActivationHistogram';
import { GradientWeightChart } from '../GradientWeightChart';
import type { TrainingStepData } from '../../types/training';
import type { ChartColors } from '../../hooks/useColorConfig';

type TabType = 'training' | 'features' | 'confusion' | 'activation' | 'gradient';

interface AnalysisTabsProps {
  visibleData: TrainingStepData[];
  currentData: TrainingStepData | null;
  currentStep: number;
  chartColors: ChartColors;
}

export function AnalysisTabs({ visibleData, currentData, currentStep, chartColors }: AnalysisTabsProps) {
  const [activeTab, setActiveTab] = useState<TabType>('training');

  const tabs = [
    { id: 'training' as TabType, label: '训练曲线' },
    { id: 'features' as TabType, label: '特征图' },
    { id: 'confusion' as TabType, label: '混淆矩阵' },
    { id: 'activation' as TabType, label: '激活分布' },
    { id: 'gradient' as TabType, label: '梯度分析' },
  ];

  const renderContent = () => {
    switch (activeTab) {
      case 'training':
        return (
          <TrainingCurvesPanel
            visibleData={visibleData}
            currentStep={currentStep}
            colors={chartColors}
          />
        );
      case 'features':
        return (
          <FeatureMapViewer
            featureMaps={currentData?.featureMaps}
            step={currentStep}
          />
        );
      case 'confusion':
        return (
          <ConfusionMatrix data={currentData} colors={chartColors} />
        );
      case 'activation':
        return (
          <ActivationHistogram data={currentData} colors={chartColors} />
        );
      case 'gradient':
        return (
          <GradientWeightChart
            visibleData={visibleData}
            currentStep={currentStep}
            colors={chartColors}
          />
        );
    }
  };

  return (
    <div className="flex h-72 flex-col border-t border-white/10 bg-[#0a0c14]">
      {/* 标签栏 - 固定高度 */}
      <div className="flex flex-shrink-0 border-b border-white/10">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`px-6 py-3 text-sm font-medium transition ${
              activeTab === tab.id
                ? 'border-b-2 border-green-400 text-green-400'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* 内容区 - 可滚动 */}
      <div className="min-h-0 flex-1 overflow-y-auto">
        {renderContent()}
      </div>
    </div>
  );
}
