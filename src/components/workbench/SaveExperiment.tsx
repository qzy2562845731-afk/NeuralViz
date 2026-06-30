import { useState } from 'react';
import { apiService } from '../../services/api';
import type { TrainingStepData } from '../../types/training';

interface SaveExperimentProps {
  modelId: string | null;
  modelName: string;
  architecture: any;
  totalParams: number;
  layerCount: number;
  currentStep: number;
  loss: number;
  accuracy: number;
  trainingStatus?: string;
  totalEpochs?: number;
  hyperparams?: Record<string, any>;
  trainingHistory?: TrainingStepData[];
  trainingLogs?: string[];
  onSaved?: (experimentId: string) => void;
}

export function SaveExperiment({
  modelId,
  modelName,
  architecture,
  totalParams,
  layerCount,
  currentStep,
  loss,
  accuracy,
  trainingStatus = 'draft',
  totalEpochs = 0,
  hyperparams,
  trainingHistory,
  trainingLogs,
  onSaved,
}: SaveExperimentProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [tags, setTags] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleOpen = () => {
    setName(`${modelName || '实验'}_${new Date().toLocaleDateString('zh-CN')}`);
    setDescription('');
    setTags('');
    setError(null);
    setIsOpen(true);
  };

  const handleSave = async () => {
    if (!name.trim()) {
      setError('请输入实验名称');
      return;
    }

    setSaving(true);
    setError(null);

    try {
      const tagList = tags
        .split(/[,，\s]+/)
        .map((t) => t.trim())
        .filter(Boolean);

      const res = await apiService.createExperiment({
        name: name.trim(),
        description: description.trim(),
        model_id: modelId || undefined,
        model_name: modelName || undefined,
        model_architecture: architecture,
        hyperparams: hyperparams || {
          learning_rate: 0.001,
          batch_size: 32,
          optimizer: 'adam',
        },
        config: {
          training_logs: trainingLogs || [],
        },
        tags: tagList,
        total_params: totalParams,
        layer_count: layerCount,
        status: trainingStatus,
        best_accuracy: accuracy,
        final_loss: loss,
        current_step: currentStep,
        total_epochs: totalEpochs || currentStep,
      });

      // 批量保存训练指标到 experiment_metrics 表
      const experimentId = res.data.experiment_id;
      if (trainingHistory && trainingHistory.length > 0) {
        try {
          const metrics = trainingHistory.map((h) => ({
            step: h.step,
            epoch: h.step,
            loss: h.trainLoss ?? h.loss ?? 0,
            accuracy: h.trainAccuracy ?? h.accuracy ?? 0,
            val_loss: h.valLoss ?? 0,
            val_accuracy: h.valAccuracy ?? 0,
            learning_rate: h.learningRate ?? 0,
            metric_type: 'training',
            extra_data: {
              precision: h.precision ?? 0,
              recall: h.recall ?? 0,
              f1: h.f1Score ?? 0,
              gradient_norm: h.gradientNorm ?? 0,
              weight_norm: h.weightNorm ?? 0,
            },
          }));
          await apiService.addExperimentMetrics(experimentId, metrics);
        } catch (metricsErr) {
          console.warn('保存训练指标失败，但实验已创建:', metricsErr);
        }
      }

      setIsOpen(false);
      if (onSaved) {
        onSaved(experimentId);
      }
    } catch (err: any) {
      setError(err.message || '保存失败');
    } finally {
      setSaving(false);
    }
  };

  return (
    <>
      <button
        onClick={handleOpen}
        className="flex items-center gap-1.5 rounded-lg border border-emerald-400/30 bg-emerald-400/10 px-3 py-1.5 text-xs font-semibold text-emerald-400 transition-all hover:bg-emerald-400/15"
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
          <path d="M17 21v-8H7v8M7 3v5h8" />
        </svg>
        保存实验
      </button>

      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div className="w-full max-w-md overflow-hidden rounded-2xl border border-white/[0.08] bg-[#0f1119]">
            <div className="border-b border-white/[0.06] px-6 py-4">
              <h2 className="text-base font-bold">保存为实验</h2>
              <p className="mt-0.5 text-xs text-muted-foreground">
                保存当前模型架构、参数和训练状态
              </p>
            </div>

            <div className="space-y-4 p-6">
              {error && (
                <div className="rounded-lg border border-red-400/20 bg-red-400/10 px-3 py-2 text-xs text-red-400">
                  {error}
                </div>
              )}

              <div>
                <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                  实验名称 <span className="text-red-400">*</span>
                </label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="输入实验名称"
                  className="w-full rounded-lg border border-white/[0.08] bg-white/[0.02] px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary/40 focus:outline-none"
                />
              </div>

              <div>
                <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                  实验描述
                </label>
                <textarea
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="描述本次实验的目的、配置等..."
                  rows={3}
                  className="w-full resize-none rounded-lg border border-white/[0.08] bg-white/[0.02] px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary/40 focus:outline-none"
                />
              </div>

              <div>
                <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
                  标签（逗号分隔）
                </label>
                <input
                  type="text"
                  value={tags}
                  onChange={(e) => setTags(e.target.value)}
                  placeholder="例如：CNN, MNIST, 测试"
                  className="w-full rounded-lg border border-white/[0.08] bg-white/[0.02] px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary/40 focus:outline-none"
                />
              </div>

              <div className="grid grid-cols-3 gap-3 rounded-lg border border-white/[0.06] bg-white/[0.01] p-3">
                <div>
                  <div className="text-[10px] uppercase text-muted-foreground">模型</div>
                  <div className="mt-0.5 font-mono text-xs">{modelName || '-'}</div>
                </div>
                <div>
                  <div className="text-[10px] uppercase text-muted-foreground">参数量</div>
                  <div className="mt-0.5 font-mono text-xs">{totalParams.toLocaleString()}</div>
                </div>
                <div>
                  <div className="text-[10px] uppercase text-muted-foreground">Step</div>
                  <div className="mt-0.5 font-mono text-xs">{currentStep}</div>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-end gap-2 border-t border-white/[0.06] px-6 py-3">
              <button
                onClick={() => setIsOpen(false)}
                disabled={saving}
                className="rounded-md border border-white/[0.08] bg-white/[0.02] px-4 py-1.5 text-xs font-medium text-muted-foreground transition-all hover:text-foreground disabled:opacity-50"
              >
                取消
              </button>
              <button
                onClick={handleSave}
                disabled={saving}
                className="flex items-center gap-1.5 rounded-md border border-emerald-400/30 bg-emerald-400/10 px-4 py-1.5 text-xs font-semibold text-emerald-400 transition-all hover:bg-emerald-400/15 disabled:opacity-50"
              >
                {saving ? (
                  <>
                    <svg className="animate-spin" width="12" height="12" viewBox="0 0 24 24" fill="none">
                      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" strokeOpacity="0.25" />
                      <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
                    </svg>
                    保存中...
                  </>
                ) : (
                  <>保存实验</>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
