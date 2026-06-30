import { useState, useEffect, useRef } from 'react';
import { DarkSelect } from '../ui/DarkSelect';

/* ============================================
   HyperparamPanel — 训练超参数配置弹窗
   - 学习率、batch size、epochs、优化器等
   - 提供预设值和手动输入
   ============================================ */

export interface Hyperparams {
  learning_rate: number;
  batch_size: number;
  optimizer: string;
  loss_function: string;
  epochs: number;
  random_seed: number;
  val_split: number;
  // 新增训练策略
  scheduler_type: string;
  early_stopping: boolean;
  early_stopping_patience: number;
  use_amp: boolean;
  resume_from_checkpoint: boolean;
}

const DEFAULT_HYPERPARAMS: Hyperparams = {
  learning_rate: 0.001,
  batch_size: 32,
  optimizer: 'adam',
  loss_function: 'cross_entropy',
  epochs: 20,
  random_seed: 42,
  val_split: 0.2,
  scheduler_type: 'none',
  early_stopping: false,
  early_stopping_patience: 5,
  use_amp: false,
  resume_from_checkpoint: false,
};

const OPTIMIZERS = [
  { value: 'adam', label: 'Adam' },
  { value: 'sgd', label: 'SGD' },
  { value: 'adamw', label: 'AdamW' },
  { value: 'rmsprop', label: 'RMSprop' },
];

const LR_PRESETS = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5];
const BATCH_PRESETS = [8, 16, 32, 64, 128];

// 损失函数选项：value 为内部标识，label 为 PyTorch 类名
const LOSS_FUNCTIONS = [
  { value: 'cross_entropy', label: 'CrossEntropyLoss' },
  { value: 'nll', label: 'NLLLoss' },
  { value: 'bce_with_logits', label: 'BCEWithLogitsLoss' },
  { value: 'bce', label: 'BCELoss' },
  { value: 'smooth_l1', label: 'SmoothL1Loss' },
];

const SCHEDULERS = [
  { value: 'none', label: '无调度器' },
  { value: 'cosine', label: 'CosineAnnealing' },
  { value: 'plateau', label: 'ReduceLROnPlateau' },
  { value: 'step', label: 'StepLR' },
  { value: 'exponential', label: 'ExponentialLR' },
];

interface HyperparamPanelProps {
  hyperparams: Hyperparams;
  onChange: (hp: Hyperparams) => void;
  disabled?: boolean;
}

export function HyperparamPanel({ hyperparams, onChange, disabled = false }: HyperparamPanelProps) {
  const modalRef = useRef<HTMLDivElement>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [local, setLocal] = useState<Hyperparams>(hyperparams);

  useEffect(() => {
    setLocal(hyperparams);
  }, [hyperparams]);

  useEffect(() => {
    if (!isOpen) return;
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsOpen(false);
    };
    document.addEventListener('keydown', handleEsc);
    return () => document.removeEventListener('keydown', handleEsc);
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (modalRef.current && !modalRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  const update = <K extends keyof Hyperparams>(key: K, value: Hyperparams[K]) => {
    setLocal(prev => ({ ...prev, [key]: value }));
  };

  const handleApply = () => {
    const validated = {
      ...local,
      learning_rate: Math.max(0.00001, Math.min(1, local.learning_rate)),
      batch_size: Math.max(1, Math.min(512, Math.round(local.batch_size))),
      epochs: Math.max(1, Math.min(500, Math.round(local.epochs))),
      random_seed: Math.round(local.random_seed),
      val_split: Math.max(0, Math.min(0.9, local.val_split)),
      early_stopping_patience: Math.max(1, Math.min(20, Math.round(local.early_stopping_patience))),
    };
    onChange(validated);
    setIsOpen(false);
  };

  const handleReset = () => {
    setLocal(DEFAULT_HYPERPARAMS);
    onChange(DEFAULT_HYPERPARAMS);
    setIsOpen(false);
  };

  // 学习率是否超出有效范围（用于输入框红色边框提示）
  const isLrOutOfRange = local.learning_rate < 0.00001 || local.learning_rate > 1;

  return (
    <>
      <button
        onClick={() => !disabled && setIsOpen(true)}
        disabled={disabled}
        className={`flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 transition-all ${
          disabled
            ? 'cursor-not-allowed border-white/[0.04] bg-white/[0.01] text-muted-foreground/40'
            : 'border-white/[0.08] bg-white/[0.02] text-muted-foreground/80 hover:bg-white/[0.05] hover:text-foreground'
        }`}
        title={disabled ? '训练进行中无法修改超参数' : '配置训练超参数'}
      >
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="3" />
          <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 01-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z" />
        </svg>
        <span className="text-[10px] font-semibold">
          lr={hyperparams.learning_rate} ep={hyperparams.epochs}
        </span>
      </button>

      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div
            ref={modalRef}
            className="w-full max-w-md overflow-hidden rounded-2xl border border-white/[0.08] bg-[#0f1119] shadow-2xl"
          >
            <div className="flex items-center justify-between border-b border-white/[0.06] px-6 py-4">
              <div>
                <h2 className="text-base font-bold">训练超参数配置</h2>
                <p className="mt-0.5 text-xs text-muted-foreground">调整训练参数以获得更好的模型效果</p>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="flex h-8 w-8 items-center justify-center rounded-md border border-white/[0.08] bg-white/[0.02] text-muted-foreground transition-all hover:text-foreground"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6 6 18M6 6l12 12" />
                </svg>
              </button>
            </div>

            <div className="space-y-4 p-6">
              {/* 学习率 */}
              <div>
                <div className="mb-1.5 flex items-center justify-between">
                  <label className="text-xs font-medium text-foreground/80">学习率 (Learning Rate)</label>
                  <span className="font-mono text-xs text-primary">{local.learning_rate}</span>
                </div>
                <div className="flex items-center gap-2">
                  <input
                    type="range"
                    min={0.0001}
                    max={1.0}
                    step={0.00001}
                    value={local.learning_rate}
                    onChange={(e) => update('learning_rate', parseFloat(e.target.value))}
                    className="h-1.5 flex-1 cursor-pointer appearance-none rounded-full bg-white/10 accent-primary"
                  />
                  <input
                    type="number"
                    min={0.00001}
                    max={1.0}
                    step={0.00001}
                    value={local.learning_rate}
                    onChange={(e) => {
                      const v = parseFloat(e.target.value);
                      update('learning_rate', isNaN(v) ? 0 : v);
                    }}
                    className={`w-20 rounded-md border bg-white/[0.02] px-2 py-1 font-mono text-xs text-foreground focus:outline-none ${
                      isLrOutOfRange
                        ? 'border-red-500/60 focus:border-red-500'
                        : 'border-white/[0.08] focus:border-primary/40'
                    }`}
                    title="输入范围：0.00001 ~ 1.0"
                  />
                </div>
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {LR_PRESETS.map(lr => (
                    <button
                      key={lr}
                      onClick={() => update('learning_rate', lr)}
                      className={`rounded px-1.5 py-0.5 font-mono text-[10px] transition-all ${
                        Math.abs(local.learning_rate - lr) < 0.00001
                          ? 'bg-primary/20 text-primary'
                          : 'bg-white/[0.04] text-muted-foreground hover:bg-white/[0.08]'
                      }`}
                    >
                      {lr}
                    </button>
                  ))}
                </div>
              </div>

              {/* Batch Size */}
              <div>
                <div className="mb-1.5 flex items-center justify-between">
                  <label className="text-xs font-medium text-foreground/80">批大小 (Batch Size)</label>
                  <span className="font-mono text-xs text-primary">{local.batch_size}</span>
                </div>
                <div className="flex gap-2">
                  {BATCH_PRESETS.map(bs => (
                    <button
                      key={bs}
                      onClick={() => update('batch_size', bs)}
                      className={`flex-1 rounded-md border py-1.5 font-mono text-xs transition-all ${
                        local.batch_size === bs
                          ? 'border-primary/40 bg-primary/15 text-primary'
                          : 'border-white/[0.08] bg-white/[0.02] text-muted-foreground hover:bg-white/[0.05]'
                      }`}
                    >
                      {bs}
                    </button>
                  ))}
                </div>
              </div>

              {/* 优化器 */}
              <div>
                <label className="mb-1.5 block text-xs font-medium text-foreground/80">优化器 (Optimizer)</label>
                <div className="grid grid-cols-4 gap-2">
                  {OPTIMIZERS.map(opt => (
                    <button
                      key={opt.value}
                      onClick={() => update('optimizer', opt.value)}
                      className={`rounded-md border py-1.5 text-xs font-medium transition-all ${
                        local.optimizer === opt.value
                          ? 'border-primary/40 bg-primary/15 text-primary'
                          : 'border-white/[0.08] bg-white/[0.02] text-muted-foreground hover:bg-white/[0.05]'
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* 损失函数 */}
              <div>
                <label className="mb-1.5 block text-xs font-medium text-foreground/80">损失函数 (Loss Function)</label>
                <DarkSelect
                  options={LOSS_FUNCTIONS}
                  value={local.loss_function}
                  onChange={(v) => update('loss_function', v)}
                />
              </div>

              {/* Epochs */}
              <div>
                <div className="mb-1.5 flex items-center justify-between">
                  <label className="text-xs font-medium text-foreground/80">训练轮数 (Epochs)</label>
                  <span className="font-mono text-xs text-primary">{local.epochs}</span>
                </div>
                <input
                  type="range"
                  min={1}
                  max={100}
                  step={1}
                  value={local.epochs}
                  onChange={(e) => update('epochs', parseInt(e.target.value))}
                  className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-primary"
                />
                <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
                  <span>1</span><span>25</span><span>50</span><span>75</span><span>100</span>
                </div>
              </div>

              {/* 验证集比例 + 随机种子 */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <div className="mb-1.5 flex items-center justify-between">
                    <label className="text-xs font-medium text-foreground/80">验证集比例</label>
                    <span className="font-mono text-xs text-primary">{(local.val_split * 100).toFixed(0)}%</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={0.5}
                    step={0.05}
                    value={local.val_split}
                    onChange={(e) => update('val_split', parseFloat(e.target.value))}
                    className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-primary"
                  />
                </div>
                <div>
                  <label className="mb-1.5 block text-xs font-medium text-foreground/80">随机种子</label>
                  <input
                    type="number"
                    value={local.random_seed}
                    onChange={(e) => update('random_seed', parseInt(e.target.value) || 0)}
                    className="w-full rounded-md border border-white/[0.08] bg-white/[0.02] px-3 py-1.5 font-mono text-xs text-foreground focus:border-primary/40 focus:outline-none"
                  />
                </div>
              </div>

              {/* 训练策略配置 */}
              <div className="rounded-lg border border-white/[0.06] bg-white/[0.01] p-3">
                <h3 className="mb-3 text-xs font-semibold text-foreground/80">训练策略</h3>

                {/* 学习率调度器 */}
                <div className="mb-3">
                  <label className="mb-1.5 block text-xs font-medium text-foreground/70">学习率调度器</label>
                  <DarkSelect
                    options={SCHEDULERS}
                    value={local.scheduler_type}
                    onChange={(v) => update('scheduler_type', v)}
                  />
                </div>

                {/* 早停 + AMP */}
                <div className="flex items-center gap-4">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={local.early_stopping}
                      onChange={(e) => update('early_stopping', e.target.checked)}
                      className="h-3.5 w-3.5 rounded border-white/[0.15] bg-white/[0.04] accent-primary"
                    />
                    <span className="text-xs text-foreground/70">早停机制</span>
                  </label>
                  {local.early_stopping && (
                    <div className="flex items-center gap-1.5">
                      <span className="text-[10px] text-muted-foreground">patience:</span>
                      <input
                        type="number"
                        min={1}
                        max={20}
                        value={local.early_stopping_patience}
                        onChange={(e) => update('early_stopping_patience', Math.max(1, parseInt(e.target.value) || 5))}
                        className="w-12 rounded border border-white/[0.08] bg-white/[0.02] px-1.5 py-0.5 text-center font-mono text-[10px] text-foreground focus:border-primary/40 focus:outline-none"
                      />
                    </div>
                  )}
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={local.use_amp}
                      onChange={(e) => update('use_amp', e.target.checked)}
                      className="h-3.5 w-3.5 rounded border-white/[0.15] bg-white/[0.04] accent-green-500"
                    />
                    <span className="text-xs text-foreground/70">混合精度(AMP)</span>
                  </label>
                </div>

                {/* 断点续训 */}
                <div className="mt-2">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={local.resume_from_checkpoint}
                      onChange={(e) => update('resume_from_checkpoint', e.target.checked)}
                      className="h-3.5 w-3.5 rounded border-white/[0.15] bg-white/[0.04] accent-amber-500"
                    />
                    <span className="text-xs text-foreground/70">从断点续训</span>
                    <span className="text-[10px] text-muted-foreground">(如有checkpoint则自动恢复)</span>
                  </label>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between border-t border-white/[0.06] px-6 py-3">
              <button
                onClick={handleReset}
                className="text-xs text-muted-foreground transition-all hover:text-red-400"
              >
                恢复默认值
              </button>
              <div className="flex gap-2">
                <button
                  onClick={() => setIsOpen(false)}
                  className="rounded-md border border-white/[0.08] bg-white/[0.02] px-4 py-1.5 text-xs text-muted-foreground transition-all hover:bg-white/[0.05] hover:text-foreground"
                >
                  取消
                </button>
                <button
                  onClick={handleApply}
                  className="rounded-md bg-primary px-4 py-1.5 text-xs font-medium text-white transition-all hover:bg-primary/90"
                >
                  应用配置
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export { DEFAULT_HYPERPARAMS };
