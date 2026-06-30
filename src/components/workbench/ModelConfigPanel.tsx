import { useState, useEffect, useRef } from 'react';
import { DarkSelect } from '../ui/DarkSelect';

/* ============================================
   ModelConfigPanel — CNN模型架构配置弹窗
   - 注意力机制、卷积通道、归一化/残差等组件开关
   - 全连接层配置、MLP注意力开关
   ============================================ */

export interface ModelConfig {
  architecture: 'cnn' | 'mlp' | 'resnet18' | 'mobilenetv3' | 'vit';
  channels: number[];
  attention: 'none' | 'se' | 'cbam' | 'self_attention';
  use_bn: boolean;
  use_dropout: boolean;
  dropout_rate: number;
  use_residual: boolean;
  fc_hidden: number;
  use_attention: boolean;
}

export const DEFAULT_MODEL_CONFIG: ModelConfig = {
  architecture: 'cnn',
  channels: [32, 64],
  attention: 'none',
  use_bn: true,
  use_dropout: true,
  dropout_rate: 0.3,
  use_residual: false,
  fc_hidden: 128,
  use_attention: false,
};

const ATTENTION_OPTIONS = [
  { value: 'none', label: '无注意力' },
  { value: 'se', label: 'SE通道注意力' },
  { value: 'cbam', label: 'CBAM通道+空间' },
  { value: 'self_attention', label: '自注意力' },
];

const ARCHITECTURE_OPTIONS = [
  { value: 'cnn', label: '可配置CNN', desc: '自定义卷积通道、注意力', icon: 'CNN' },
  { value: 'resnet18', label: 'ResNet-18', desc: '残差网络，深层特征提取', icon: 'RN' },
  { value: 'mobilenetv3', label: 'MobileNetV3', desc: '轻量级移动端，深度可分离卷积', icon: 'MB' },
  { value: 'vit', label: 'Vision Transformer', desc: '注意力驱动，全局感受野', icon: 'VT' },
  { value: 'mlp', label: 'MLP', desc: '全连接网络，表格数据适用', icon: 'ML' },
] as const;

const CHANNEL_PRESETS = [
  { value: [16, 32], label: '[16,32]' },
  { value: [32, 64], label: '[32,64]' },
  { value: [32, 64, 128], label: '[32,64,128]' },
  { value: [64, 128, 256], label: '[64,128,256]' },
];

const FC_HIDDEN_PRESETS = [64, 128, 256, 512];

interface ModelConfigPanelProps {
  modelConfig: ModelConfig;
  datasetType?: string;
  onChange: (mc: ModelConfig) => void;
  disabled?: boolean;
}

export function ModelConfigPanel({ modelConfig, datasetType, onChange, disabled = false }: ModelConfigPanelProps) {
  const modalRef = useRef<HTMLDivElement>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [local, setLocal] = useState<ModelConfig>(modelConfig);
  const [customChannels, setCustomChannels] = useState('');

  const isTabularData = datasetType === 'csv' || datasetType === 'numpy';

  useEffect(() => {
    setLocal(modelConfig);
    setCustomChannels(modelConfig.channels.join(','));
  }, [modelConfig]);

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

  const update = <K extends keyof ModelConfig>(key: K, value: ModelConfig[K]) => {
    setLocal(prev => ({ ...prev, [key]: value }));
  };

  const handleChannelsPreset = (channels: number[]) => {
    update('channels', channels);
    setCustomChannels(channels.join(','));
  };

  const handleCustomChannelsChange = (value: string) => {
    setCustomChannels(value);
    const parsed = value
      .split(',')
      .map(s => parseInt(s.trim()))
      .filter(n => !isNaN(n) && n > 0);
    if (parsed.length > 0) {
      update('channels', parsed);
    }
  };

  const isChannelsPresetActive = (preset: number[]) => {
    if (local.channels.length !== preset.length) return false;
    return local.channels.every((c, i) => c === preset[i]);
  };

  const handleApply = () => {
    const validated = {
      ...local,
      dropout_rate: Math.max(0, Math.min(0.7, local.dropout_rate)),
    };
    onChange(validated);
    setIsOpen(false);
  };

  const handleReset = () => {
    setLocal(DEFAULT_MODEL_CONFIG);
    setCustomChannels(DEFAULT_MODEL_CONFIG.channels.join(','));
    onChange(DEFAULT_MODEL_CONFIG);
    setIsOpen(false);
  };

  const channelsSummary = `[${local.channels.join(',')}]`;
  const attentionLabel = ATTENTION_OPTIONS.find(o => o.value === local.attention)?.label.split('通道')[0] || 'none';
  const archLabel = ARCHITECTURE_OPTIONS.find(o => o.value === local.architecture)?.icon || 'CNN';
  const buttonSummary = [
    local.use_bn ? 'BN' : null,
    local.use_residual ? 'Res' : null,
  ].filter(Boolean).join('|') || 'plain';

  const isCNNArch = local.architecture === 'cnn';

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
        title={disabled ? '训练进行中无法修改模型配置' : '配置模型架构'}
      >
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="3" />
          <path d="M19.4 15a1.65 1.65 0 00.33 1.82l.06.06a2 2 0 01-2.83 2.83l-.06-.06a1.65 1.65 0 00-1.82-.33 1.65 1.65 0 00-1 1.51V21a2 2 0 01-4 0v-.09A1.65 1.65 0 009 19.4a1.65 1.65 0 00-1.82.33l-.06.06a2 2 0 01-2.83-2.83l.06-.06A1.65 1.65 0 004.68 15a1.65 1.65 0 00-1.51-1H3a2 2 0 010-4h.09A1.65 1.65 0 004.6 9a1.65 1.65 0 00-.33-1.82l-.06-.06a2 2 0 012.83-2.83l.06.06A1.65 1.65 0 009 4.68a1.65 1.65 0 001-1.51V3a2 2 0 014 0v.09a1.65 1.65 0 001 1.51 1.65 1.65 0 001.82-.33l.06-.06a2 2 0 012.83 2.83l-.06.06A1.65 1.65 0 0019.4 9a1.65 1.65 0 001.51 1H21a2 2 0 010 4h-.09a1.65 1.65 0 00-1.51 1z" />
        </svg>
        <span className="text-[10px] font-semibold">
          {archLabel} {channelsSummary} {attentionLabel} {buttonSummary}
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
                <h2 className="text-base font-bold">模型架构配置</h2>
                <p className="mt-0.5 text-xs text-muted-foreground">配置CNN网络结构与消融实验组件</p>
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

            <div className="space-y-4 p-6 max-h-[70vh] overflow-y-auto">
              {/* 模型架构选择 */}
              <div>
                <label className="mb-2 block text-xs font-medium text-foreground/80">模型架构</label>
                <div className="grid grid-cols-2 gap-2">
                  {ARCHITECTURE_OPTIONS.map(arch => (
                    <button
                      key={arch.value}
                      onClick={() => update('architecture', arch.value as ModelConfig['architecture'])}
                      className={`rounded-lg border p-2 text-left transition-all ${
                        local.architecture === arch.value
                          ? 'border-primary/40 bg-primary/10'
                          : 'border-white/[0.08] bg-white/[0.02] hover:bg-white/[0.04]'
                      }`}
                    >
                      <div className="flex items-center gap-1.5">
                        <span className={`rounded px-1 py-0.5 text-[9px] font-bold ${
                          local.architecture === arch.value
                            ? 'bg-primary/20 text-primary'
                            : 'bg-white/[0.06] text-muted-foreground'
                        }`}>{arch.icon}</span>
                        <span className="text-xs font-medium text-foreground/80">{arch.label}</span>
                      </div>
                      <div className="mt-0.5 text-[9px] text-muted-foreground/60">{arch.desc}</div>
                    </button>
                  ))}
                </div>
              </div>

              {/* 注意力机制 - 仅CNN架构显示 */}
              {isCNNArch && (
              <div>
                <label className="mb-1.5 block text-xs font-medium text-foreground/80">注意力机制 (Attention)</label>
                <DarkSelect
                  options={ATTENTION_OPTIONS}
                  value={local.attention}
                  onChange={(v) => update('attention', v as ModelConfig['attention'])}
                />
              </div>
              )}

              {/* 卷积通道数 - 仅CNN架构 */}
              {isCNNArch && (
              <div>
                <div className="mb-1.5 flex items-center justify-between">
                  <label className="text-xs font-medium text-foreground/80">卷积通道数 (Channels)</label>
                  <span className="font-mono text-xs text-primary">{channelsSummary}</span>
                </div>
                <div className="grid grid-cols-4 gap-2">
                  {CHANNEL_PRESETS.map(preset => (
                    <button
                      key={preset.label}
                      onClick={() => handleChannelsPreset(preset.value)}
                      className={`rounded-md border py-1.5 font-mono text-[10px] transition-all ${
                        isChannelsPresetActive(preset.value)
                          ? 'border-primary/40 bg-primary/15 text-primary'
                          : 'border-white/[0.08] bg-white/[0.02] text-muted-foreground hover:bg-white/[0.05]'
                      }`}
                    >
                      {preset.label}
                    </button>
                  ))}
                </div>
                <div className="mt-2">
                  <input
                    type="text"
                    value={customChannels}
                    onChange={(e) => handleCustomChannelsChange(e.target.value)}
                    placeholder="自定义：如 16,32,64"
                    className="w-full rounded-md border border-white/[0.08] bg-white/[0.02] px-3 py-1.5 font-mono text-xs text-foreground placeholder:text-muted-foreground/50 focus:border-primary/40 focus:outline-none"
                  />
                </div>
              </div>
              )}

              {/* 组件开关 - 消融实验（仅CNN和ResNet） */}
              {isCNNArch && (
              <div>
                <label className="mb-2 block text-xs font-medium text-foreground/80">组件开关（消融实验）</label>
                <div className="space-y-2">
                  {/* use_bn */}
                  <label className="flex items-center justify-between rounded-md border border-white/[0.08] bg-white/[0.02] px-3 py-2 cursor-pointer hover:bg-white/[0.04] transition-all">
                    <div>
                      <div className="text-xs font-medium text-foreground/80">批归一化 (BatchNorm)</div>
                      <div className="text-[10px] text-muted-foreground">加速训练收敛，稳定分布</div>
                    </div>
                    <button
                      type="button"
                      onClick={() => update('use_bn', !local.use_bn)}
                      className={`relative h-5 w-9 rounded-full transition-all ${
                        local.use_bn ? 'bg-primary' : 'bg-white/20'
                      }`}
                    >
                      <span
                        className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${
                          local.use_bn ? 'left-[18px]' : 'left-0.5'
                        }`}
                      />
                    </button>
                  </label>

                  {/* use_dropout */}
                  <label className="flex items-center justify-between rounded-md border border-white/[0.08] bg-white/[0.02] px-3 py-2 cursor-pointer hover:bg-white/[0.04] transition-all">
                    <div>
                      <div className="text-xs font-medium text-foreground/80">Dropout 正则化</div>
                      <div className="text-[10px] text-muted-foreground">防止过拟合</div>
                    </div>
                    <button
                      type="button"
                      onClick={() => update('use_dropout', !local.use_dropout)}
                      className={`relative h-5 w-9 rounded-full transition-all ${
                        local.use_dropout ? 'bg-primary' : 'bg-white/20'
                      }`}
                    >
                      <span
                        className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${
                          local.use_dropout ? 'left-[18px]' : 'left-0.5'
                        }`}
                      />
                    </button>
                  </label>

                  {/* use_residual */}
                  <label className="flex items-center justify-between rounded-md border border-white/[0.08] bg-white/[0.02] px-3 py-2 cursor-pointer hover:bg-white/[0.04] transition-all">
                    <div>
                      <div className="text-xs font-medium text-foreground/80">残差连接 (Residual)</div>
                      <div className="text-[10px] text-muted-foreground">缓解梯度消失，支持更深网络</div>
                    </div>
                    <button
                      type="button"
                      onClick={() => update('use_residual', !local.use_residual)}
                      className={`relative h-5 w-9 rounded-full transition-all ${
                        local.use_residual ? 'bg-primary' : 'bg-white/20'
                      }`}
                    >
                      <span
                        className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${
                          local.use_residual ? 'left-[18px]' : 'left-0.5'
                        }`}
                      />
                    </button>
                  </label>
                </div>
              </div>
              )}

              {/* Dropout Rate */}
              {local.use_dropout && (
                <div>
                  <div className="mb-1.5 flex items-center justify-between">
                    <label className="text-xs font-medium text-foreground/80">Dropout 率</label>
                    <span className="font-mono text-xs text-primary">{local.dropout_rate.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={0.7}
                    step={0.05}
                    value={local.dropout_rate}
                    onChange={(e) => update('dropout_rate', parseFloat(e.target.value))}
                    className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-primary"
                  />
                  <div className="mt-1 flex justify-between text-[10px] text-muted-foreground">
                    <span>0.0</span><span>0.2</span><span>0.4</span><span>0.6</span>
                  </div>
                </div>
              )}

              {/* 全连接隐藏层维度 */}
              <div>
                <div className="mb-1.5 flex items-center justify-between">
                  <label className="text-xs font-medium text-foreground/80">全连接隐藏层维度</label>
                  <span className="font-mono text-xs text-primary">{local.fc_hidden}</span>
                </div>
                <div className="grid grid-cols-4 gap-2">
                  {FC_HIDDEN_PRESETS.map(dim => (
                    <button
                      key={dim}
                      onClick={() => update('fc_hidden', dim)}
                      className={`rounded-md border py-1.5 font-mono text-xs transition-all ${
                        local.fc_hidden === dim
                          ? 'border-primary/40 bg-primary/15 text-primary'
                          : 'border-white/[0.08] bg-white/[0.02] text-muted-foreground hover:bg-white/[0.05]'
                      }`}
                    >
                      {dim}
                    </button>
                  ))}
                </div>
              </div>

              {/* MLP注意力开关（表格数据时显示） */}
              {isTabularData && (
                <div>
                  <label className="mb-2 block text-xs font-medium text-foreground/80">MLP 配置</label>
                  <label className="flex items-center justify-between rounded-md border border-white/[0.08] bg-white/[0.02] px-3 py-2 cursor-pointer hover:bg-white/[0.04] transition-all">
                    <div>
                      <div className="text-xs font-medium text-foreground/80">MLP 自注意力</div>
                      <div className="text-[10px] text-muted-foreground">表格数据特征注意力加权</div>
                    </div>
                    <button
                      type="button"
                      onClick={() => update('use_attention', !local.use_attention)}
                      className={`relative h-5 w-9 rounded-full transition-all ${
                        local.use_attention ? 'bg-primary' : 'bg-white/20'
                      }`}
                    >
                      <span
                        className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${
                          local.use_attention ? 'left-[18px]' : 'left-0.5'
                        }`}
                      />
                    </button>
                  </label>
                </div>
              )}
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
