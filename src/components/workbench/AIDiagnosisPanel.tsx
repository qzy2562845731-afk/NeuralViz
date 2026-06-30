import { useState, useEffect, useCallback, useRef } from 'react';
import { useWorkbench } from './WorkbenchContext';
import { LAYER_COLORS_HEX, type NetworkArchitecture } from '../cnn3d/types';
import { callAIAnalysis, isAIConfigured, type AnalysisLevel } from '../../services/aiService';

/* ============================================
   AIDiagnosisPanel — AI 智能诊断面板
   - 可开关（AI 辅助开关）
   - 展示建议 / 异常检测
   - 与 3D 联动高亮重点层
   - 支持三档分析程度：精简 / 标准 / 深度
   - 已接入 LLM API（配置后调用真实大模型，未配置降级为本地规则）
   ============================================ */

interface AIDiagnosisPanelProps {
  width?: number;
}

/* ---------- 本地建议类型（与 API 返回兼容） ---------- */
interface AISuggestion {
  type: 'info' | 'warning' | 'success' | 'critical';
  title: string;
  description: string;
  layerId?: string;
  layerType?: string;
  confidence: number;
}

export function AIDiagnosisPanel({ width = 280 }: AIDiagnosisPanelProps) {
  const {
    aiEnabled,
    toggleAI,
    selectLayer,
    selectedLayerId,
    architecture,
    isPlaying,
    currentStep,
  } = useWorkbench();

  const [showDetails, setShowDetails] = useState(true);
  // 分析程度：精简 / 标准（默认） / 深度
  const [analysisLevel, setAnalysisLevel] = useState<AnalysisLevel>('standard');

  // AI 建议状态管理
  const [suggestions, setSuggestions] = useState<AISuggestion[]>([]);
  const [aiLoading, setAiLoading] = useState(false);
  const [aiError, setAiError] = useState<string | null>(null);
  const [useRemoteAI, setUseRemoteAI] = useState(false);
  // 防抖标记：避免短时间内重复请求
  const lastRequestRef = useRef<string>('');

  // 检测 AI 是否已配置
  useEffect(() => {
    setUseRemoteAI(isAIConfigured());
  }, []);

  // 获取 AI 建议（配置了 API 时调用远程，否则用本地规则）
  const fetchSuggestions = useCallback(async () => {
    // 生成请求标识用于防抖
    const reqKey = `${analysisLevel}-${currentStep}-${isPlaying}-${architecture.layers.length}`;
    if (reqKey === lastRequestRef.current) return;
    lastRequestRef.current = reqKey;

    if (isAIConfigured()) {
      setAiLoading(true);
      setAiError(null);
      try {
        const result = await callAIAnalysis({
          architecture: {
            name: architecture.name,
            layers: architecture.layers.map(l => ({
              id: l.id,
              name: l.name,
              type: l.type,
              params: l.params,
              nodeCount: l.nodeCount,
              kernelSize: l.kernelSize,
              outputShape: l.outputShape,
            })),
          },
          currentStep,
          isPlaying,
          analysisLevel,
        });
        // API 返回的建议直接使用
        setSuggestions(result as AISuggestion[]);
        setUseRemoteAI(true);
      } catch (err) {
        setAiError(err instanceof Error ? err.message : 'AI 分析请求失败');
        // 降级到本地规则
        setSuggestions(generateAISuggestions(architecture, currentStep, isPlaying, analysisLevel));
        setUseRemoteAI(false);
      } finally {
        setAiLoading(false);
      }
    } else {
      // 未配置 API，使用本地规则
      setSuggestions(generateAISuggestions(architecture, currentStep, isPlaying, analysisLevel));
      setUseRemoteAI(false);
    }
  }, [architecture, currentStep, isPlaying, analysisLevel]);

  // 依赖变化时触发分析（带防抖延迟）
  useEffect(() => {
    if (!aiEnabled) return;
    const timer = setTimeout(() => {
      fetchSuggestions();
    }, 300);
    return () => clearTimeout(timer);
  }, [aiEnabled, fetchSuggestions]);

  // 手动刷新
  const handleRefresh = useCallback(() => {
    lastRequestRef.current = ''; // 重置防抖
    fetchSuggestions();
  }, [fetchSuggestions]);

  return (
    <div
      className="flex h-full flex-col border-l border-white/[0.06] bg-[#0c0e17]"
      style={{ width: `${width}px` }}
    >
      {/* 顶部标题 + 开关 */}
      <div className="flex items-center justify-between border-b border-white/[0.06] px-4 py-3">
        <div className="flex items-center gap-2">
          <div className={`flex h-5 w-5 items-center justify-center rounded-md ${aiEnabled ? 'bg-amber-400/20 text-amber-400' : 'bg-white/5 text-muted-foreground/60'}`}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83" />
              <circle cx="12" cy="12" r="3" />
            </svg>
          </div>
          <span className="text-[11px] font-semibold uppercase tracking-wider text-foreground/85">
            AI 诊断
          </span>
        </div>

        {/* 开关 */}
        <button
          onClick={toggleAI}
          className={`relative inline-flex h-4 w-7 items-center rounded-full transition-all ${
            aiEnabled ? 'bg-amber-400/80' : 'bg-white/10'
          }`}
        >
          <span
            className={`inline-block h-3 w-3 transform rounded-full bg-white shadow-lg transition-transform ${
              aiEnabled ? 'translate-x-[14px]' : 'translate-x-1'
            }`}
          />
        </button>
      </div>

      {/* AI 关闭状态 */}
      {!aiEnabled && (
        <div className="flex flex-1 flex-col items-center justify-center p-6 text-center">
          <div className="mb-2 flex h-10 w-10 items-center justify-center rounded-full bg-white/[0.03]">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-muted-foreground/50">
              <path d="M18.36 6.64A9 9 0 1 1 21 12M13 4v8h8" />
            </svg>
          </div>
          <p className="text-[11px] font-medium text-foreground/80">AI 诊断已关闭</p>
          <p className="mt-1 text-[9px] text-muted-foreground">
            启用 AI 可获得智能训练建议
          </p>
          <button
            onClick={toggleAI}
            className="mt-3 rounded-md bg-amber-400/15 px-3 py-1 text-[10px] font-semibold text-amber-400 transition-all hover:bg-amber-400/25"
          >
            启用 AI
          </button>
        </div>
      )}

      {/* AI 开启状态 */}
      {aiEnabled && (
        <div className="flex flex-1 flex-col overflow-hidden">
          {/* 分析程度选择器：精简 / 标准 / 深度 */}
          <div className="border-b border-white/[0.04] px-3 py-2">
            <span className="mb-1.5 block text-[9px] uppercase tracking-wider text-muted-foreground">
              分析程度
            </span>
            <div className="flex gap-1 rounded-md bg-white/[0.03] p-0.5">
              {([
                { key: 'brief', label: '精简' },
                { key: 'standard', label: '标准' },
                { key: 'deep', label: '深度' },
              ] as const).map((opt) => (
                <button
                  key={opt.key}
                  onClick={() => setAnalysisLevel(opt.key)}
                  className={`flex-1 rounded px-2 py-1 text-[10px] font-medium transition-all ${
                    analysisLevel === opt.key
                      ? 'bg-amber-400/20 text-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.15)]'
                      : 'text-muted-foreground/70 hover:text-foreground/80'
                  }`}
                >
                  {opt.label}
                </button>
              ))}
            </div>
          </div>

          {/* 模式指示 + 刷新 */}
          <div className="flex items-center justify-between border-b border-white/[0.04] px-3 py-2">
            <span className="text-[9px] uppercase text-muted-foreground">
              {aiLoading ? '分析中...' : `${suggestions.length} 条建议`}
            </span>
            <div className="flex items-center gap-2">
              <button
                onClick={handleRefresh}
                disabled={aiLoading}
                title="重新分析"
                className="text-[9px] text-muted-foreground/70 hover:text-foreground/80 disabled:opacity-40"
              >
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className={aiLoading ? 'animate-spin' : ''}>
                  <path d="M21 12a9 9 0 1 1-3-6.7L21 8" />
                  <path d="M21 3v5h-5" />
                </svg>
              </button>
              <button
                onClick={() => setShowDetails(!showDetails)}
                className="text-[9px] text-muted-foreground/70 hover:text-foreground/80"
              >
                {showDetails ? '隐藏' : '详情'}
              </button>
            </div>
          </div>

          {/* 加载状态 */}
          {aiLoading && (
            <div className="flex items-center justify-center py-6">
              <div className="h-6 w-6 animate-spin rounded-full border-2 border-transparent border-t-amber-400" />
              <span className="ml-2 text-[10px] text-muted-foreground">AI 分析中...</span>
            </div>
          )}

          {/* 错误提示 */}
          {aiError && !aiLoading && (
            <div className="mx-3 mt-2 rounded-md border border-red-400/20 bg-red-400/5 p-2">
              <p className="text-[9px] text-red-400/80">{aiError}</p>
              <p className="mt-0.5 text-[8px] text-muted-foreground/60">已降级为本地规则分析</p>
            </div>
          )}

          {/* 建议列表 */}
          {!aiLoading && (
            <div className="flex-1 space-y-1.5 overflow-y-auto p-3">
              {suggestions.map((suggestion, idx) => (
                <SuggestionCard
                  key={idx}
                  suggestion={suggestion}
                  isSelected={selectedLayerId === suggestion.layerId}
                  onFocus={() => suggestion.layerId && selectLayer(suggestion.layerId)}
                  showDetails={showDetails}
                />
              ))}
            </div>
          )}

          {/* 底部：AI 状态 */}
          <div className="border-t border-white/[0.06] bg-[#0a0c14] px-3 py-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <span className={`size-1.5 rounded-full ${useRemoteAI ? 'bg-emerald-400' : isPlaying ? 'animate-pulse bg-amber-400' : 'bg-muted-foreground/50'}`} />
                <span className="font-mono text-[9px] text-muted-foreground">
                  {useRemoteAI ? (aiLoading ? 'API 请求中' : 'API 已连接') : (isPlaying ? '本地规则分析' : '待命')}
                </span>
              </div>
              <span className="font-mono text-[9px] text-muted-foreground/60">
                {useRemoteAI ? 'LLM v3.0' : 'v2.0'}
              </span>
            </div>
            <p className="mt-1 text-center text-[10px] text-muted-foreground/60">
              {useRemoteAI
                ? '已接入大模型 API，正在使用真实 AI 分析'
                : '本地规则分析 · 前往 AI 设置配置 API Key'}
            </p>
          </div>
        </div>
      )}
    </div>
  );
}

/* ---------- 单个建议卡片 ---------- */
function SuggestionCard({
  suggestion,
  isSelected,
  onFocus,
  showDetails,
}: {
  suggestion: AISuggestion;
  isSelected: boolean;
  onFocus: () => void;
  showDetails: boolean;
}) {
  const colorMap = {
    info: { border: 'border-primary/20', bg: 'bg-primary/5', text: 'text-primary', icon: 'info' },
    warning: { border: 'border-amber-400/25', bg: 'bg-amber-400/10', text: 'text-amber-400', icon: 'warn' },
    success: { border: 'border-emerald-400/25', bg: 'bg-emerald-400/10', text: 'text-emerald-400', icon: 'check' },
    critical: { border: 'border-red-400/30', bg: 'bg-red-400/10', text: 'text-red-400', icon: 'alert' },
  };
  const colors = colorMap[suggestion.type];

  return (
    <button
      onClick={onFocus}
      className={`block w-full rounded-lg border p-2.5 text-left transition-all hover:bg-white/[0.02] ${colors.border} ${colors.bg} ${
        isSelected ? 'ring-1 ring-' + colors.text : ''
      }`}
    >
      <div className="flex items-start gap-2">
        {/* 图标 */}
        <div className={`mt-0.5 flex-shrink-0 ${colors.text}`}>
          {suggestion.type === 'success' && (
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M20 6 9 17l-5-5" />
            </svg>
          )}
          {suggestion.type === 'warning' && (
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M12 9v4M12 17h.01" />
              <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
            </svg>
          )}
          {suggestion.type === 'critical' && (
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <circle cx="12" cy="12" r="10" />
              <path d="M15 9l-6 6M9 9l6 6" />
            </svg>
          )}
          {suggestion.type === 'info' && (
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <circle cx="12" cy="12" r="10" />
              <path d="M12 16v-4M12 8h.01" />
            </svg>
          )}
        </div>

        <div className="min-w-0 flex-1">
          <h4 className={`text-[10.5px] font-semibold ${colors.text}`}>
            {suggestion.title}
          </h4>
          {showDetails && (
            <p className="mt-1 text-[9.5px] leading-relaxed text-slate-300">
              {suggestion.description}
            </p>
          )}

          {/* 关联层 */}
          {suggestion.layerId && (
            <div className="mt-1.5 flex items-center gap-1">
              <span className="text-[8px] text-muted-foreground">→</span>
              <span className="font-mono text-[9px] text-foreground/80">
                {suggestion.layerId}
              </span>
              <div
                className="ml-1 size-1 rounded-full"
                style={{
                  backgroundColor: LAYER_COLORS_HEX[
                    (suggestion.layerType as keyof typeof LAYER_COLORS_HEX) || 'conv'
                  ] ?? '#999',
                }}
              />
            </div>
          )}
        </div>

        {/* 置信度 */}
        <div className="flex-shrink-0">
          <span className="font-mono text-[8px] text-muted-foreground/70">
            {Math.round(suggestion.confidence * 100)}%
          </span>
        </div>
      </div>
    </button>
  );
}

/* ---------- 根据架构生成 AI 建议（支持三档分析程度） ----------
   - brief:    仅 3 条核心结论（结构健康度 / 核心问题 / 优化建议）
   - standard: 标准 5 项诊断（保留原有完整逻辑）
   - deep:     标准版 + 每层评估 + 调参方案 + 训练路线图
   【扩展位】后续接入大模型 API 时，level 直接对应 Prompt 长度与 max_tokens。
----------------------------------------------------------- */
function generateAISuggestions(
  architecture: NetworkArchitecture,
  currentStep: number,
  isPlaying: boolean,
  level: AnalysisLevel = 'standard'
): AISuggestion[] {
  // 精简版：仅 3 条核心结论
  if (level === 'brief') {
    return generateBriefSuggestions(architecture, currentStep, isPlaying);
  }

  // 标准版：完整 5 项诊断
  const standard = generateStandardSuggestions(architecture, currentStep, isPlaying);

  // 深度版：标准版 + 深度扩展内容
  if (level === 'deep') {
    return [...standard, ...generateDeepExtras(architecture, currentStep)];
  }

  return standard;
}

/* ---------- 精简版：3 条核心结论 ---------- */
function generateBriefSuggestions(
  architecture: NetworkArchitecture,
  currentStep: number,
  isPlaying: boolean
): AISuggestion[] {
  const totalParams = architecture.layers.reduce((sum, l) => sum + l.params, 0);
  const convLayers = architecture.layers.filter((l) => l.type === 'conv');
  const biggestConv = convLayers.length > 0
    ? convLayers.reduce((big, l) => (l.params > big.params ? l : big), convLayers[0])
    : null;

  // 1. 结构健康度
  const healthScore = Math.min(0.98, 0.7 + architecture.layers.length * 0.03);

  // 2. 核心问题（按优先级判定）
  let coreIssue: AISuggestion;
  if (biggestConv && totalParams > 0 && biggestConv.params > totalParams * 0.3) {
    coreIssue = {
      type: 'warning',
      title: '核心问题：参数集中',
      description: `${biggestConv.name} 参数占比过高 (${((biggestConv.params / totalParams) * 100).toFixed(0)}%)，存在过拟合风险。`,
      layerId: biggestConv.id,
      layerType: biggestConv.type,
      confidence: 0.78,
    };
  } else if (isPlaying && currentStep < 10) {
    coreIssue = {
      type: 'warning',
      title: '核心问题：训练初期',
      description: `仅 ${currentStep} 步，激活值尚未稳定，建议继续观察。`,
      confidence: 0.7,
    };
  } else {
    coreIssue = {
      type: 'info',
      title: '核心问题：结构均衡',
      description: '各层参数分布相对均衡，无明显瓶颈层。',
      confidence: 0.75,
    };
  }

  // 3. 优化建议
  const optimization: AISuggestion = {
    type: 'success',
    title: '优化建议',
    description: biggestConv
      ? `优先关注 ${biggestConv.name} 的正则化（Dropout / L2），并监控验证集损失。`
      : '保持当前训练节奏，关注验证集指标变化。',
    layerId: biggestConv?.id,
    layerType: biggestConv?.type,
    confidence: 0.8,
  };

  return [
    {
      type: 'success',
      title: `结构健康度 ${(healthScore * 100).toFixed(0)}%`,
      description: `${architecture.layers.length} 层 / ${(totalParams / 1000).toFixed(1)}K 参数，整体结构合理。`,
      confidence: healthScore,
    },
    coreIssue,
    optimization,
  ];
}

/* ---------- 标准版：5 项诊断（保留原有完整逻辑） ---------- */
function generateStandardSuggestions(
  architecture: NetworkArchitecture,
  currentStep: number,
  isPlaying: boolean
): AISuggestion[] {
  const suggestions: AISuggestion[] = [];

  // 1. 基于层类型的分析
  const typeCount = architecture.layers.reduce<Record<string, number>>((acc, l) => {
    acc[l.type] = (acc[l.type] || 0) + 1;
    return acc;
  }, {});

  // 2. 总体参数规模
  const totalParams = architecture.layers.reduce((sum, l) => sum + l.params, 0);

  // 总体
  suggestions.push({
    type: 'success',
    title: '网络结构健康',
    description: `当前网络包含 ${architecture.layers.length} 层，总参数 ${(totalParams / 1000).toFixed(1)}K。层分布合理，特征提取与分类平衡。`,
    confidence: 0.85,
  });

  // 3. 针对卷积层
  if ((typeCount.conv || 0) > 0) {
    const convLayers = architecture.layers.filter((l) => l.type === 'conv');
    const biggestConv = convLayers.reduce((big, l) => (l.params > big.params ? l : big), convLayers[0]);
    if (biggestConv) {
      suggestions.push({
        type: 'info',
        title: '特征层参数关注',
        description: `${biggestConv.name} 拥有最多参数 (${(biggestConv.params / 1000).toFixed(1)}K)。训练时该层需要更多优化迭代。`,
        layerId: biggestConv.id,
        layerType: biggestConv.type,
        confidence: 0.72,
      });
    }
  }

  // 4. 训练相关
  if (isPlaying) {
    suggestions.push({
      type: currentStep > 10 ? 'info' : 'warning',
      title: currentStep > 10 ? '训练进度正常' : '训练初期',
      description: currentStep > 10
        ? `已进行 ${currentStep} 步训练，激活值稳定分布，网络正在稳步学习。`
        : `训练仅进行 ${currentStep} 步，激活值可能不稳定，建议继续观察。`,
      confidence: 0.68,
    });
  }

  // 5. 如果有 pooling 层
  if ((typeCount.pool || 0) > 0) {
    suggestions.push({
      type: 'info',
      title: '池化层降维',
      description: `使用 ${typeCount.pool} 个池化层进行空间降维，有助于降低计算量并减少过拟合。`,
      confidence: 0.9,
    });
  }

  // 6. 输出层
  const outputLayer = architecture.layers.find((l) => l.type === 'output');
  if (outputLayer && outputLayer.nodeCount > 0) {
    suggestions.push({
      type: 'info',
      title: `分类任务：${outputLayer.nodeCount} 类别`,
      description: `输出层配置 ${outputLayer.nodeCount} 个类别，使用 ${outputLayer.activation || 'Softmax'} 激活。`,
      layerId: outputLayer.id,
      layerType: outputLayer.type,
      confidence: 0.95,
    });
  }

  return suggestions;
}

/* ---------- 深度版扩展内容 ----------
   - 每层详细评估（参数占比 + 类型特征）
   - 具体调参方案（基于当前 learning_rate / batch_size）
   - 训练改进路线图（初期 / 中期 / 后期）
   注：当前 WorkbenchContext 未透传 hyperparams，此处使用默认值 (lr=0.001, bs=32)。
       后续可由调用方传入真实超参以提升建议准确性。          */
function generateDeepExtras(
  architecture: NetworkArchitecture,
  currentStep: number
): AISuggestion[] {
  const extras: AISuggestion[] = [];
  const totalParams = architecture.layers.reduce((sum, l) => sum + l.params, 0);

  // ===== 每层详细评估 =====
  const typeFeatureMap: Record<string, string> = {
    input: '输入层，负责原始数据接入，无参数',
    conv: '卷积层，提取局部空间特征',
    pool: '池化层，进行空间降维，无参数',
    fc: '全连接层，整合特征进行分类',
    output: '输出层，产出最终预测分布',
    norm: '归一化层，稳定激活分布，加速收敛',
    dropout: 'Dropout 层，随机失活以抑制过拟合',
  };

  architecture.layers.forEach((layer) => {
    const ratio = totalParams > 0 ? (layer.params / totalParams) * 100 : 0;
    const kernelInfo = layer.kernelSize ? `核 ${layer.kernelSize}，` : '';
    const feature = typeFeatureMap[layer.type] ?? `${layer.type} 层`;

    extras.push({
      type: ratio > 30 ? 'warning' : 'info',
      title: `${layer.name} 评估`,
      description: `参数 ${(layer.params / 1000).toFixed(2)}K（占比 ${ratio.toFixed(1)}%）· ${kernelInfo}${feature}。`,
      layerId: layer.id,
      layerType: layer.type,
      confidence: 0.65,
    });
  });

  // ===== 具体调参方案 =====
  // 默认超参（Context 未透传 hyperparams 时使用）
  const learning_rate = 0.001;
  const batch_size = 32;

  const tuningAdvice: string[] = [];
  if (learning_rate > 0.01) {
    tuningAdvice.push(`学习率 ${learning_rate} 偏高，建议降至 1e-3 以下以防发散`);
  } else if (learning_rate < 0.0001) {
    tuningAdvice.push(`学习率 ${learning_rate} 偏低，收敛缓慢，可尝试 5e-4 ~ 1e-3`);
  } else {
    tuningAdvice.push(`学习率 ${learning_rate} 处于合理区间，建议配合余弦退火调度`);
  }
  if (batch_size > 128) {
    tuningAdvice.push(`batch_size=${batch_size} 偏大，泛化能力可能下降，建议 32~64`);
  } else if (batch_size < 8) {
    tuningAdvice.push(`batch_size=${batch_size} 过小，梯度噪声大，建议提升至 16~32`);
  } else {
    tuningAdvice.push(`batch_size=${batch_size} 适中，BN 统计量稳定`);
  }

  extras.push({
    type: 'info',
    title: '调参方案建议',
    description: tuningAdvice.join('；') + '。',
    confidence: 0.7,
  });

  // ===== 训练改进路线图 =====
  const phase: '初期' | '中期' | '后期' = currentStep < 10 ? '初期' : currentStep < 50 ? '中期' : '后期';
  const roadmap: Record<'初期' | '中期' | '后期', string> = {
    初期: '关注损失是否下降、梯度是否爆炸/消失，可启用 warmup 稳定前几步',
    中期: '监控训练/验证损失差距，适时加入正则化（Dropout / L2）防过拟合',
    后期: '降低学习率进行精调，关注验证集指标，必要时启用早停',
  };
  extras.push({
    type: 'success',
    title: `训练路线图 · 当前 [${phase}]`,
    description: `初期：${roadmap.初期}。中期：${roadmap.中期}。后期：${roadmap.后期}。`,
    confidence: 0.73,
  });

  return extras;
}

export default AIDiagnosisPanel;
