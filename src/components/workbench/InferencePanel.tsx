import { useState, useRef, useCallback, useEffect } from 'react';
import { apiService } from '../../services/api';
import { useWorkbench } from './WorkbenchContext';

export interface InferenceResult {
  activations: Record<string, number[]>;
  predictions: Array<{ class_id: number; probability: number }>;
  input_size: [number, number];
  success: boolean;
  inference_time?: number;
}

interface InferencePanelProps {
  onActivationsChange?: (activations: Record<string, number[]>) => void;
  onInferenceComplete?: (result: InferenceResult) => void;
  onOpenImporter?: () => void;
}

function mapActivationsToLayerIds(
  rawActivations: Record<string, number[]>,
  layers: Array<{ id: string; name: string }>
): Record<string, number[]> {
  const result: Record<string, number[]> = {};
  
  const rawKeys = Object.keys(rawActivations);
  if (rawKeys.length === 0) return result;
  
  // Strategy 1: Exact ID match
  layers.forEach((layer) => {
    if (layer.id in rawActivations) {
      result[layer.id] = rawActivations[layer.id];
    }
  });
  
  // Strategy 2: Case-insensitive name/ID match
  const rawKeysLower = rawKeys.map((k) => k.toLowerCase());
  layers.forEach((layer) => {
    if (result[layer.id]) return;
    
    const layerIdLower = layer.id.toLowerCase();
    const layerNameLower = layer.name.toLowerCase();
    
    const idxById = rawKeysLower.indexOf(layerIdLower);
    if (idxById >= 0) {
      result[layer.id] = rawActivations[rawKeys[idxById]];
      return;
    }
    
    const idxByName = rawKeysLower.indexOf(layerNameLower);
    if (idxByName >= 0) {
      result[layer.id] = rawActivations[rawKeys[idxByName]];
      return;
    }
  });
  
  // Strategy 3: Fuzzy match (contains)
  layers.forEach((layer) => {
    if (result[layer.id]) return;
    
    const layerIdLower = layer.id.toLowerCase();
    const layerNameLower = layer.name.toLowerCase();
    
    for (let i = 0; i < rawKeys.length; i++) {
      const rawKeyLower = rawKeys[i].toLowerCase();
      if (
        rawKeyLower.includes(layerIdLower) ||
        layerIdLower.includes(rawKeyLower) ||
        rawKeyLower.includes(layerNameLower) ||
        layerNameLower.includes(rawKeyLower)
      ) {
        result[layer.id] = rawActivations[rawKeys[i]];
        break;
      }
    }
  });
  
  // Strategy 4: Map by positional index (for ordered layers)
  const paramLayers = layers.filter((l) => !result[l.id]);
  const unusedKeys = rawKeys.filter((k) => !Object.values(result).includes(rawActivations[k]));
  
  if (paramLayers.length > 0 && unusedKeys.length > 0) {
    const step = Math.max(1, Math.floor(unusedKeys.length / paramLayers.length));
    paramLayers.forEach((layer, idx) => {
      const keyIdx = Math.min(idx * step, unusedKeys.length - 1);
      result[layer.id] = rawActivations[unusedKeys[keyIdx]];
    });
  }
  
  return result;
}

export function InferencePanel({
  onActivationsChange,
  onInferenceComplete,
  onOpenImporter,
}: InferencePanelProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isInferring, setIsInferring] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [inferenceResult, setInferenceResult] = useState<InferenceResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);

  const {
    serverStatus,
    currentModelId,
    currentModelName,
    setServerStatus,
    architecture,
  } = useWorkbench();

  // 检查服务器状态
  const checkServer = useCallback(async () => {
    setServerStatus('connecting', null);
    try {
      const result = await apiService.getServiceStatus();
      if (result.code === 200 && result.data.status === 'online') {
        setServerStatus('connected', null);
        return true;
      }
      throw new Error(result.message || '服务响应异常');
    } catch (err) {
      setServerStatus('error', err instanceof Error ? err.message : '未知错误');
    }
    return false;
  }, [setServerStatus]);

  // 初始化时检查服务器
  useEffect(() => {
    if (isOpen) {
      checkServer();
    }
  }, [isOpen, checkServer]);

  // 潜在问题修复：组件卸载时释放 Object URL，避免 Blob 内存泄漏
  useEffect(() => {
    return () => {
      if (previewUrl) URL.revokeObjectURL(previewUrl);
    };
  }, [previewUrl]);

  // 处理文件选择
  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setSelectedFile(file);
    setError(null);
    setInferenceResult(null);

    // 创建预览 URL
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(URL.createObjectURL(file));
  }, [previewUrl]);

  // 运行推理
  const runInference = useCallback(async () => {
    if (!selectedFile) {
      setError('请先选择一张图片');
      return;
    }

    if (serverStatus !== 'connected') {
      setError('服务器未连接，请先点击"检测连接"');
      return;
    }

    if (!currentModelId) {
      setError('请先导入模型，再进行推理');
      return;
    }

    setIsInferring(true);
    setError(null);

    try {
      const result = await apiService.inferenceImage(currentModelId, selectedFile);
      const data = result.data;

      const layerInfo = architecture.layers.map((l) => ({ id: l.id, name: l.name }));
      const mappedActivations = mapActivationsToLayerIds(data.activations || {}, layerInfo);

      const inferenceResult: InferenceResult = {
        activations: mappedActivations,
        predictions: data.predictions || [],
        input_size: data.input_size || [224, 224],
        success: data.success !== false,
        inference_time: data.inference_time,
      };

      setInferenceResult(inferenceResult);

      if (onActivationsChange && inferenceResult.activations) {
        onActivationsChange(inferenceResult.activations);
      }
      if (onInferenceComplete) {
        onInferenceComplete(inferenceResult);
      }
    } catch (err) {
      setError(`推理失败: ${err instanceof Error ? err.message : '未知错误'}`);
    } finally {
      setIsInferring(false);
    }
  }, [selectedFile, serverStatus, currentModelId, architecture, onActivationsChange, onInferenceComplete]);

  // 清除结果
  const clearResult = useCallback(() => {
    setInferenceResult(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
    setSelectedFile(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
    if (onActivationsChange) {
      onActivationsChange({});
    }
  }, [previewUrl, onActivationsChange]);

  // 格式化概率为百分比
  const formatProbability = (prob: number): string => {
    return (prob * 100).toFixed(2) + '%';
  };

  const canRun = selectedFile && serverStatus === 'connected' && currentModelId && !isInferring;
  const isDisabled = !canRun;

  return (
    <>
      {/* 触发按钮 */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs font-semibold transition-all ${
          isOpen
            ? 'border-cyan-400/40 bg-cyan-400/10 text-cyan-400'
            : 'border-white/[0.08] bg-white/[0.02] text-muted-foreground hover:border-primary/30 hover:text-foreground'
        }`}
        title="图片推理 - 上传图片获取真实激活值"
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <circle cx="8.5" cy="8.5" r="1.5" />
          <polyline points="21 15 16 10 5 21" />
        </svg>
        推理
      </button>

      {/* 面板内容 */}
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setIsOpen(false)} />

          <div className="relative w-full max-w-lg overflow-hidden rounded-xl border border-border bg-card shadow-2xl">
            {/* 头部 */}
            <div className="flex items-center justify-between border-b border-border px-5 py-4">
              <div>
                <h2 className="text-base font-semibold">图片推理</h2>
                <p className="text-[10px] text-muted-foreground mt-0.5">
                  上传图片，通过模型推理获取真实激活值
                </p>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="rounded-full p-1 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>

            {/* 内容 */}
            <div className="p-5 space-y-4">
              {/* 服务器状态 */}
              <div className="flex items-center gap-3 rounded-lg border border-border bg-muted/20 p-3">
                <div className="flex items-center gap-2 flex-1">
                  <span
                    className={`inline-block h-2 w-2 rounded-full ${
                      serverStatus === 'connected'
                        ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]'
                        : serverStatus === 'error'
                        ? 'bg-red-400 shadow-[0_0_8px_rgba(248,113,113,0.6)]'
                        : serverStatus === 'connecting'
                        ? 'bg-amber-400 animate-pulse'
                        : 'bg-slate-500'
                    }`}
                  />
                  <span className="text-xs text-muted-foreground">
                    {serverStatus === 'connected'
                      ? '已连接本地服务'
                      : serverStatus === 'error'
                      ? '服务器未响应'
                      : serverStatus === 'connecting'
                      ? '连接中...'
                      : '未连接'}
                  </span>
                </div>
                <button
                  onClick={checkServer}
                  disabled={serverStatus === 'connecting'}
                  className="rounded-md border border-border bg-background px-2 py-1 text-[10px] font-medium hover:bg-muted transition-colors disabled:opacity-50"
                >
                  {serverStatus === 'connecting' ? '检测中...' : '检测连接'}
                </button>
              </div>

              {/* 模型名称 */}
              <div className="flex items-center justify-between rounded-lg border border-border bg-muted/10 p-3">
                <div className="flex items-center gap-2.5 min-w-0">
                  <div className="flex h-8 w-8 flex-shrink-0 items-center justify-center rounded-md border border-dashed border-white/[0.12] bg-white/[0.02]">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-muted-foreground/60">
                      <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
                    </svg>
                  </div>
                  <div className="min-w-0">
                    <p className="text-[10px] font-medium uppercase text-muted-foreground">使用的模型</p>
                    <p className="truncate text-sm font-mono text-foreground">
                      {currentModelId ? currentModelName : '未选择（请先导入模型）'}
                    </p>
                  </div>
                </div>
                {onOpenImporter && !currentModelId && (
                  <button
                    onClick={() => {
                      onOpenImporter();
                      setIsOpen(false);
                    }}
                    className="flex-shrink-0 text-[10px] font-semibold text-primary transition-colors hover:text-primary/80"
                  >
                    去导入 →
                  </button>
                )}
              </div>

              {/* 图片上传 */}
              <div
                onClick={() => fileInputRef.current?.click()}
                className={`relative cursor-pointer rounded-lg border-2 border-dashed transition-all ${
                  selectedFile
                    ? 'border-emerald-400/50 bg-emerald-400/5'
                    : 'border-border bg-muted/20 hover:border-primary/50 hover:bg-muted/30'
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden"
                />

                {previewUrl ? (
                  <div className="p-2">
                    <img
                      src={previewUrl}
                      alt="Preview"
                      className="w-full h-40 object-contain rounded-md"
                    />
                    <p className="text-[10px] text-muted-foreground text-center mt-2">
                      {selectedFile?.name}
                    </p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-8 px-4">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="mb-2 text-muted-foreground">
                      <rect x="3" y="3" width="18" height="18" rx="2" />
                      <circle cx="8.5" cy="8.5" r="1.5" />
                      <polyline points="21 15 16 10 5 21" />
                    </svg>
                    <p className="text-sm font-medium text-foreground">点击选择图片</p>
                    <p className="text-[10px] text-muted-foreground mt-1">支持 JPG、PNG、WebP 等格式</p>
                  </div>
                )}
              </div>

              {/* 错误提示 */}
              {error && (
                <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-3">
                  <p className="text-sm text-destructive">{error}</p>
                </div>
              )}

              {/* 推理结果 */}
              {inferenceResult && (
                <div className="space-y-3">
                  {/* 预测结果 */}
                  <div className="rounded-lg border border-emerald-400/30 bg-emerald-400/5 p-3">
                    <div className="flex items-center justify-between mb-2">
                      <p className="text-[10px] font-semibold uppercase text-emerald-400">预测结果 (Top-5)</p>
                      {inferenceResult.inference_time !== undefined && (
                        <span className="text-[10px] text-emerald-400/70 font-mono">
                          耗时: {inferenceResult.inference_time.toFixed(2)}ms
                        </span>
                      )}
                    </div>
                    <div className="space-y-1">
                      {inferenceResult.predictions.map((pred, idx) => (
                        <div key={idx} className="flex items-center justify-between">
                          <span className="text-xs text-muted-foreground">类别 {pred.class_id}</span>
                          <div className="flex items-center gap-2">
                            <div className="w-20 h-1.5 bg-muted rounded-full overflow-hidden">
                              <div
                                className="h-full bg-emerald-400 rounded-full transition-all"
                                style={{ width: `${pred.probability * 100}%` }}
                              />
                            </div>
                            <span className="text-xs font-mono text-emerald-400 w-14 text-right">
                              {formatProbability(pred.probability)}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* 激活值统计 */}
                  <div className="rounded-lg border border-cyan-400/30 bg-cyan-400/5 p-3">
                    <p className="text-[10px] font-semibold uppercase text-cyan-400 mb-2">
                      激活值已更新
                    </p>
                    <p className="text-xs text-muted-foreground">
                      共提取 {Object.keys(inferenceResult.activations).length} 层的激活值
                    </p>
                    <p className="text-[10px] text-muted-foreground/80 mt-1">
                      切换到「激活视图」查看 3D 可视化效果
                    </p>
                  </div>
                </div>
              )}

              {/* 操作按钮 */}
              <div className="flex gap-3">
                {inferenceResult && (
                  <button
                    onClick={clearResult}
                    className="flex-1 rounded-lg border border-border bg-muted/20 py-2.5 text-sm font-medium text-foreground transition-all hover:bg-muted/30"
                  >
                    清除
                  </button>
                )}
                <button
                  onClick={runInference}
                  disabled={isDisabled}
                  title={
                    serverStatus !== 'connected'
                      ? '服务器未连接，无法运行推理'
                      : !currentModelId
                      ? '请先导入模型'
                      : !selectedFile
                      ? '请先选择一张图片'
                      : '运行推理'
                  }
                  className={`flex-1 rounded-lg py-2.5 text-sm font-semibold transition-all ${
                    canRun
                      ? 'bg-cyan-500 text-white hover:bg-cyan-600 hover:shadow-[0_0_16px_rgba(6,182,212,0.25)]'
                      : 'cursor-not-allowed bg-muted/60 text-muted-foreground opacity-40'
                  }`}
                >
                  {isInferring ? (
                    <span className="flex items-center justify-center gap-2">
                      <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                      </svg>
                      推理中...
                    </span>
                  ) : (
                    '运行推理'
                  )}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
}

export default InferencePanel;
