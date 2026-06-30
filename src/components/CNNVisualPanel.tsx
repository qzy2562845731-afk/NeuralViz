import { useState, useEffect, useRef, useCallback } from 'react';
import { trainingApi } from '@/services/api';
import type { VisualizationData } from '@/services/api';
import { useWorkbench } from './workbench/WorkbenchContext';
import { ExperimentSelector } from './workbench/ExperimentSelector';

interface CNNVisualPanelProps {
  experimentId: string | null;
  hasImageData: boolean;
}

type PanelTab = 'feature_maps' | 'kernels' | 'attention';

const HEATMAP_COLORS = [
  { pos: 0, r: 10, g: 20, b: 80 },
  { pos: 0.2, r: 20, g: 60, b: 180 },
  { pos: 0.4, r: 30, g: 160, b: 200 },
  { pos: 0.6, r: 50, g: 200, b: 80 },
  { pos: 0.8, r: 240, g: 220, b: 40 },
  { pos: 1, r: 230, g: 40, b: 40 },
];

function getHeatmapColor(value: number): string {
  const v = Math.max(0, Math.min(1, value));
  for (let i = 0; i < HEATMAP_COLORS.length - 1; i++) {
    const c1 = HEATMAP_COLORS[i];
    const c2 = HEATMAP_COLORS[i + 1];
    if (v >= c1.pos && v <= c2.pos) {
      const t = (v - c1.pos) / (c2.pos - c1.pos);
      const r = Math.round(c1.r + (c2.r - c1.r) * t);
      const g = Math.round(c1.g + (c2.g - c1.g) * t);
      const b = Math.round(c1.b + (c2.b - c1.b) * t);
      return `rgb(${r}, ${g}, ${b})`;
    }
  }
  return `rgb(${HEATMAP_COLORS[HEATMAP_COLORS.length - 1].r}, ${HEATMAP_COLORS[HEATMAP_COLORS.length - 1].g}, ${HEATMAP_COLORS[HEATMAP_COLORS.length - 1].b})`;
}

function getGrayColor(value: number, min: number, max: number): string {
  if (max === min) return 'rgb(128, 128, 128)';
  const v = Math.max(0, Math.min(1, (value - min) / (max - min)));
  const gray = Math.round(v * 255);
  return `rgb(${gray}, ${gray}, ${gray})`;
}

interface FeatureMapGridProps {
  data: number[][];
  label: string;
  cellSize?: number;
}

function FeatureMapGrid({ data, label, cellSize = 6 }: FeatureMapGridProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rows = data.length;
  const cols = data[0]?.length ?? 0;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || rows === 0 || cols === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = cols * cellSize;
    canvas.height = rows * cellSize;

    let min = Infinity, max = -Infinity;
    for (const row of data) {
      for (const v of row) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const v = max === min ? 0.5 : (data[y][x] - min) / (max - min);
        ctx.fillStyle = getHeatmapColor(v);
        ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
      }
    }
  }, [data, rows, cols, cellSize]);

  return (
    <div className="flex flex-col items-center gap-1">
      <canvas
        ref={canvasRef}
        className="rounded border border-white/[0.06]"
        style={{ imageRendering: 'pixelated' }}
      />
      <span className="text-[10px] text-muted-foreground">{label}</span>
    </div>
  );
}

interface KernelGridProps {
  data: number[][];
  label: string;
  cellSize?: number;
}

function KernelGrid({ data, label, cellSize = 10 }: KernelGridProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rows = data.length;
  const cols = data[0]?.length ?? 0;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || rows === 0 || cols === 0) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.width = cols * cellSize;
    canvas.height = rows * cellSize;

    let min = Infinity, max = -Infinity;
    for (const row of data) {
      for (const v of row) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
    }

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        ctx.fillStyle = getGrayColor(data[y][x], min, max);
        ctx.fillRect(x * cellSize, y * cellSize, cellSize, cellSize);
      }
    }
  }, [data, rows, cols, cellSize]);

  return (
    <div className="flex flex-col items-center gap-1">
      <canvas
        ref={canvasRef}
        className="rounded border border-white/[0.06]"
        style={{ imageRendering: 'pixelated' }}
      />
      <span className="text-[10px] text-muted-foreground">{label}</span>
    </div>
  );
}

interface AttentionBarChartProps {
  weights: number[];
  label?: string;
}

function AttentionBarChart({ weights, label }: AttentionBarChartProps) {
  const maxWeight = Math.max(...weights, 0.001);
  const sortedIndices = weights
    .map((w, i) => ({ w, i }))
    .sort((a, b) => b.w - a.w)
    .slice(0, 32);

  return (
    <div className="space-y-2">
      {label && <div className="text-xs font-medium text-foreground/80">{label}</div>}
      <div className="flex items-end gap-0.5 h-24 px-1">
        {sortedIndices.map(({ w, i }) => {
          const height = (w / maxWeight) * 100;
          return (
            <div
              key={i}
              className="flex-1 min-w-0 relative group"
              title={`Ch${i}: ${w.toFixed(4)}`}
            >
              <div
                className="absolute bottom-0 w-full rounded-t-sm transition-all"
                style={{
                  height: `${height}%`,
                  background: `linear-gradient(to top, #3b82f6, #60a5fa)`,
                }}
              />
            </div>
          );
        })}
      </div>
      <div className="flex justify-between text-[10px] text-muted-foreground">
        <span>Top {sortedIndices.length} 通道</span>
        <span>Max: {maxWeight.toFixed(4)}</span>
      </div>
    </div>
  );
}

export function CNNVisualPanel({ experimentId: propExperimentId, hasImageData }: CNNVisualPanelProps) {
  const { currentExperimentId, selectedVisualExperimentId, setSelectedVisualExperimentId } = useWorkbench();
  const [activeTab, setActiveTab] = useState<PanelTab>('feature_maps');
  const [isExpanded, setIsExpanded] = useState(true);
  const [visualData, setVisualData] = useState<VisualizationData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const hasDataRef = useRef(false);

  // 使用 context 中持久化的实验ID，如果没有则使用 prop 传入的（当前训练实验）
  const activeExperimentId = selectedVisualExperimentId ?? propExperimentId ?? currentExperimentId;

  const fetchVisualizations = useCallback(async () => {
    if (!activeExperimentId) return;
    try {
      setError(null);
      const res = await trainingApi.getVisualizations(activeExperimentId);
      if (res.data) {
        // 检查返回的数据结构（后端可能返回 conv_kernels 而非 kernels）
        const data = res.data as any;
        const hasFeatureMaps = data.feature_maps && Object.keys(data.feature_maps).length > 0;
        const hasKernels = (data.conv_kernels || data.kernels) && Object.keys(data.conv_kernels || data.kernels).length > 0;
        if (hasFeatureMaps || hasKernels) {
          // 标准化数据格式
          const normalized: VisualizationData = {
            ...data,
            conv_kernels: data.conv_kernels || data.kernels,
          };
          setVisualData(normalized);
          hasDataRef.current = true;
          if (pollTimerRef.current) {
            clearInterval(pollTimerRef.current);
            pollTimerRef.current = null;
          }
        }
      }
    } catch (err: any) {
      setError(err.message || '获取可视化数据失败');
    }
  }, [activeExperimentId]);

  useEffect(() => {
    hasDataRef.current = false;
    setVisualData(null);
    setError(null);

    if (!activeExperimentId) {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
      return;
    }

    if (!hasImageData) {
      // 即使没有图像数据，也允许选择其他有数据的实验查看
    }

    setLoading(true);
    fetchVisualizations().finally(() => setLoading(false));

    pollTimerRef.current = setInterval(() => {
      if (!hasDataRef.current) {
        fetchVisualizations();
      }
    }, 5000);

    return () => {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [activeExperimentId, hasImageData, fetchVisualizations]);

  const featureMapLayers = visualData?.feature_maps
    ? Object.entries(visualData.feature_maps).sort(([a], [b]) => a.localeCompare(b))
    : [];

  const kernelLayers = visualData?.conv_kernels
    ? Object.entries(visualData.conv_kernels).sort(([a], [b]) => a.localeCompare(b))
    : [];

  const attentionWeights = visualData?.attention_weights as number[] | undefined;

  const renderContent = () => {
    if (!activeExperimentId) {
      return (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" className="mb-3 text-muted-foreground/30">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <circle cx="8.5" cy="8.5" r="1.5" />
            <path d="m21 15-5-5L5 21" />
          </svg>
          <p className="text-xs text-muted-foreground/70">请先选择一个实验</p>
          <p className="mt-1 text-[10px] text-muted-foreground/50">训练完成后或从实验库选择已有实验</p>
        </div>
      );
    }

    if (loading && !visualData) {
      return (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <div className="mb-3 h-6 w-6 animate-spin rounded-full border-2 border-primary border-t-transparent" />
          <p className="text-xs text-muted-foreground/70">正在加载可视化数据...</p>
          <p className="mt-1 text-[10px] text-muted-foreground/50">每5秒自动刷新</p>
        </div>
      );
    }

    if (error) {
      return (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" className="mb-3 text-red-400/50">
            <circle cx="12" cy="12" r="10" />
            <line x1="12" y1="8" x2="12" y2="12" />
            <line x1="12" y1="16" x2="12.01" y2="16" />
          </svg>
          <p className="text-xs text-red-400/80">{error}</p>
        </div>
      );
    }

    if (!visualData) {
      return (
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" className="mb-3 text-muted-foreground/30">
            <path d="M2 12h20" />
            <path d="M12 2v20" />
            <circle cx="12" cy="12" r="10" />
          </svg>
          <p className="text-xs text-muted-foreground/70">训练完成后将显示特征图和卷积核</p>
          <p className="mt-1 text-[10px] text-muted-foreground/50">数据将在训练过程中自动生成</p>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        {activeTab === 'feature_maps' && (
          <div className="space-y-4">
            {featureMapLayers.length === 0 ? (
              <div className="py-8 text-center text-xs text-muted-foreground/50">暂无特征图数据</div>
            ) : (
              featureMapLayers.map(([layerName, channels]) => {
                const channelArray = Array.isArray(channels) ? channels : [];
                return (
                  <div key={layerName} className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
                    <div className="mb-2 flex items-center justify-between">
                      <span className="text-xs font-medium text-foreground/80">{layerName}</span>
                      <span className="text-[10px] text-muted-foreground">{channelArray.length} 通道</span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {channelArray.slice(0, 32).map((map, idx) => {
                        const mapData = Array.isArray(map) ? map : [];
                        return (
                          <FeatureMapGrid
                            key={idx}
                            data={mapData}
                            label={`Ch${idx}`}
                            cellSize={Math.max(3, Math.min(6, Math.floor(200 / Math.max(mapData[0]?.length || 16, 1))))}
                          />
                        );
                      })}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        )}

        {activeTab === 'kernels' && (
          <div className="space-y-4">
            {kernelLayers.length === 0 ? (
              <div className="py-8 text-center text-xs text-muted-foreground/50">暂无卷积核数据</div>
            ) : (
              kernelLayers.map(([layerName, kernels]) => {
                const kernelArray = Array.isArray(kernels) ? kernels : [];
                const isNestedArray = kernelArray.length > 0 && Array.isArray(kernelArray[0]) && Array.isArray((kernelArray[0] as any[])[0]);
                const kernelCount = isNestedArray ? kernelArray.length : 1;
                const displayKernels = kernelCount > 1
                  ? kernelArray.slice(0, 16)
                  : kernelArray.length > 0 ? [kernelArray] : [];
                return (
                  <div key={layerName} className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
                    <div className="mb-2 flex items-center justify-between">
                      <span className="text-xs font-medium text-foreground/80">{layerName}</span>
                      <span className="text-[10px] text-muted-foreground">
                        {isNestedArray
                          ? `${kernelArray.length} 个卷积核`
                          : kernelArray.length > 0
                            ? `${kernelArray.length}x${(kernelArray[0] as number[])?.length || 0}`
                            : '0'}
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {displayKernels.map((kernel, idx) => (
                        <KernelGrid
                          key={idx}
                          data={kernel as number[][]}
                          label={kernelCount > 1 ? `K${idx}` : ''}
                          cellSize={12}
                        />
                      ))}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        )}

        {activeTab === 'attention' && (
          <div className="space-y-4">
            {!attentionWeights || attentionWeights.length === 0 ? (
              <div className="py-8 text-center text-xs text-muted-foreground/50">
                暂无注意力权重数据
                <p className="mt-1 text-[10px]">使用 SE 或 CBAM 注意力模块后将显示</p>
              </div>
            ) : (
              <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
                <AttentionBarChart weights={attentionWeights} label="通道注意力权重分布" />
              </div>
            )}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="rounded-xl border border-white/[0.08] bg-[#0f1119] overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="flex w-full items-center justify-between px-4 py-3 transition hover:bg-white/[0.02]"
      >
        <div className="flex items-center gap-2">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
            <circle cx="12" cy="12" r="3" />
            <path d="M12 1v6m0 6v6M4.22 4.22l4.24 4.24m7.08 7.08l4.24 4.24M1 12h6m6 0h6M4.22 19.78l4.24-4.24m7.08-7.08l4.24-4.24" />
          </svg>
          <h3 className="text-sm font-bold">CNN 特征可视化</h3>
          {visualData && (
            <span className="rounded bg-primary/20 px-1.5 py-0.5 text-[10px] text-primary">已加载</span>
          )}
        </div>
        <svg
          width="14"
          height="14"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          className={`text-muted-foreground transition-transform ${isExpanded ? 'rotate-180' : ''}`}
        >
          <polyline points="6 9 12 15 18 9" />
        </svg>
      </button>

      {isExpanded && (
        <>
          <div className="border-t border-white/[0.06] px-4 pt-3 space-y-2">
            {/* 实验选择器 */}
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-muted-foreground whitespace-nowrap">实验:</span>
              <ExperimentSelector
                selectedId={selectedVisualExperimentId}
                onSelect={setSelectedVisualExperimentId}
                currentExperimentId={propExperimentId ?? currentExperimentId}
                label="选择实验查看特征"
                className="flex-1"
              />
            </div>

            <div className="flex gap-1 rounded-lg bg-white/[0.03] p-1">
              {[
                { key: 'feature_maps' as PanelTab, label: '特征图' },
                { key: 'kernels' as PanelTab, label: '卷积核' },
                { key: 'attention' as PanelTab, label: '注意力' },
              ].map(({ key, label }) => (
                <button
                  key={key}
                  onClick={() => setActiveTab(key)}
                  className={`flex-1 rounded-md px-2 py-1.5 text-xs font-medium transition-all ${
                    activeTab === key
                      ? 'bg-white/[0.08] text-foreground'
                      : 'text-muted-foreground hover:text-foreground'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          <div className="px-4 py-4 max-h-[400px] overflow-y-auto custom-scrollbar">
            {renderContent()}
          </div>

          {activeTab === 'feature_maps' && visualData && featureMapLayers.length > 0 && (
            <div className="border-t border-white/[0.06] px-4 py-2 flex items-center justify-center gap-4 text-[10px] text-muted-foreground">
              <div className="flex items-center gap-1">
                <div className="h-2 w-4 rounded" style={{ background: 'linear-gradient(to right, rgb(10,20,80), rgb(20,60,180), rgb(30,160,200), rgb(50,200,80), rgb(240,220,40), rgb(230,40,40))' }} />
                <span>低 → 高</span>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}
