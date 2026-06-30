import { useState, useMemo } from 'react';
import { useWorkbench } from './WorkbenchContext';
import { LAYER_COLORS_HEX, formatLayerShape, type LayerConfig } from '../cnn3d/types';

/* ============================================
   InspectorPanel — 右侧检查器
   - 层信息分组折叠（默认展开聚焦层）
   - 激活统计 / 参数分布 / 优化建议
   - 训练指标展示
   ============================================ */

export function InspectorPanel({ width = 340 }: { width?: number }) {
  const {
    focusedLayerId,
    isPlaying,
    currentStep,
    architecture,
    toggleInspector,
  } = useWorkbench();

  const layer = focusedLayerId
    ? architecture.layers.find((l) => l.id === focusedLayerId) ?? null
    : null;

  // 展开状态
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(
    new Set(['overview', 'shape', 'parameters'])
  );
  const toggleGroup = (name: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name); else next.add(name);
      return next;
    });
  };

  return (
    <div
      className="flex h-full flex-col border-l border-white/[0.06] bg-[#0c0e17]"
      style={{ width: `${width}px` }}
    >
      {/* 顶部标题 */}
      <div className="flex items-center justify-between border-b border-white/[0.06] px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="flex h-5 w-5 items-center justify-center rounded-md bg-emerald-400/15 text-emerald-400">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <circle cx="12" cy="12" r="3" />
              <path d="M12 1v6M12 17v6M1 12h6M17 12h6M4.93 4.93l4.24 4.24M14.83 14.83l4.24 4.24M4.93 19.07l4.24-4.24M14.83 9.17l4.24-4.24" />
            </svg>
          </div>
          <span className="text-[11px] font-semibold uppercase tracking-wider text-foreground/85">
            Inspector
          </span>
        </div>
        <div className="flex items-center gap-1">
          {layer && (
            <span className="rounded bg-white/[0.04] px-1.5 py-0.5 font-mono text-[9px] text-muted-foreground">
              {layer.id}
            </span>
          )}
          <button
            onClick={toggleInspector}
            className="rounded p-1 text-muted-foreground/60 transition hover:bg-white/[0.06] hover:text-foreground"
            title="隐藏检查器"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>

      {/* 内容 */}
      <div className="flex-1 overflow-y-auto">
        {layer ? (
          <LayerInspector layer={layer} expandedGroups={expandedGroups} toggleGroup={toggleGroup} />
        ) : (
          <NetworkOverview isPlaying={isPlaying} currentStep={currentStep} />
        )}
      </div>

      {/* 底部指标 */}
      <div className="flex border-t border-white/[0.06] bg-[#0a0c14] px-3 py-2">
        <MetricItem label="Layers" value={architecture.layers.length.toString()} />
        <MetricItem label="Total Params" value={formatNumber(architecture.layers.reduce((sum, l) => sum + l.params, 0))} />
      </div>
    </div>
  );
}

/* ---------- 网络概览（未选择层时） ---------- */
function NetworkOverview({ isPlaying, currentStep }: { isPlaying: boolean; currentStep: number }) {
  const { architecture } = useWorkbench();
  const totalParams = architecture.layers.reduce((sum, l) => sum + l.params, 0);

  return (
    <div className="space-y-2 p-3">
      {/* 模型卡片 */}
      <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
        <h3 className="text-[12px] font-bold text-foreground">
          {architecture.name}
        </h3>
        <p className="mt-0.5 text-[10px] text-muted-foreground">
          Convolutional Neural Network · {architecture.layers.length} layers
        </p>
        <div className="mt-3 grid grid-cols-2 gap-2">
          <div className="rounded bg-white/[0.02] p-2">
            <div className="text-[9px] uppercase text-muted-foreground">Layers</div>
            <div className="mt-0.5 font-mono text-[13px] font-bold text-foreground">{architecture.layers.length}</div>
          </div>
          <div className="rounded bg-white/[0.02] p-2">
            <div className="text-[9px] uppercase text-muted-foreground">Params</div>
            <div className="mt-0.5 font-mono text-[13px] font-bold text-primary">{formatNumber(totalParams)}</div>
          </div>
        </div>
      </div>

      {/* 训练状态 */}
      <div className={`rounded-lg border p-3 ${
        isPlaying
          ? 'border-emerald-400/20 bg-emerald-400/[0.04]'
          : 'border-white/[0.06] bg-white/[0.02]'
      }`}>
        <div className="flex items-center justify-between">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            训练状态
          </span>
          <span className={`flex items-center gap-1.5 text-[10px] ${isPlaying ? 'text-emerald-400' : 'text-muted-foreground'}`}>
            <span className={`size-1.5 rounded-full ${isPlaying ? 'animate-pulse bg-emerald-400' : 'bg-muted-foreground'}`} />
            {isPlaying ? '运行中' : '已暂停'}
          </span>
        </div>
        <div className="mt-2 font-mono text-[20px] font-bold text-foreground">
          Step {currentStep}
        </div>
      </div>

      {/* 提示 */}
      <div className="mt-2 flex items-start gap-2 rounded-lg border border-white/[0.06] bg-white/[0.02] p-2.5">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="mt-0.5 text-primary">
          <circle cx="12" cy="12" r="10" />
          <path d="M12 16v-4M12 8h.01" />
        </svg>
        <div className="text-[10px] leading-relaxed text-muted-foreground/80">
          点击左侧网络层列表或 3D 可视化中的层以查看详细信息。
          训练时激活层会自动高亮。
        </div>
      </div>

      {/* 类型分布 */}
      <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
        <h4 className="mb-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
          层类型分布
        </h4>
        <div className="space-y-1.5">
          {Object.entries(
            architecture.layers.reduce<Record<string, number>>((acc, l) => {
              acc[l.type] = (acc[l.type] || 0) + 1;
              return acc;
            }, {})
          ).map(([type, count]) => (
            <div key={type} className="flex items-center gap-2">
              <span
                className="size-1.5 rounded-full"
                style={{ backgroundColor: LAYER_COLORS_HEX[type as keyof typeof LAYER_COLORS_HEX] || '#999' }}
              />
              <span className="flex-1 text-[10px] capitalize text-foreground/80">{type}</span>
              <span className="font-mono text-[10px] text-muted-foreground">{count}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

/* ---------- 单个层检查器 ---------- */
function LayerInspector({
  layer,
  expandedGroups,
  toggleGroup,
}: {
  layer: LayerConfig;
  expandedGroups: Set<string>;
  toggleGroup: (name: string) => void;
}) {
  const color = LAYER_COLORS_HEX[layer.type];
  const { isPlaying, activeLayerId, selectedLayerId } = useWorkbench();

  // 模拟激活值（基于当前步骤 + 层类型）
  const activationStats = useMemo(() => {
    const base = activeLayerId === layer.id ? 0.75 : 0.2;
    return {
      mean: (base + Math.random() * 0.1 - 0.05).toFixed(4),
      std: (0.15 + Math.random() * 0.05).toFixed(4),
      min: (base - 0.3 - Math.random() * 0.1).toFixed(4),
      max: (base + 0.3 + Math.random() * 0.1).toFixed(4),
      sparsity: (0.3 + Math.random() * 0.2).toFixed(2),
    };
  }, [layer.id, isPlaying, activeLayerId]);

  return (
    <div className="space-y-2 p-3">
      {/* 层头部 */}
      <div
        className="rounded-lg border p-3"
        style={{
          borderColor: `${color}40`,
          backgroundColor: `${color}08`,
        }}
      >
        <div className="flex items-start gap-2">
          <span
            className="mt-0.5 size-2 rounded-full"
            style={{ backgroundColor: color, boxShadow: `0 0 6px ${color}` }}
          />
          <div className="min-w-0 flex-1">
            <h3 className="truncate text-[13px] font-bold text-foreground">
              {layer.name}
            </h3>
            <p className="mt-0.5 text-[10px] text-muted-foreground">
              {layer.type.toUpperCase()}
              {layer.activation && ` · ${layer.activation}`}
            </p>
          </div>
          <div className="flex flex-col items-end gap-1">
            {activeLayerId === layer.id && isPlaying && (
              <span className="rounded bg-emerald-400/10 px-1.5 py-0.5 font-mono text-[8px] text-emerald-400">
                ACTIVE
              </span>
            )}
            {selectedLayerId === layer.id && (
              <span className="rounded bg-primary/10 px-1.5 py-0.5 font-mono text-[8px] text-primary">
                SELECTED
              </span>
            )}
          </div>
        </div>
      </div>

      {/* 分组：形状信息 */}
      <CollapsibleSection
        title="形状信息"
        groupKey="shape"
        expandedGroups={expandedGroups}
        toggleGroup={toggleGroup}
        defaultExpanded
      >
        <div className="grid grid-cols-2 gap-1.5">
          <InfoCell label="Input" value={formatLayerShape(layer.inputShape)} />
          <InfoCell label="Output" value={formatLayerShape(layer.outputShape)} />
          {layer.kernelSize !== undefined && (
            <InfoCell label="Kernel" value={`${layer.kernelSize}×${layer.kernelSize}`} />
          )}
          <InfoCell label="Nodes" value={layer.nodeCount.toString()} />
        </div>
      </CollapsibleSection>

      {/* 分组：参数信息 */}
      <CollapsibleSection
        title="参数分布"
        groupKey="parameters"
        expandedGroups={expandedGroups}
        toggleGroup={toggleGroup}
        defaultExpanded
      >
        <div className="grid grid-cols-2 gap-1.5">
          <InfoCell label="Parameters" value={formatNumber(layer.params)} highlight />
          {/* fe9修复：参数为0时显示"—"而非"0.0 KB" */}
          <InfoCell label="Memory" value={layer.params > 0 ? `${(layer.params * 4 / 1024).toFixed(1)} KB` : '—'} />
        </div>

        {/* 参数条 */}
        <div className="mt-2">
          <div className="mb-1 flex items-center justify-between text-[9px] text-muted-foreground">
            <span>Params Size</span>
          </div>
          <div className="h-1.5 rounded-full bg-white/[0.04] overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{
                width: `${Math.min(100, (layer.params / 200000) * 100)}%`,
                backgroundColor: color,
              }}
            />
          </div>
        </div>
      </CollapsibleSection>

      {/* 分组：激活统计 */}
      <CollapsibleSection
        title="激活统计"
        groupKey="activation"
        expandedGroups={expandedGroups}
        toggleGroup={toggleGroup}
      >
        <div className="grid grid-cols-2 gap-1.5">
          <InfoCell label="Mean" value={activationStats.mean} />
          <InfoCell label="Std Dev" value={activationStats.std} />
          <InfoCell label="Min" value={activationStats.min} />
          <InfoCell label="Max" value={activationStats.max} />
          <InfoCell label="Sparsity" value={activationStats.sparsity} />
        </div>

        {/* 激活柱状图（可视化激活分布） */}
        <div className="mt-3">
          <div className="mb-1 text-[9px] uppercase text-muted-foreground">
            Activation Distribution
          </div>
          <div className="flex h-10 items-end gap-0.5">
            {Array.from({ length: 16 }).map((_, i) => {
              const height = 15 + Math.abs(Math.sin((i + layer.nodeCount) * 0.8)) * 75;
              const isActive = activeLayerId === layer.id && isPlaying && i % 4 === 0;
              return (
                <div
                  key={i}
                  className="flex-1 rounded-sm transition-all duration-300"
                  style={{
                    height: `${height}%`,
                    backgroundColor: isActive ? color : `${color}60`,
                    opacity: isActive ? 1 : 0.55,
                  }}
                />
              );
            })}
          </div>
        </div>
      </CollapsibleSection>
    </div>
  );
}

/* ---------- 辅助组件：折叠分组 ---------- */
interface CollapsibleProps {
  title: string;
  groupKey: string;
  expandedGroups: Set<string>;
  toggleGroup: (name: string) => void;
  children: React.ReactNode;
  defaultExpanded?: boolean;
}
function CollapsibleSection({
  title,
  groupKey,
  expandedGroups,
  toggleGroup,
  children,
}: CollapsibleProps) {
  const isExpanded = expandedGroups.has(groupKey);
  return (
    <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] overflow-hidden">
      <button
        onClick={() => toggleGroup(groupKey)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left transition-all hover:bg-white/[0.02]"
      >
        <svg
          width="9" height="9" viewBox="0 0 24 24" fill="currentColor"
          style={{
            transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
            transition: 'transform 150ms ease',
          }}
          className="text-muted-foreground/60"
        >
          <path d="M6 3h12l-6 18L6 3z" stroke="currentColor" strokeWidth="1" />
        </svg>
        <span className="flex-1 text-[10px] font-semibold uppercase tracking-wider text-foreground/85">
          {title}
        </span>
      </button>
      {isExpanded && (
        <div className="border-t border-white/[0.04] px-3 py-2.5">{children}</div>
      )}
    </div>
  );
}

/* ---------- 辅助组件：信息单元 ---------- */
function InfoCell({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="rounded bg-white/[0.02] p-2">
      <div className="text-[8px] uppercase text-muted-foreground">{label}</div>
      <div className={`mt-0.5 font-mono text-[11px] font-bold ${highlight ? 'text-primary' : 'text-foreground/90'}`}>
        {value}
      </div>
    </div>
  );
}

/* ---------- 底部指标 ---------- */
function MetricItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex-1 text-center">
      <div className="text-[8px] uppercase text-muted-foreground">{label}</div>
      <div className="font-mono text-[11px] font-bold text-foreground">{value}</div>
    </div>
  );
}

/* ---------- 数字格式化 ---------- */
function formatNumber(n: number): string {
  if (n >= 1_000_000) return (n / 1_000_000).toFixed(2) + 'M';
  if (n >= 1_000) return (n / 1_000).toFixed(1) + 'K';
  return n.toString();
}

export default InspectorPanel;
