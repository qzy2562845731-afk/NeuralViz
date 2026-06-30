import { useState, useMemo, useRef, useEffect } from 'react';
import { useWorkbench, buildLayerGroups } from './WorkbenchContext';
import { LAYER_COLORS_HEX, formatLayerShape, type LayerConfig } from '../cnn3d/types';

/* ============================================
   LayerRail — 左侧树状层列表
   - 分组折叠 + 折叠/展开全部
   - 搜索定位 + 高亮匹配
   - 高亮当前选中/激活层
   ============================================ */

interface LayerRailProps {
  width?: number;
}

export function LayerRail({ width = 260 }: LayerRailProps) {
  const {
    selectedLayerId,
    activeLayerId,
    isPlaying,
    selectLayer,
    hoverLayer,
    resumeFollow,
    followTraining,
    toggleGroupCollapse,
    collapsedGroups,
    architecture,
    toggleLayerRail,
  } = useWorkbench();

  const [searchQuery, setSearchQuery] = useState('');
  const listRef = useRef<HTMLDivElement>(null);

  // 构建分组
  const groups = useMemo(() => buildLayerGroups(architecture), [architecture]);

  // 过滤层（搜索）
  const filteredGroups = useMemo(() => {
    if (!searchQuery.trim()) return groups;
    const query = searchQuery.toLowerCase();
    return groups
      .map((g) => ({
        ...g,
        layers: g.layers.filter((l) =>
          l.name.toLowerCase().includes(query) ||
          l.type.toLowerCase().includes(query) ||
          l.id.toLowerCase().includes(query)
        ),
      }))
      .filter((g) => g.layers.length > 0);
  }, [groups, searchQuery]);

  // 折叠/展开全部
  const allCollapsed = groups.length > 0 && groups.every(g => collapsedGroups.has(g.name));
  const toggleAllGroups = () => {
    groups.forEach(g => {
      const isCollapsed = collapsedGroups.has(g.name);
      if (allCollapsed && isCollapsed) {
        toggleGroupCollapse(g.name);
      } else if (!allCollapsed && !isCollapsed) {
        toggleGroupCollapse(g.name);
      }
    });
  };

  // 搜索时自动滚动到第一个匹配项
  useEffect(() => {
    if (searchQuery.trim() && filteredGroups.length > 0 && listRef.current) {
      const firstMatch = listRef.current.querySelector('[data-match="true"]');
      firstMatch?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, [searchQuery, filteredGroups]);

  return (
    <div
      className="flex h-full flex-col border-r border-white/[0.06] bg-[#0c0e17]"
      style={{ width: `${width}px` }}
    >
      {/* 顶部标题 */}
      <div className="flex items-center justify-between border-b border-white/[0.06] px-4 py-3">
        <div className="flex items-center gap-2">
          <div className="flex h-5 w-5 items-center justify-center rounded-md bg-primary/15 text-primary">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <rect x="3" y="3" width="7" height="7" />
              <rect x="14" y="3" width="7" height="7" />
              <rect x="3" y="14" width="7" height="7" />
              <rect x="14" y="14" width="7" height="7" />
            </svg>
          </div>
          <span className="text-[11px] font-semibold uppercase tracking-wider text-foreground/85">
            Network Layers
          </span>
        </div>
        <div className="flex items-center gap-1">
          <span className="rounded bg-white/[0.04] px-1.5 py-0.5 font-mono text-[9px] text-muted-foreground">
            {architecture.layers.length}
          </span>
          {/* 折叠/展开全部 */}
          <button
            onClick={toggleAllGroups}
            className="rounded p-1 text-muted-foreground transition hover:bg-white/[0.04] hover:text-foreground"
            title={allCollapsed ? '展开全部' : '折叠全部'}
          >
            {allCollapsed ? (
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 3h18M3 9h18M3 15h18M3 21h18" />
              </svg>
            ) : (
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 3v18M9 3v18M15 3v18M21 3v18" />
              </svg>
            )}
          </button>
          {/* 关闭面板按钮 */}
          <button
            onClick={toggleLayerRail}
            className="rounded p-1 text-muted-foreground/60 transition hover:bg-white/[0.06] hover:text-foreground"
            title="隐藏层列表"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 6L6 18M6 6l12 12" />
            </svg>
          </button>
        </div>
      </div>

      {/* 搜索框 */}
      <div className="px-3 py-2">
        <div className="relative flex items-center gap-2 rounded-md border border-white/[0.06] bg-white/[0.02] px-2.5 py-1.5 transition-all focus-within:border-primary/30 focus-within:bg-primary/5">
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-muted-foreground">
            <circle cx="11" cy="11" r="7" />
            <path d="m21 21-4.3-4.3" />
          </svg>
          <input
            type="text"
            placeholder="搜索层..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-transparent font-mono text-[11px] text-foreground placeholder:text-muted-foreground/60 focus:outline-none"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery('')}
              className="text-muted-foreground hover:text-foreground"
            >
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <path d="M18 6 6 18M6 6l12 12" />
              </svg>
            </button>
          )}
        </div>
      </div>

      {/* 模式指示 */}
      <div className="px-3 pb-2">
        {followTraining ? (
          <div className="flex items-center justify-between rounded-md border border-emerald-400/20 bg-emerald-400/[0.05] px-2.5 py-1.5">
            <div className="flex items-center gap-1.5">
              <span className="size-1.5 animate-pulse rounded-full bg-emerald-400" />
              <span className="font-mono text-[10px] text-emerald-400">
                Following Training
              </span>
            </div>
          </div>
        ) : (
          <button
            onClick={resumeFollow}
            className="flex w-full items-center justify-between rounded-md border border-primary/25 bg-primary/[0.06] px-2.5 py-1.5 transition-all hover:border-primary/40 hover:bg-primary/10"
          >
            <div className="flex items-center gap-1.5">
              <span className="size-1.5 rounded-full bg-primary" />
              <span className="font-mono text-[10px] text-primary">
                Manual · {selectedLayerId ? architecture.layers.find(l => l.id === selectedLayerId)?.name : 'Selecting'}
              </span>
            </div>
            <span className="font-mono text-[9px] text-primary/80 hover:text-primary">
              Resume →
            </span>
          </button>
        )}
      </div>

      {/* 分组层列表 */}
      <div ref={listRef} className="flex-1 overflow-y-auto px-2 pb-3">
        {filteredGroups.map((group) => {
          const isCollapsed = collapsedGroups.has(group.name) && !searchQuery.trim();
          return (
            <div key={group.name} className="mb-1.5">
              {/* 分组标题 */}
              <button
                onClick={() => toggleGroupCollapse(group.name)}
                className="flex w-full items-center gap-1.5 rounded px-2 py-1 text-left transition-all hover:bg-white/[0.03]"
              >
                <svg
                  width="10" height="10" viewBox="0 0 24 24"
                  fill="currentColor"
                  style={{
                    transform: isCollapsed ? 'rotate(0deg)' : 'rotate(90deg)',
                    transition: 'transform 150ms ease',
                  }}
                  className="text-muted-foreground/70"
                >
                  <path d="M6 3h12l-6 18L6 3z" stroke="currentColor" strokeWidth="1" />
                </svg>
                <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                  {group.name}
                </span>
                <span className="ml-auto rounded bg-white/[0.03] px-1.5 py-0.5 font-mono text-[8px] text-muted-foreground/80">
                  {group.layers.length}
                </span>
              </button>

              {/* 层项 */}
              {!isCollapsed && (
                <div className="ml-2 space-y-1 border-l border-white/[0.04] pl-2">
                  {group.layers.map((layer) => (
                    <LayerItem
                      key={layer.id}
                      layer={layer}
                      isSelected={selectedLayerId === layer.id}
                      isActive={activeLayerId === layer.id && isPlaying}
                      onSelect={selectLayer}
                      onHover={hoverLayer}
                      searchQuery={searchQuery}
                    />
                  ))}
                </div>
              )}
            </div>
          );
        })}

        {filteredGroups.length === 0 && (
          <div className="flex flex-col items-center justify-center py-6 text-center">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="mb-2 text-muted-foreground/40">
              <circle cx="11" cy="11" r="7" />
              <path d="m21 21-4.3-4.3" />
            </svg>
            <p className="text-[10px] text-muted-foreground/60">未找到匹配的层</p>
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------- 高亮匹配文本 ---------- */
function HighlightText({ text, query }: { text: string; query: string }) {
  if (!query.trim()) return <>{text}</>;
  const idx = text.toLowerCase().indexOf(query.toLowerCase());
  if (idx === -1) return <>{text}</>;
  return (
    <>
      {text.substring(0, idx)}
      <mark className="rounded bg-amber-400/30 px-0.5 text-amber-300">{text.substring(idx, idx + query.length)}</mark>
      {text.substring(idx + query.length)}
    </>
  );
}

/* ---------- 单个层项 ---------- */
interface LayerItemProps {
  layer: LayerConfig;
  isSelected: boolean;
  isActive: boolean;
  onSelect: (id: string | null) => void;
  onHover: (id: string | null) => void;
  searchQuery: string;
}

function LayerItem({ layer, isSelected, isActive, onSelect, onHover, searchQuery }: LayerItemProps) {
  const color = LAYER_COLORS_HEX[layer.type] || '#666';
  const isMatch = searchQuery.trim() && (
    layer.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
    layer.type.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <button
      data-match={isMatch ? 'true' : undefined}
      onClick={() => onSelect(isSelected ? null : layer.id)}
      onMouseEnter={() => onHover(layer.id)}
      onMouseLeave={() => onHover(null)}
      className={`group flex w-full items-center gap-2 rounded-md border px-2 py-1.5 text-left transition-all duration-150 ${
        isSelected
          ? 'border-primary/35 bg-primary/[0.1] shadow-[0_0_12px_rgba(99,102,241,0.15)]'
          : isActive
            ? 'border-emerald-400/25 bg-emerald-400/[0.06]'
            : isMatch
              ? 'border-amber-400/20 bg-amber-400/[0.05]'
              : 'border-transparent hover:border-white/[0.08] hover:bg-white/[0.03]'
      }`}
    >
      {/* 颜色点 */}
      <span
        className={`size-1.5 flex-shrink-0 rounded-full transition-all ${
          isActive ? 'animate-pulse' : ''
        }`}
        style={{ backgroundColor: color, boxShadow: isSelected || isActive ? `0 0 6px ${color}` : undefined }}
      />

      {/* 层信息 */}
      <div className="flex min-w-0 flex-1 flex-col">
        <span className={`truncate text-[11px] font-semibold ${
          isSelected ? 'text-primary'
            : isActive ? 'text-emerald-400'
              : 'text-foreground/85 group-hover:text-foreground'
        }`}>
          <HighlightText text={layer.name} query={searchQuery} />
        </span>
        <span className="truncate font-mono text-[9px] text-muted-foreground/70">
          {layer.type} · {formatLayerShape(layer.outputShape)}
        </span>
      </div>

      {/* 状态指示 */}
      <div className="flex items-center gap-1">
        {isActive && (
          <span className="rounded bg-emerald-400/10 px-1 py-0.5 font-mono text-[8px] text-emerald-400">
            ACTIVE
          </span>
        )}
        {isSelected && (
          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" className="text-primary">
            <path d="M20 6 9 17l-5-5" />
          </svg>
        )}
      </div>
    </button>
  );
}

export default LayerRail;
