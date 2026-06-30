import { useMemo } from 'react';
import type { FeatureMaps } from '../types/training';

interface FeatureMapViewerProps {
  featureMaps: FeatureMaps | undefined;
  step: number;
}

interface HeatmapCellProps {
  value: number;
  size: number;
  x: number;
  y: number;
}

function HeatmapCell({ value, size, x, y }: HeatmapCellProps) {
  const intensity = Math.max(0, Math.min(1, value));
  const r = Math.round(255 * intensity);
  const g = Math.round(100 * (1 - intensity));
  const b = Math.round(255 * (1 - intensity));
  
  return (
    <rect
      x={x * size}
      y={y * size}
      width={size}
      height={size}
      fill={`rgb(${r}, ${g}, ${b})`}
      stroke="rgba(255,255,255,0.08)"
      strokeWidth={0.5}
      rx={1}
    >
      <title>{value.toFixed(3)}</title>
    </rect>
  );
}

interface FeatureMapGridProps {
  data: number[][];
  channelIndex: number;
  cellSize: number;
}

function FeatureMapGrid({ data, channelIndex, cellSize }: FeatureMapGridProps) {
  const rows = data.length;
  const cols = data[0]?.length ?? 0;
  const avgValue = data.flat().reduce((a, b) => a + b, 0) / data.flat().length;
  
  return (
    <div className="feature-map-item">
      <div className="feature-map-header">
        <span className="channel-badge">Ch {channelIndex}</span>
        <span className="channel-avg">avg: {avgValue.toFixed(2)}</span>
      </div>
      <svg
        width={cols * cellSize}
        height={rows * cellSize}
        className="heatmap-svg"
        viewBox={`0 0 ${cols * cellSize} ${rows * cellSize}`}
      >
        {data.map((row, rowIdx) =>
          row.map((value, colIdx) => (
            <HeatmapCell
              key={`${rowIdx}-${colIdx}`}
              value={value}
              size={cellSize}
              x={colIdx}
              y={rowIdx}
            />
          ))
        )}
      </svg>
    </div>
  );
}

export function FeatureMapViewer({ featureMaps, step }: FeatureMapViewerProps) {
  const conv1Maps = useMemo(() => {
    return featureMaps?.conv1 ?? [];
  }, [featureMaps]);

  if (!featureMaps || conv1Maps.length === 0) {
    return (
      <div className="feature-map-viewer card">
        <div className="card-header">
          <h2 className="card-title">Feature Maps</h2>
        </div>
        <div className="empty-state">
          <span>等待数据加载...</span>
        </div>
      </div>
    );
  }

  const CELL_SIZE = 18;

  return (
    <div className="feature-map-viewer card">
      <div className="card-header">
        <h2 className="card-title">Feature Maps</h2>
        <span className="step-badge">Step {step}</span>
      </div>
      <div className="feature-map-description">
        {/* fe11修复：移除 Mock 标签，保留描述文案 */}
        <span className="mock-hint">模型学习到的局部特征响应</span>
      </div>
      <div className="heatmap-grid">
        {conv1Maps.map((map, idx) => (
          <FeatureMapGrid
            key={idx}
            data={map}
            channelIndex={idx}
            cellSize={CELL_SIZE}
          />
        ))}
      </div>
    </div>
  );
}
