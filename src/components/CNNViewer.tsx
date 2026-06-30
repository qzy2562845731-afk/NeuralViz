import { useMemo } from 'react';
import type { LayerActivations } from '../types/training';
import {
  activationToColor,
  activationToStrokeColor,
  activationToOpacity,
  activationToRadius
} from '../utils/activation';

interface CNNViewerProps {
  activations: LayerActivations | null;
  step: number;
}

interface NodePosition {
  x: number;
  y: number;
  channelId: string;
  activation: number;
  radius: number;
}

interface LayerConfig {
  id: keyof LayerActivations;
  label: string;
  fullLabel: string;
  color: string;
}

interface Edge {
  from: NodePosition;
  to: NodePosition;
  avgActivation: number;
  key: string;
}

interface LayerData {
  id: keyof LayerActivations;
  label: string;
  fullLabel: string;
  color: string;
  x: number;
  positions: NodePosition[];
  avgActivation: number;
}

// ===== 布局常量 =====
// 每个节点垂直空间分配（考虑最大半径 20px）：
//   top text: radius(20) + gap(15) + textHeight(~12) = 47px above center
//   circle diameter: 40px
//   bottom text: radius(20) + gap(15) + textHeight(~12) = 47px below center
//   每个节点槽: 47 + 40 + 47 = 134px, 取 120px（节点间距 20px）
const NODE_SLOT_HEIGHT = 120;
const TEXT_GAP = 15;          // 文字距离圆边缘的固定间距（增加到 15px）
const TOP_HEADER = 60;        // 顶部层标题区域
const BOTTOM_FOOTER = 50;     // 底部统计区域

const LAYER_CONFIGS: LayerConfig[] = [
  { id: 'input', label: 'Input', fullLabel: 'Input Layer', color: '#4285f4' },
  { id: 'conv1', label: 'Conv1', fullLabel: 'Conv1 Layer', color: '#34a853' },
  { id: 'conv2', label: 'Conv2', fullLabel: 'Conv2 Layer', color: '#fbbc04' },
  { id: 'fc', label: 'FC', fullLabel: 'FC Layer', color: '#ea4335' }
];

function getChannelLabel(layerId: keyof LayerActivations, index: number): string {
  const prefix = layerId === 'input' ? 'I' : layerId === 'conv1' ? 'C1' : layerId === 'conv2' ? 'C2' : 'F';
  return `${prefix}${index}`;
}

// ===== 布局计算 =====
interface LayoutResult {
  width: number;
  height: number;
  paddingX: number;
  layerXPositions: number[];
  nodeStartY: number;
  nodeSpacingY: number;
}

function calculateLayout(layerCount: number, maxNodesInLayer: number): LayoutResult {
  const width = 800;
  const paddingX = 60;

  // 水平位置：均匀分布
  const usableWidth = width - paddingX * 2;
  const layerXPositions: number[] = [];
  for (let i = 0; i < layerCount; i++) {
    layerXPositions.push(
      layerCount === 1
        ? width / 2
        : paddingX + (usableWidth / (layerCount - 1)) * i
    );
  }

  // 垂直间距：基于节点数动态计算
  // 每个节点需要 NODE_SLOT_HEIGHT 的空间
  const minSpacingY = NODE_SLOT_HEIGHT;
  const calculatedSpacing = Math.max(minSpacingY, NODE_SLOT_HEIGHT);

  // 动态高度：确保所有节点都能放下
  const nodesHeight = maxNodesInLayer * calculatedSpacing;
  const height = TOP_HEADER + nodesHeight + BOTTOM_FOOTER;

  // 节点起始 Y：在顶部标题下方，垂直居中
  const totalNodesHeight = (maxNodesInLayer - 1) * calculatedSpacing;
  const availableHeight = height - TOP_HEADER - BOTTOM_FOOTER;
  const nodeStartY = TOP_HEADER + (availableHeight - totalNodesHeight) / 2;

  return {
    width,
    height: Math.max(height, 400),
    paddingX,
    layerXPositions,
    nodeStartY,
    nodeSpacingY: calculatedSpacing
  };
}

function calculateLayerNodes(
  values: number[] | undefined,
  x: number,
  layerId: keyof LayerActivations,
  startY: number,
  spacingY: number
): NodePosition[] {
  if (!values || values.length === 0) return [];

  return values.map((activation, idx) => ({
    x,
    y: startY + idx * spacingY,
    channelId: getChannelLabel(layerId, idx),
    activation,
    radius: activationToRadius(activation)
  }));
}

function calculateEdges(layerData: LayerData[]): Edge[] {
  const edges: Edge[] = [];

  for (let i = 0; i < layerData.length - 1; i++) {
    const fromLayer = layerData[i];
    const toLayer = layerData[i + 1];

    const maxConns = Math.min(fromLayer.positions.length, toLayer.positions.length, 5);

    for (let j = 0; j < maxConns; j++) {
      const fromIdx = Math.floor((j / maxConns) * fromLayer.positions.length);
      const toIdx = Math.floor((j / maxConns) * toLayer.positions.length);

      const fromNode = fromLayer.positions[fromIdx];
      const toNode = toLayer.positions[toIdx];

      if (fromNode && toNode) {
        edges.push({
          from: fromNode,
          to: toNode,
          avgActivation: (fromNode.activation + toNode.activation) / 2,
          key: `${fromLayer.id}-${fromIdx}-${toLayer.id}-${toIdx}`
        });
      }
    }
  }

  return edges;
}

// ===== 主组件 =====
export function CNNViewer({ activations, step }: CNNViewerProps) {
  const visibleLayers = useMemo(() => {
    if (!activations) return LAYER_CONFIGS.filter(c => c.id === 'input' || c.id === 'fc');
    const hasConv2 = activations.conv2 && activations.conv2.length > 0;
    if (hasConv2) return LAYER_CONFIGS;
    return LAYER_CONFIGS.filter(c => c.id !== 'conv2');
  }, [activations]);

  const maxNodesInLayer = useMemo(() => {
    if (!activations) return 0;
    return Math.max(
      ...visibleLayers.map(layer => activations[layer.id]?.length ?? 0)
    );
  }, [activations, visibleLayers]);

  const layout = useMemo(() => {
    return calculateLayout(visibleLayers.length, maxNodesInLayer);
  }, [visibleLayers.length, maxNodesInLayer]);

  const layerData = useMemo(() => {
    return visibleLayers.map((layer, idx) => {
      const values = activations?.[layer.id] ?? [];
      const x = layout.layerXPositions[idx];
      const positions = calculateLayerNodes(values, x, layer.id, layout.nodeStartY, layout.nodeSpacingY);
      const avg = values.length > 0
        ? values.reduce((a, b) => a + b, 0) / values.length
        : 0;

      return { ...layer, x, positions, avgActivation: avg };
    });
  }, [visibleLayers, activations, layout]);

  const edges = useMemo(() => calculateEdges(layerData), [layerData]);

  if (!activations) {
    return (
      <div className="cnn-viewer card">
        <div className="card-header">
          <h2 className="card-title">CNN 结构可视化</h2>
        </div>
        <div className="empty-state">
          <span>等待数据加载...</span>
        </div>
      </div>
    );
  }

  // 层标签 Y 位置（顶部标题区中间）
  const layerLabelY = TOP_HEADER / 2;

  return (
    <div className="cnn-viewer card">
      <div className="card-header">
        <h2 className="card-title">CNN 结构可视化</h2>
        <span className="step-badge">Step {step}</span>
      </div>
      <div className="cnn-container">
        <svg
          viewBox={`0 0 ${layout.width} ${layout.height}`}
          className="cnn-svg"
          preserveAspectRatio="xMidYMid meet"
        >
          <defs>
            <filter id="nodeGlow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
              <feMerge>
                <feMergeNode in="coloredBlur"/>
                <feMergeNode in="SourceGraphic"/>
              </feMerge>
            </filter>
          </defs>

          {/* ===== 层标题区域 ===== */}
          <g className="layer-labels">
            {layerData.map((layer) => (
              <g key={`label-${layer.id}`}>
                <text
                  x={layer.x}
                  y={layerLabelY - 6}
                  textAnchor="middle"
                  dominantBaseline="auto"
                  fill={layer.color}
                  fontSize="13"
                  fontWeight="600"
                >
                  {layer.label}
                </text>
                <text
                  x={layer.x}
                  y={layerLabelY + 10}
                  textAnchor="middle"
                  dominantBaseline="auto"
                  fill="#666"
                  fontSize="10"
                >
                  mean: {layer.avgActivation.toFixed(3)}
                </text>
              </g>
            ))}
          </g>

          {/* ===== 边渲染层（在节点下方） ===== */}
          <g className="edges-layer">
            {edges.map((edge) => {
              const midX = (edge.from.x + edge.to.x) / 2;
              const midY = (edge.from.y + edge.to.y) / 2;
              const pathD = `M ${edge.from.x} ${edge.from.y} Q ${midX} ${midY - 15} ${edge.to.x} ${edge.to.y}`;

              return (
                <path
                  key={edge.key}
                  d={pathD}
                  fill="none"
                  stroke={`rgba(100, 140, 200, ${0.08 + edge.avgActivation * 0.25})`}
                  strokeWidth={0.5 + edge.avgActivation * 2}
                  className="connection-line"
                />
              );
            })}
          </g>

          {/* ===== 节点渲染层 ===== */}
          <g className="nodes-layer">
            {layerData.map((layer) => (
              <g key={layer.id} className="layer-group">
                {layer.positions.map((pos) => {
                  const fillColor = activationToColor(pos.activation, layer.id);
                  const strokeColor = activationToStrokeColor(pos.activation, layer.id);
                  const nodeOpacity = activationToOpacity(pos.activation);

                  return (
                    <g
                      key={`${layer.id}-${pos.channelId}`}
                      className="node-group"
                    >
                      <title>{`${pos.channelId}: ${pos.activation.toFixed(3)}`}</title>

                      {/* 通道标签（圆上方，不受滤镜影响） */}
                      <text
                        x={pos.x}
                        y={pos.y - pos.radius - TEXT_GAP - 4}
                        textAnchor="middle"
                        dominantBaseline="auto"
                        fontSize="9"
                        fill="#aaa"
                        fontWeight="500"
                      >
                        {pos.channelId}
                      </text>

                      {/* 节点圆（仅此元素受滤镜和透明度影响） */}
                      <circle
                        cx={pos.x}
                        cy={pos.y}
                        r={pos.radius}
                        fill={fillColor}
                        stroke={strokeColor}
                        strokeWidth={1.5}
                        opacity={nodeOpacity}
                        filter="url(#nodeGlow)"
                        className="node-circle"
                      />

                      {/* 激活值（圆下方，不受滤镜影响） */}
                      <text
                        x={pos.x}
                        y={pos.y + pos.radius + TEXT_GAP + 2}
                        textAnchor="middle"
                        dominantBaseline="hanging"
                        fontSize="9"
                        fill="#888"
                      >
                        {pos.activation.toFixed(3)}
                      </text>
                    </g>
                  );
                })}

                {/* 层底部统计 */}
                <text
                  x={layer.x}
                  y={layout.height - BOTTOM_FOOTER / 2}
                  textAnchor="middle"
                  dominantBaseline="middle"
                  fill="#555"
                  fontSize="10"
                >
                  {layer.positions.length} channels
                </text>
              </g>
            ))}
          </g>
        </svg>
      </div>
    </div>
  );
}
