import type { LayerStats } from '../LayerInspector';

interface WorkbenchSidebarProps {
  currentModel: string;
  selectedLayer: { layerId: string; layerInfo: LayerStats } | null;
  onLayerSelect: (selected: { layerId: string; layerInfo: LayerStats } | null) => void;
}

export function WorkbenchSidebar({ currentModel, selectedLayer, onLayerSelect }: WorkbenchSidebarProps) {
  const layers = [
    { id: 'input', name: 'Input', type: 'input' },
    { id: 'conv1', name: 'Conv1', type: 'conv' },
    { id: 'conv2', name: 'Conv2', type: 'conv' },
    { id: 'pool1', name: 'Pool1', type: 'pool' },
    { id: 'fc1', name: 'FC1', type: 'fc' },
    { id: 'output', name: 'Output', type: 'output' },
  ];

  return (
    <div className="flex h-full w-64 flex-col border-r border-white/10 bg-[#0f1119]">
      {/* 标题 - 固定高度 */}
      <div className="flex-shrink-0 border-b border-white/10 px-4 py-3">
        <h3 className="text-sm font-semibold">Network Layers</h3>
        <p className="mt-1 text-xs text-gray-500">{currentModel.toUpperCase()} Architecture</p>
      </div>

      {/* 层列表 - 可滚动 */}
      <div className="min-h-0 flex-1 overflow-y-auto p-2">
        {layers.map((layer, index) => {
          const isSelected = selectedLayer?.layerId === layer.id;
          return (
            <div
              key={layer.id}
              onClick={() => onLayerSelect({
                layerId: layer.id,
                layerInfo: {
                  name: layer.name,
                  type: layer.type as 'input' | 'conv' | 'fc' | 'output',
                  activationMean: 0,
                  activationMax: 0,
                  activationMin: 0,
                  sparsity: 0,
                  weightNorm: 0,
                  gradientNorm: 0,
                  nodeCount: 32,
                },
              })}
              className={`mb-1 cursor-pointer rounded-lg px-3 py-2 text-sm transition ${
                isSelected
                  ? 'bg-green-500/20 text-green-400'
                  : 'hover:bg-white/5'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="font-medium">{layer.name}</span>
                <span className="text-xs text-gray-500">#{index}</span>
              </div>
              <div className="mt-1 text-xs text-gray-500">{layer.type}</div>
            </div>
          );
        })}
      </div>

      {/* 底部信息 - 固定高度 */}
      <div className="flex-shrink-0 border-t border-white/10 px-4 py-3">
        <div className="text-xs text-gray-500">
          <div>Total Layers: {layers.length}</div>
          <div className="mt-1">Trainable Params: 13.2K</div>
        </div>
      </div>
    </div>
  );
}
