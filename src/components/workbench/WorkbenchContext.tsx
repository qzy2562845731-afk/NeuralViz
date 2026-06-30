import {
  createContext,
  useContext,
  useState,
  useCallback,
  useMemo,
  useEffect,
} from 'react';
import { DEFAULT_ARCHITECTURE, type LayerConfig, type ViewMode, type NetworkArchitecture } from '../cnn3d/types';
import type { DatasetData, AblationResultItem } from '../../services/api';

/* ============================================
   WorkbenchContext — 统一状态管理
   管理：选中层 / 悬停层 / 激活层 / 训练状态 / 视图模式 / AI 诊断
   - 新增：CNN可视化选择的实验ID（跨页面持久化）
   - 新增：消融实验结果（跨页面持久化，切页不丢失）
   ============================================ */

// sessionStorage 键名
const STORAGE_KEYS = {
  aiEnabled: 'wb_aiEnabled',
  currentModelId: 'wb_currentModelId',
  currentModelName: 'wb_currentModelName',
  selectedDataset: 'wb_selectedDataset',
  architecture: 'wb_architecture',
  viewMode: 'wb_viewMode',
  currentExperimentId: 'wb_currentExperimentId',
  selectedVisualExperimentId: 'wb_selectedVisualExperimentId',
  ablationResults: 'wb_ablationResults',
  activeAblationGroupName: 'wb_activeAblationGroupName',
  navSidebarCollapsed: 'wb_navSidebarCollapsed',
  showLayerRail: 'wb_showLayerRail',
  showInspector: 'wb_showInspector',
};

// 安全读取 sessionStorage
function loadFromStorage<T>(key: string, fallback: T): T {
  try {
    const raw = sessionStorage.getItem(key);
    if (raw === null) return fallback;
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

// 安全写入 sessionStorage
function saveToStorage(key: string, value: unknown): void {
  try {
    sessionStorage.setItem(key, JSON.stringify(value));
  } catch {
    // 忽略写入失败（如存储空间不足）
  }
}

export type LayerId = string;

/* ---------- 消融实验持久化结果 ---------- */
export interface SavedAblationGroup {
  groupName: string;
  presetKey: string;
  presetLabel: string;
  datasetId: string;
  results: AblationResultItem[];
  savedAt: number;
  saved: boolean;
}

/* ---------- Context 状态接口 ---------- */
interface WorkbenchState {
  // 层状态
  selectedLayerId: LayerId | null;
  hoveredLayerId: LayerId | null;
  activeLayerId: LayerId | null;
  focusedLayerId: LayerId | null;

  // 模式
  followTraining: boolean;
  inspectMode: 'follow' | 'manual';

  // 训练状态
  currentStep: number;
  isPlaying: boolean;
  speed: number;

  // 视图模式
  viewMode: ViewMode;

  // 真实激活值（来自推理）
  realActivations: Record<string, number[]>;

  // AI 诊断
  aiEnabled: boolean;
  aiTips: AITip[];

  // 网络架构
  architecture: NetworkArchitecture;

  // 层折叠状态
  collapsedGroups: Set<string>;

  // 后端服务状态
  serverStatus: 'idle' | 'connecting' | 'connected' | 'error';
  serverError: string | null;

  // 当前加载的模型信息
  currentModelId: string | null;
  currentModelName: string;

  // 当前选中的数据集
  selectedDataset: DatasetData | null;

  // 当前训练实验ID（启动训练后设置）
  currentExperimentId: string | null;

  // CNN可视化面板选择的实验ID（用户可手动切换查看不同实验）
  selectedVisualExperimentId: string | null;

  // 消融实验持久化结果（按组名存储）
  ablationResults: Record<string, SavedAblationGroup>;

  // 当前活跃的消融实验组名
  activeAblationGroupName: string | null;

  // 面板可见性控制（布局优化）
  navSidebarCollapsed: boolean;
  showLayerRail: boolean;
  showInspector: boolean;
}

/* ---------- AI 诊断提示 ---------- */
export interface AITip {
  id: string;
  type: 'info' | 'warning' | 'success' | 'suggestion';
  title: string;
  description: string;
  layerId?: string;
}

interface WorkbenchActions {
  selectLayer: (layerId: LayerId | null) => void;
  hoverLayer: (layerId: LayerId | null) => void;
  setActiveLayer: (layerId: LayerId | null | ((prev: LayerId | null) => LayerId | null)) => void;
  resumeFollow: () => void;
  setPlaying: (playing: boolean) => void;
  setStep: (step: number | ((prev: number) => number)) => void;
  setSpeed: (speed: number) => void;
  setViewMode: (mode: ViewMode) => void;
  setRealActivations: (activations: Record<string, number[]>) => void;
  toggleAI: () => void;
  addAITip: (tip: Omit<AITip, 'id'>) => void;
  clearAITips: () => void;
  toggleGroupCollapse: (groupName: string) => void;
  loadArchitecture: (architecture: NetworkArchitecture) => void;
  setServerStatus: (status: 'idle' | 'connecting' | 'connected' | 'error', error?: string | null) => void;
  setCurrentModel: (modelId: string | null, modelName: string) => void;
  setSelectedDataset: (dataset: DatasetData | null) => void;
  setCurrentExperimentId: (id: string | null) => void;
  setSelectedVisualExperimentId: (id: string | null) => void;
  setAblationGroup: (group: SavedAblationGroup) => void;
  markAblationSaved: (groupName: string) => void;
  clearAblationGroup: (groupName: string) => void;
  setActiveAblationGroupName: (name: string | null) => void;
  setNavSidebarCollapsed: (collapsed: boolean) => void;
  toggleLayerRail: () => void;
  toggleInspector: () => void;
  setShowLayerRail: (show: boolean) => void;
  setShowInspector: (show: boolean) => void;
}

type WorkbenchContextValue = WorkbenchState & WorkbenchActions;

const WorkbenchCtx = createContext<WorkbenchContextValue | null>(null);

/* ---------- Provider ---------- */
interface WorkbenchProviderProps {
  children: React.ReactNode;
  architecture?: NetworkArchitecture;
}

const DEFAULT_TIPS: AITip[] = [
  {
    id: 'tip-structure',
    type: 'info',
    title: '网络结构健康',
    description: '整体层级分布合理，特征通道逐渐增加以捕获复杂特征。',
  },
];

export function WorkbenchProvider({ children, architecture }: WorkbenchProviderProps) {
  const [selectedLayerId, setSelectedLayerId] = useState<LayerId | null>(null);
  const [hoveredLayerId, setHoveredLayerId] = useState<LayerId | null>(null);
  const [activeLayerId, setActiveLayerIdState] = useState<LayerId | null>(null);
  const [followTraining, setFollowTraining] = useState(true);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState<number>(1);
  const [viewMode, setViewMode] = useState<ViewMode>(() => loadFromStorage(STORAGE_KEYS.viewMode, 'structure' as ViewMode));
  const [realActivations, setRealActivations] = useState<Record<string, number[]>>({});
  const [aiEnabled, setAiEnabled] = useState<boolean>(() => loadFromStorage(STORAGE_KEYS.aiEnabled, true));
  const [aiTips, setAiTips] = useState<AITip[]>(DEFAULT_TIPS);
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(new Set());
  const [currentArchitecture, setCurrentArchitecture] = useState<NetworkArchitecture>(
    () => loadFromStorage(STORAGE_KEYS.architecture, architecture ?? DEFAULT_ARCHITECTURE)
  );
  const [serverStatus, setServerStatusState] = useState<'idle' | 'connecting' | 'connected' | 'error'>('idle');
  const [serverError, setServerError] = useState<string | null>(null);
  const [currentModelId, setCurrentModelId] = useState<string | null>(() => loadFromStorage(STORAGE_KEYS.currentModelId, 'sample_cnn'));
  const [currentModelName, setCurrentModelName] = useState<string>(() => loadFromStorage(STORAGE_KEYS.currentModelName, 'SampleCNN'));
  const [selectedDataset, setSelectedDatasetState] = useState<DatasetData | null>(() => loadFromStorage<DatasetData | null>(STORAGE_KEYS.selectedDataset, null));
  const [currentExperimentId, setCurrentExperimentIdState] = useState<string | null>(() => loadFromStorage<string | null>(STORAGE_KEYS.currentExperimentId, null));
  const [selectedVisualExperimentId, setSelectedVisualExperimentIdState] = useState<string | null>(() => loadFromStorage<string | null>(STORAGE_KEYS.selectedVisualExperimentId, null));
  const [ablationResults, setAblationResults] = useState<Record<string, SavedAblationGroup>>(() => loadFromStorage<Record<string, SavedAblationGroup>>(STORAGE_KEYS.ablationResults, {}));
  const [activeAblationGroupName, setActiveAblationGroupNameState] = useState<string | null>(() => loadFromStorage<string | null>(STORAGE_KEYS.activeAblationGroupName, null));
  const [navSidebarCollapsed, setNavSidebarCollapsedState] = useState<boolean>(() => loadFromStorage<boolean>(STORAGE_KEYS.navSidebarCollapsed, false));
  const [showLayerRail, setShowLayerRailState] = useState<boolean>(() => loadFromStorage<boolean>(STORAGE_KEYS.showLayerRail, true));
  const [showInspector, setShowInspectorState] = useState<boolean>(() => loadFromStorage<boolean>(STORAGE_KEYS.showInspector, true));

  // 状态变化时持久化到 sessionStorage
  useEffect(() => { saveToStorage(STORAGE_KEYS.aiEnabled, aiEnabled); }, [aiEnabled]);
  useEffect(() => { saveToStorage(STORAGE_KEYS.currentModelId, currentModelId); }, [currentModelId]);
  useEffect(() => { saveToStorage(STORAGE_KEYS.currentModelName, currentModelName); }, [currentModelName]);
  useEffect(() => { saveToStorage(STORAGE_KEYS.selectedDataset, selectedDataset); }, [selectedDataset]);
  useEffect(() => { saveToStorage(STORAGE_KEYS.architecture, currentArchitecture); }, [currentArchitecture]);
  useEffect(() => { saveToStorage(STORAGE_KEYS.viewMode, viewMode); }, [viewMode]);
  useEffect(() => { saveToStorage(STORAGE_KEYS.currentExperimentId, currentExperimentId); }, [currentExperimentId]);
  useEffect(() => { saveToStorage(STORAGE_KEYS.selectedVisualExperimentId, selectedVisualExperimentId); }, [selectedVisualExperimentId]);
  useEffect(() => { saveToStorage(STORAGE_KEYS.ablationResults, ablationResults); }, [ablationResults]);
  useEffect(() => { saveToStorage(STORAGE_KEYS.activeAblationGroupName, activeAblationGroupName); }, [activeAblationGroupName]);
  useEffect(() => { saveToStorage(STORAGE_KEYS.navSidebarCollapsed, navSidebarCollapsed); }, [navSidebarCollapsed]);
  useEffect(() => { saveToStorage(STORAGE_KEYS.showLayerRail, showLayerRail); }, [showLayerRail]);
  useEffect(() => { saveToStorage(STORAGE_KEYS.showInspector, showInspector); }, [showInspector]);

  const arch = currentArchitecture;

  // 聚焦层 = 选中（优先） 或 激活
  const focusedLayerId = selectedLayerId ?? activeLayerId;

  // 模式推导
  const inspectMode: 'follow' | 'manual' = followTraining ? 'follow' : 'manual';

  /* ---------- Actions ---------- */
  const selectLayer = useCallback((layerId: LayerId | null) => {
    setSelectedLayerId(layerId);
    if (layerId !== null) {
      setFollowTraining(false);
    }
  }, []);

  const hoverLayer = useCallback((layerId: LayerId | null) => {
    setHoveredLayerId(layerId);
  }, []);

  const setActiveLayer = useCallback((layerIdOrFn: LayerId | null | ((prev: LayerId | null) => LayerId | null)) => {
    setActiveLayerIdState((prev) =>
      typeof layerIdOrFn === 'function' ? layerIdOrFn(prev) : layerIdOrFn
    );
  }, []);

  const setStepImpl = useCallback((stepOrFn: number | ((prev: number) => number)) => {
    setCurrentStep((prev) =>
      typeof stepOrFn === 'function' ? stepOrFn(prev) : stepOrFn
    );
  }, []);

  const resumeFollow = useCallback(() => {
    setSelectedLayerId(null);
    setFollowTraining(true);
  }, []);

  const toggleAI = useCallback(() => {
    setAiEnabled((prev) => !prev);
  }, []);

  const addAITip = useCallback((tip: Omit<AITip, 'id'>) => {
    const id = `${tip.type}-${tip.title}-${Date.now()}`;
    setAiTips((prev) => [...prev, { id, ...tip }]);
  }, []);

  const clearAITips = useCallback(() => {
    setAiTips([]);
  }, []);

  const toggleGroupCollapse = useCallback((groupName: string) => {
    setCollapsedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(groupName)) next.delete(groupName);
      else next.add(groupName);
      return next;
    });
  }, []);

  const loadArchitecture = useCallback((newArchitecture: NetworkArchitecture) => {
    setCurrentArchitecture(newArchitecture);
    setSelectedLayerId(null);
    setActiveLayerIdState(null);
    setCurrentStep(0);
    setIsPlaying(false);
    setRealActivations({});
  }, []);

  const setServerStatus = useCallback((status: 'idle' | 'connecting' | 'connected' | 'error', error: string | null = null) => {
    setServerStatusState(status);
    setServerError(error);
  }, []);

  const setCurrentModel = useCallback((modelId: string | null, modelName: string) => {
    setCurrentModelId(modelId);
    setCurrentModelName(modelName);
  }, []);

  const setSelectedDataset = useCallback((dataset: DatasetData | null) => {
    setSelectedDatasetState(dataset);
  }, []);

  const setCurrentExperimentId = useCallback((id: string | null) => {
    setCurrentExperimentIdState(id);
    // 同时设置为可视化选择的实验（如果还没有选择）
    setSelectedVisualExperimentIdState((prev) => prev ?? id);
  }, []);

  const setSelectedVisualExperimentId = useCallback((id: string | null) => {
    setSelectedVisualExperimentIdState(id);
  }, []);

  const setAblationGroup = useCallback((group: SavedAblationGroup) => {
    setAblationResults((prev) => ({
      ...prev,
      [group.groupName]: group,
    }));
  }, []);

  const markAblationSaved = useCallback((groupName: string) => {
    setAblationResults((prev) => {
      const group = prev[groupName];
      if (!group) return prev;
      return {
        ...prev,
        [groupName]: { ...group, saved: true },
      };
    });
  }, []);

  const clearAblationGroup = useCallback((groupName: string) => {
    setAblationResults((prev) => {
      const next = { ...prev };
      delete next[groupName];
      return next;
    });
  }, []);

  const setActiveAblationGroupName = useCallback((name: string | null) => {
    setActiveAblationGroupNameState(name);
  }, []);

  const setNavSidebarCollapsed = useCallback((collapsed: boolean) => {
    setNavSidebarCollapsedState(collapsed);
  }, []);

  const setShowLayerRail = useCallback((show: boolean) => {
    setShowLayerRailState(show);
  }, []);

  const setShowInspector = useCallback((show: boolean) => {
    setShowInspectorState(show);
  }, []);

  const toggleLayerRail = useCallback(() => {
    setShowLayerRailState((prev) => !prev);
  }, []);

  const toggleInspector = useCallback(() => {
    setShowInspectorState((prev) => !prev);
  }, []);

  const value = useMemo<WorkbenchContextValue>(
    () => ({
      selectedLayerId,
      hoveredLayerId,
      activeLayerId,
      focusedLayerId,
      followTraining,
      inspectMode,
      currentStep,
      isPlaying,
      speed,
      viewMode,
      realActivations,
      aiEnabled,
      aiTips,
      architecture: arch,
      collapsedGroups,
      serverStatus,
      serverError,
      currentModelId,
      currentModelName,
      selectedDataset,
      currentExperimentId,
      selectedVisualExperimentId,
      ablationResults,
      activeAblationGroupName,
      navSidebarCollapsed,
      showLayerRail,
      showInspector,
      selectLayer,
      hoverLayer,
      setActiveLayer,
      resumeFollow,
      setPlaying: setIsPlaying,
      setStep: setStepImpl,
      setSpeed,
      setViewMode,
      setRealActivations,
      toggleAI,
      addAITip,
      clearAITips,
      toggleGroupCollapse,
      loadArchitecture,
      setServerStatus,
      setCurrentModel,
      setSelectedDataset,
      setCurrentExperimentId,
      setSelectedVisualExperimentId,
      setAblationGroup,
      markAblationSaved,
      clearAblationGroup,
      setActiveAblationGroupName,
      setNavSidebarCollapsed,
      toggleLayerRail,
      toggleInspector,
      setShowLayerRail,
      setShowInspector,
    }),
    [
      selectedLayerId, hoveredLayerId, activeLayerId, focusedLayerId,
      followTraining, inspectMode, currentStep, isPlaying, speed,
      viewMode, realActivations, aiEnabled, aiTips, arch, collapsedGroups,
      serverStatus, serverError, currentModelId, currentModelName, selectedDataset,
      currentExperimentId, selectedVisualExperimentId, ablationResults, activeAblationGroupName,
      navSidebarCollapsed, showLayerRail, showInspector,
      selectLayer, hoverLayer, setActiveLayer, resumeFollow,
      setSpeed, setViewMode, setRealActivations, toggleAI, addAITip, clearAITips, toggleGroupCollapse,
      loadArchitecture, setServerStatus, setCurrentModel, setSelectedDataset,
      setCurrentExperimentId, setSelectedVisualExperimentId, setAblationGroup, markAblationSaved, clearAblationGroup, setActiveAblationGroupName,
      setNavSidebarCollapsed, toggleLayerRail, toggleInspector, setShowLayerRail, setShowInspector,
    ]
  );

  return <WorkbenchCtx.Provider value={value}>{children}</WorkbenchCtx.Provider>;
}

/* ---------- Hook ---------- */
export function useWorkbench(): WorkbenchContextValue {
  const ctx = useContext(WorkbenchCtx);
  if (!ctx) throw new Error('useWorkbench must be used within WorkbenchProvider');
  return ctx;
}

/* ---------- 辅助：从架构推导分组信息 ---------- */
export function buildLayerGroups(architecture: NetworkArchitecture): Array<{
  name: string; layers: LayerConfig[]; category: string;
}> {
  const groupsMap = new Map<string, LayerConfig[]>();
  architecture.layers.forEach((layer) => {
    const groupKey = layer.group ?? (
      layer.type === 'input' ? '输入层'
        : layer.type === 'output' ? '输出层'
          : layer.type === 'conv' || layer.type === 'pool' ? '卷积池化'
            : layer.type === 'fc' ? '全连接层'
              : '其他层'
    );
    if (!groupsMap.has(groupKey)) groupsMap.set(groupKey, []);
    groupsMap.get(groupKey)!.push(layer);
  });

  return Array.from(groupsMap.entries()).map(([name, layers]) => ({
    name,
    layers,
    category: layers[0].type,
  }));
}
