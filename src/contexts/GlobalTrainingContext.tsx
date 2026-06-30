import { createContext, useContext, useState, useEffect, useCallback, useMemo, useRef } from 'react';
import type { TrainingStepData } from '../types/training';
import { useRealTraining } from '../hooks/useRealTraining';

/* ============================================
   GlobalTrainingContext — 全局训练状态管理
   - 在App层提供，跨页面持久化
   - 仅支持后端真实训练引擎
   - 训练完成后支持历史数据回放
   - 供Workbench和Visualization共享
   ============================================ */

// 模块级变量用于跨页面持久化（不受React重新渲染影响）
let globalSessionId = 0;
let globalIsRealTraining = false;

// 回放间隔（毫秒）
const REPLAY_INTERVAL_MS = 600;

// 终态集合
const TERMINAL_STATUSES = new Set(['completed', 'failed', 'stopped', 'idle']);

interface GlobalTrainingState {
  // 训练数据历史
  data: TrainingStepData[];

  // 当前步骤
  currentStep: number;

  // 回放播放状态
  isPlaying: boolean;

  // 速度
  speed: number;

  // 总步骤数（0-based索引的最大值）
  maxStep: number;

  // 用于UI显示的总步数（1-based）
  displayTotalSteps: number;

  // 当前数据点
  currentData: TrainingStepData | null;

  // 训练会话ID（每次新训练递增，用于隔离不同轮次数据）
  sessionId: number;

  // 是否有活跃训练（用于判断是否显示空状态）
  hasActiveTraining: boolean;

  // ===== 真实训练模式状态 =====
  // 是否使用后端真实训练引擎
  isRealTraining: boolean;
  // 后端训练状态
  backendStatus: string;
  // 当前 epoch
  currentEpoch: number;
  // 总 epoch
  totalEpochs: number;
  // 训练日志
  trainingLogs: string[];
  // 训练错误信息
  trainingError: string | null;
  // 已运行秒数
  elapsedSeconds: number;
}

interface GlobalTrainingActions {
  // 回放控制（仅用于训练完成后的历史数据回放）
  play: () => void;
  pause: () => void;
  togglePlay: () => void;

  // 跳转步骤
  goToStep: (step: number) => void;

  // 速度控制
  setSpeed: (speed: number) => void;

  // 批量设置训练数据（供真实训练和历史恢复使用）
  setTrainingData: (data: TrainingStepData[]) => void;

  // 重置
  reset: () => void;

  // ===== 真实训练操作 =====
  // 启动后端真实训练
  startRealTraining: (
    experimentId: string,
    datasetId: string,
    hyperparams?: Record<string, any>,
    modelConfig?: Record<string, any>
  ) => Promise<boolean>;
  // 停止后端真实训练
  stopRealTraining: () => Promise<boolean>;
  // 重置真实训练状态
  resetRealTraining: () => void;
  // 拉取全量指标（用于页面恢复）
  fetchTrainingMetrics: (experimentId: string) => Promise<TrainingStepData[]>;
}

type GlobalTrainingContextValue = GlobalTrainingState & GlobalTrainingActions;

const GlobalTrainingCtx = createContext<GlobalTrainingContextValue | null>(null);

/* ---------- Hook ---------- */
export function useGlobalTraining(): GlobalTrainingContextValue {
  const ctx = useContext(GlobalTrainingCtx);
  if (!ctx) throw new Error('useGlobalTraining must be used within GlobalTrainingProvider');
  return ctx;
}

/* ---------- Provider ---------- */
interface GlobalTrainingProviderProps {
  children: React.ReactNode;
}

export function GlobalTrainingProvider({ children }: GlobalTrainingProviderProps) {
  const [data, setData] = useState<TrainingStepData[]>([]);
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeedState] = useState(1);
  const [sessionId, setSessionId] = useState(0);
  const [hasActiveTraining, setHasActiveTraining] = useState(false);

  const replayIntervalRef = useRef<number | null>(null);

  // ===== 真实训练模式对接 =====
  // 当后端推送全量指标时，替换整个 data 数组
  const handleRealTrainingData = useCallback((newData: TrainingStepData[]) => {
    setData(newData);
    if (newData.length > 0) {
      setCurrentStep(newData.length - 1);
    }
  }, []);

  // 当后端推送单条新指标时，追加到 data
  const handleRealTrainingStep = useCallback((step: TrainingStepData) => {
    setData(prev => {
      // 避免重复
      if (prev.length > 0 && prev[prev.length - 1].step >= step.step) {
        return prev;
      }
      return [...prev, step];
    });
    setCurrentStep(prev => Math.max(prev, step.step));
  }, []);

  // 状态变化回调：终态时停止播放并重置训练状态
  const handleStatusChange = useCallback((status: string) => {
    if (TERMINAL_STATUSES.has(status)) {
      setIsPlaying(false);
      setHasActiveTraining(false);
      globalIsRealTraining = false;
    }
  }, []);

  const realTraining = useRealTraining(
    handleRealTrainingData,
    handleRealTrainingStep,
    handleStatusChange
  );

  const isRealTraining = realTraining.isRealTraining || globalIsRealTraining;

  // 总步数：真实训练时使用后端返回的 totalEpochs
  const displayTotalSteps = realTraining.totalEpochs > 0
    ? realTraining.totalEpochs
    : data.length;
  const maxStep = Math.max(0, displayTotalSteps - 1);
  const currentData = data[Math.min(currentStep, data.length - 1)] ?? null;

  // ===== 回放定时器（仅用于训练完成后的历史数据回放）=====
  useEffect(() => {
    // 训练进行中或未播放时，不运行回放定时器
    if (isRealTraining || !isPlaying) {
      if (replayIntervalRef.current !== null) {
        window.clearInterval(replayIntervalRef.current);
        replayIntervalRef.current = null;
      }
      return;
    }

    const intervalMs = REPLAY_INTERVAL_MS / speed;

    replayIntervalRef.current = window.setInterval(() => {
      setCurrentStep(prev => {
        if (prev >= maxStep) {
          setIsPlaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, intervalMs);

    return () => {
      if (replayIntervalRef.current !== null) {
        window.clearInterval(replayIntervalRef.current);
        replayIntervalRef.current = null;
      }
    };
  }, [isPlaying, speed, maxStep, isRealTraining]);

  // 清理
  useEffect(() => {
    return () => {
      if (replayIntervalRef.current !== null) {
        window.clearInterval(replayIntervalRef.current);
      }
    };
  }, []);

  // ===== Actions =====

  // 回放播放：仅用于训练完成后的历史数据回放
  const play = useCallback(() => {
    if (data.length === 0) return;
    // 如果已到末尾，从头开始
    if (currentStep >= maxStep) {
      setCurrentStep(0);
    }
    setIsPlaying(true);
  }, [data.length, currentStep, maxStep]);

  const pause = useCallback(() => {
    setIsPlaying(false);
  }, []);

  const togglePlay = useCallback(() => {
    if (isPlaying) {
      pause();
    } else {
      play();
    }
  }, [isPlaying, play, pause]);

  const goToStep = useCallback((step: number) => {
    const targetStep = Math.max(0, Math.min(step, maxStep));
    setCurrentStep(targetStep);
  }, [maxStep]);

  const setSpeed = useCallback((newSpeed: number) => {
    setSpeedState(newSpeed);
  }, []);

  // 批量设置训练数据（供真实训练和历史恢复使用）
  const setTrainingData = useCallback((newData: TrainingStepData[]) => {
    setData(newData);
    if (newData.length > 0) {
      setCurrentStep(newData.length - 1);
    }
  }, []);

  const reset = useCallback(() => {
    setIsPlaying(false);
    globalIsRealTraining = false;
    globalSessionId++;
    setSessionId(globalSessionId);
    setData([]);
    setCurrentStep(0);
    setHasActiveTraining(false);
    realTraining.resetRealTraining();
  }, [realTraining]);

  // ===== 真实训练操作 =====
  const startRealTraining = useCallback(async (
    expId: string,
    datasetId: string,
    hyperparams?: Record<string, any>,
    modelConfig?: Record<string, any>
  ): Promise<boolean> => {
    // 清空旧数据，开始新会话
    globalSessionId++;
    globalIsRealTraining = true;
    setSessionId(globalSessionId);
    setData([]);
    setCurrentStep(0);
    setHasActiveTraining(true);
    setIsPlaying(false);

    const success = await realTraining.startRealTraining(expId, datasetId, hyperparams, modelConfig);
    if (!success) {
      globalIsRealTraining = false;
      setHasActiveTraining(false);
    }
    return success;
  }, [realTraining]);

  const stopRealTraining = useCallback(async (): Promise<boolean> => {
    const success = await realTraining.stopRealTraining();
    globalIsRealTraining = false;
    setIsPlaying(false);
    return success;
  }, [realTraining]);

  const resetRealTraining = useCallback(() => {
    globalIsRealTraining = false;
    realTraining.resetRealTraining();
  }, [realTraining]);

  const fetchTrainingMetrics = useCallback(async (expId: string): Promise<TrainingStepData[]> => {
    return realTraining.fetchMetrics(expId);
  }, [realTraining]);

  const value = useMemo<GlobalTrainingContextValue>(() => ({
    data,
    currentStep,
    isPlaying,
    speed,
    maxStep,
    displayTotalSteps,
    currentData,
    sessionId,
    hasActiveTraining,
    isRealTraining,
    backendStatus: realTraining.backendStatus,
    currentEpoch: realTraining.currentEpoch,
    totalEpochs: realTraining.totalEpochs,
    trainingLogs: realTraining.logs,
    trainingError: realTraining.error,
    elapsedSeconds: realTraining.elapsedSeconds,
    // Actions
    play,
    pause,
    togglePlay,
    goToStep,
    setSpeed,
    setTrainingData,
    reset,
    startRealTraining,
    stopRealTraining,
    resetRealTraining,
    fetchTrainingMetrics,
  }), [data, currentStep, isPlaying, speed, maxStep, displayTotalSteps, currentData, sessionId, hasActiveTraining, isRealTraining, realTraining, play, pause, togglePlay, goToStep, setSpeed, setTrainingData, reset, startRealTraining, stopRealTraining, resetRealTraining, fetchTrainingMetrics]);

  return <GlobalTrainingCtx.Provider value={value}>{children}</GlobalTrainingCtx.Provider>;
}
