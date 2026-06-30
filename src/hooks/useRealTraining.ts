/**
 * useRealTraining - 真实训练引擎对接 Hook
 *
 * 功能：
 * 1. 启动/停止后端训练任务
 * 2. 轮询训练状态（1秒1次），终态自动停止
 * 3. 增量拉取训练日志
 * 4. 拉取全量指标并转换为 TrainingStepData 格式
 * 5. 网络异常自动降级提示
 */
import { useState, useCallback, useRef, useEffect } from 'react';
import apiService from '../services/api';
import type { TrainingStatusData, TrainingMetricItem } from '../services/api';
import type { TrainingStepData, LayerActivations, FeatureMaps } from '../types/training';
import { createActivationStats } from '../utils/metrics';

// 终态集合：检测到这些状态后立即停止轮询
// idle: 后端任务不存在（服务重启等），也应停止轮询
const TERMINAL_STATUSES = new Set(['completed', 'failed', 'stopped', 'idle']);

// 轮询间隔（毫秒）
const POLL_INTERVAL_MS = 1000;

export interface RealTrainingState {
  /** 是否正在使用真实训练模式 */
  isRealTraining: boolean;
  /** 后端训练状态：pending/running/completed/failed/stopped/idle */
  backendStatus: string;
  /** 当前 epoch */
  currentEpoch: number;
  /** 总 epoch */
  totalEpochs: number;
  /** 训练日志列表 */
  logs: string[];
  /** 错误信息 */
  error: string | null;
  /** 已运行的秒数 */
  elapsedSeconds: number;
  /** 当前实验ID */
  experimentId: string | null;
}

export interface RealTrainingActions {
  /** 启动真实训练 */
  startRealTraining: (
    experimentId: string,
    datasetId: string,
    hyperparams?: Record<string, any>,
    modelConfig?: Record<string, any>
  ) => Promise<boolean>;
  /** 停止真实训练 */
  stopRealTraining: () => Promise<boolean>;
  /** 重置状态 */
  resetRealTraining: () => void;
  /** 手动拉取全量指标（用于页面恢复） */
  fetchMetrics: (experimentId: string) => Promise<TrainingStepData[]>;
}

/**
 * 将后端指标转换为前端 TrainingStepData 格式
 */
export function convertMetricToStepData(
  metric: TrainingMetricItem,
  index: number
): TrainingStepData {
  // 解析 extra_data 中的进阶指标
  let precision = 0;
  let recall = 0;
  let f1Score = 0;
  let gradientNorm = 0;
  let weightNorm = 0;
  let confusionMatrix: number[][] = [];
  let layerActivations: LayerActivations = { input: [], conv1: [], fc: [] };
  let featureMaps: FeatureMaps = { conv1: [] };
  let predictionDistribution: number[] = [];

  if (metric.extra_data) {
    try {
      const extra = typeof metric.extra_data === 'string'
        ? JSON.parse(metric.extra_data)
        : metric.extra_data;
      precision = extra.precision || 0;
      recall = extra.recall || 0;
      f1Score = extra.f1 || extra.f1_score || 0;
      gradientNorm = extra.gradient_norm || 0;
      weightNorm = extra.weight_norm || 0;
      confusionMatrix = extra.confusion_matrix || [];

      // 解析激活分布
      if (extra.layer_activations) {
        layerActivations = {
          input: extra.layer_activations.input || [],
          conv1: extra.layer_activations.conv1 || [],
          conv2: extra.layer_activations.conv2 || undefined,
          fc: extra.layer_activations.fc || [],
        };
      }

      // 解析特征图
      if (extra.feature_maps && Array.isArray(extra.feature_maps) && extra.feature_maps.length > 0) {
        featureMaps = {
          conv1: extra.feature_maps,
          conv2: undefined,
          fc: undefined,
        };
      }

      // 解析预测分布
      if (extra.prediction_distribution && Array.isArray(extra.prediction_distribution)) {
        predictionDistribution = extra.prediction_distribution;
      }
    } catch {
      // ignore parse error
    }
  }

  // 关键修复：后端 epoch/step 从 1 开始，前端统一使用 0-based 索引
  const rawStep = metric.step ?? (index + 1);
  const normalizedStep = Math.max(0, rawStep - 1);

  return {
    step: normalizedStep,
    loss: metric.loss,
    accuracy: metric.accuracy,
    trainLoss: metric.loss,
    valLoss: metric.val_loss,
    trainAccuracy: metric.accuracy,
    valAccuracy: metric.val_accuracy,
    learningRate: metric.learning_rate,
    precision,
    recall,
    f1Score,
    gradientNorm,
    weightNorm,
    layerActivations,
    activationStats: createActivationStats(layerActivations),
    featureMaps,
    confusionMatrix,
    predictionDistribution,
  };
}

export function useRealTraining(
  onTrainingData: (data: TrainingStepData[]) => void,
  onTrainingStep: (step: TrainingStepData) => void,
  onStatusChange?: (status: string) => void
): RealTrainingState & RealTrainingActions {
  const [isRealTraining, setIsRealTraining] = useState(false);
  const [backendStatus, setBackendStatus] = useState('idle');
  const [currentEpoch, setCurrentEpoch] = useState(0);
  const [totalEpochs, setTotalEpochs] = useState(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [experimentId, setExperimentId] = useState<string | null>(null);

  const pollTimerRef = useRef<number | null>(null);
  const logSinceRef = useRef(0);
  // 初始化为 -1，确保首个 epoch（step=0）能触发 0 > -1 = true，不被跳过
  const lastMetricStepRef = useRef(-1);
  const experimentIdRef = useRef<string | null>(null);
  const isPollingRef = useRef(false);
  const requestSeqRef = useRef(0);
  const isRequestInFlightRef = useRef(false);
  // 双重兜底：连续3次轮询返回进度100%且状态非终态，强制判定结束
  const consecutiveFullProgressRef = useRef(0);

  /** 停止轮询 */
  const stopPolling = useCallback(() => {
    if (pollTimerRef.current !== null) {
      window.clearTimeout(pollTimerRef.current);
      pollTimerRef.current = null;
    }
    isPollingRef.current = false;
    isRequestInFlightRef.current = false;
  }, []);

  /** 拉取训练日志（增量） */
  const fetchLogs = useCallback(async (expId: string) => {
    try {
      const res = await apiService.getTrainingLogs(expId, logSinceRef.current);
      if (res.code === 200 && res.data) {
        const newLogs = res.data.logs || [];
        if (newLogs.length > 0) {
          setLogs(prev => [...prev, ...newLogs]);
        }
        logSinceRef.current = res.data.total;
      }
    } catch {
      // 日志拉取失败不中断主流程
    }
  }, []);

  /** 拉取全量指标并推送 */
  const fetchAllMetrics = useCallback(async (expId: string) => {
    try {
      const res = await apiService.getTrainingMetrics(expId);
      if (res.code === 200 && res.data && res.data.metrics) {
        const stepData = res.data.metrics.map(convertMetricToStepData);
        onTrainingData(stepData);
        if (stepData.length > 0) {
          // stepData中的step已经是0-based了
          lastMetricStepRef.current = stepData[stepData.length - 1].step;
        }
      }
    } catch {
      // ignore
    }
  }, [onTrainingData]);

  /** 轮询训练状态 */
  const pollStatus = useCallback(async () => {
    const expId = experimentIdRef.current;
    if (!expId || !isPollingRef.current) return;
    // 潜在问题修复：互斥锁，防止并发轮询
    if (isRequestInFlightRef.current) return;
    isRequestInFlightRef.current = true;
    const mySeq = ++requestSeqRef.current;

    try {
      const res = await apiService.getTrainingStatus(expId);
      // 潜在问题修复：序列号校验，丢弃过期响应
      if (!isPollingRef.current || mySeq !== requestSeqRef.current) return;

      if (res.code === 200 && res.data) {
        const statusData = res.data as TrainingStatusData;
        setBackendStatus(statusData.status);
        setCurrentEpoch(statusData.current_epoch || 0);
        setTotalEpochs(statusData.total_epochs || 0);
        setElapsedSeconds(statusData.elapsed_seconds || 0);

        if (statusData.error) {
          setError(statusData.error);
        }

        // 状态变化回调
        if (onStatusChange) {
          onStatusChange(statusData.status);
        }

        // 有新指标时推送
        if (statusData.latest_metrics) {
          // 后端epoch是1-based，lastMetricStepRef存储0-based的step
          const metricEpoch = statusData.latest_metrics.epoch ?? 0;
          const metricStep = Math.max(0, metricEpoch - 1);
          if (metricStep > lastMetricStepRef.current) {
            // 拉取全量指标以获取完整数据
            await fetchAllMetrics(expId);
            // 序列号再次校验（fetchAllMetrics后可能已过期）
            if (!isPollingRef.current || mySeq !== requestSeqRef.current) return;
            // 同时推送最新单条数据供实时显示
            const stepData = convertMetricToStepData({
              step: metricEpoch,
              epoch: metricEpoch,
              loss: statusData.latest_metrics.loss,
              accuracy: statusData.latest_metrics.accuracy,
              val_loss: statusData.latest_metrics.val_loss,
              val_accuracy: statusData.latest_metrics.val_accuracy,
              learning_rate: statusData.latest_metrics.learning_rate,
              batch_size: statusData.latest_metrics.batch_size,
              metric_type: 'training',
              extra_data: statusData.latest_metrics.extra_data || null,
              created_at: null,
            }, metricStep);
            onTrainingStep(stepData);
            lastMetricStepRef.current = metricStep;
          }
        }

        // 增量拉取日志
        await fetchLogs(expId);
        if (!isPollingRef.current || mySeq !== requestSeqRef.current) return;

        // 检测终态：立即停止轮询
        if (TERMINAL_STATUSES.has(statusData.status)) {
          stopPolling();
          setIsRealTraining(false);
          // 终态时再拉一次全量指标确保数据完整
          await fetchAllMetrics(expId);
        } else {
          // 双重兜底：连续3次轮询返回进度100%且状态非终态，强制判定为完成
          if (
            statusData.total_epochs > 0 &&
            statusData.current_epoch >= statusData.total_epochs
          ) {
            consecutiveFullProgressRef.current++;
            if (consecutiveFullProgressRef.current >= 3) {
              console.warn('[useRealTraining] 连续3次进度100%但状态未更新，强制判定为完成');
              stopPolling();
              setIsRealTraining(false);
              setBackendStatus('completed');
              await fetchAllMetrics(expId);
              if (onStatusChange) onStatusChange('completed');
            }
          } else {
            consecutiveFullProgressRef.current = 0;
          }
        }
      }
    } catch (err: any) {
      // 网络异常：不中断轮询，但记录错误
      console.warn('[useRealTraining] 轮询状态失败:', err);
    } finally {
      isRequestInFlightRef.current = false;
    }
  }, [fetchAllMetrics, fetchLogs, onStatusChange, onTrainingStep, stopPolling]);

  /** 启动轮询 - 潜在问题修复：使用递归 setTimeout 替代 setInterval，确保上一次完成后才调度下一次 */
  const startPolling = useCallback((expId: string) => {
    stopPolling();
    experimentIdRef.current = expId;
    isPollingRef.current = true;
    logSinceRef.current = 0;
    // 初始化为 -1，确保首个 epoch 不被跳过
    lastMetricStepRef.current = -1;
    requestSeqRef.current = 0;
    consecutiveFullProgressRef.current = 0;
    setLogs([]);
    setError(null);

    // 递归调度：上一次 pollStatus 完成后再设置下一次定时器
    const scheduleNext = () => {
      if (!isPollingRef.current) return;
      pollTimerRef.current = window.setTimeout(async () => {
        if (!isPollingRef.current) return;
        await pollStatus();
        scheduleNext();
      }, POLL_INTERVAL_MS);
    };

    // 立即执行一次，完成后开始定时调度
    pollStatus().then(() => scheduleNext());
  }, [pollStatus, stopPolling]);

  /** 启动真实训练 */
  const startRealTraining = useCallback(async (
    expId: string,
    datasetId: string,
    hyperparams?: Record<string, any>,
    modelConfig?: Record<string, any>
  ): Promise<boolean> => {
    try {
      setError(null);
      setExperimentId(expId);
      setBackendStatus('pending');
      setCurrentEpoch(0);
      setLogs([]);

      const res = await apiService.startTraining(expId, {
        dataset_id: datasetId,
        hyperparams,
        model_config: modelConfig || undefined,
      });

      if (res.code === 200 && res.data) {
        setIsRealTraining(true);
        setBackendStatus('running');
        setTotalEpochs(res.data.total_epochs || hyperparams?.epochs || 0);
        startPolling(expId);
        return true;
      } else {
        setError(res.message || '启动训练失败');
        setBackendStatus('failed');
        return false;
      }
    } catch (err: any) {
      setError(err.message || '启动训练失败，已降级为模拟模式');
      setBackendStatus('failed');
      return false;
    }
  }, [startPolling]);

  /** 停止真实训练 */
  const stopRealTraining = useCallback(async (): Promise<boolean> => {
    const expId = experimentIdRef.current;
    if (!expId) return false;

    try {
      const res = await apiService.stopTraining(expId);
      if (res.code === 200) {
        // 潜在问题修复：立即停止轮询并重置状态，不依赖后端确认
        // 后端会处理实际的训练停止，前端无需等待
        stopPolling();
        setIsRealTraining(false);
        setBackendStatus('stopped');
        // 终态时拉一次全量指标确保数据完整
        await fetchAllMetrics(expId);
        return true;
      }
      return false;
    } catch (err: any) {
      // 潜在问题修复：网络异常也强制停止前端状态，避免卡死
      stopPolling();
      setIsRealTraining(false);
      setBackendStatus('stopped');
      setError(err.message || '停止训练失败');
      return false;
    }
  }, [stopPolling, fetchAllMetrics]);

  /** 重置状态 */
  const resetRealTraining = useCallback(() => {
    stopPolling();
    setIsRealTraining(false);
    setBackendStatus('idle');
    setCurrentEpoch(0);
    setTotalEpochs(0);
    setLogs([]);
    setError(null);
    setElapsedSeconds(0);
    setExperimentId(null);
    experimentIdRef.current = null;
    logSinceRef.current = 0;
    lastMetricStepRef.current = 0;
  }, [stopPolling]);

  /** 手动拉取全量指标（用于页面恢复） */
  const fetchMetrics = useCallback(async (expId: string): Promise<TrainingStepData[]> => {
    try {
      const res = await apiService.getTrainingMetrics(expId);
      if (res.code === 200 && res.data && res.data.metrics) {
        return res.data.metrics.map(convertMetricToStepData);
      }
    } catch {
      // ignore
    }
    return [];
  }, []);

  // 组件卸载时清理
  useEffect(() => {
    return () => {
      stopPolling();
    };
  }, [stopPolling]);

  return {
    isRealTraining,
    backendStatus,
    currentEpoch,
    totalEpochs,
    logs,
    error,
    elapsedSeconds,
    experimentId,
    startRealTraining,
    stopRealTraining,
    resetRealTraining,
    fetchMetrics,
  };
}
