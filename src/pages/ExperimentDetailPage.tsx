import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import ReactECharts from 'echarts-for-react';
import * as echarts from 'echarts';
import { jsPDF } from 'jspdf';
import { apiService, type ExperimentDetailData } from '../services/api';
import { useGlobalTraining } from '../contexts/GlobalTrainingContext';
import { useToast } from '../contexts/ToastContext';
import { DatasetDetailModal } from '../components/dataset/DatasetDetailModal';
import type { TrainingStepData } from '../types/training';
import type { ChartColors } from '../hooks/useColorConfig';

/* ============================================
   ExperimentDetailPage — 实验详情页
   - 多图表Tab：训练曲线 / 混淆矩阵 / 每类指标 / 梯度&权重
   - 科研导出：SVG矢量图 / PNG / 逐epoch指标CSV / 完整JSON / 模型权重
   - 规范化图表：白色背景、清晰坐标轴、图例、标题（适合论文/报告）
   ============================================ */

const STATUS_LABELS: Record<string, { label: string; color: string }> = {
  draft: { label: '草稿', color: 'bg-gray-400/20 text-gray-400' },
  running: { label: '训练中', color: 'bg-blue-400/20 text-blue-400' },
  completed: { label: '已完成', color: 'bg-emerald-400/20 text-emerald-400' },
  failed: { label: '失败', color: 'bg-red-400/20 text-red-400' },
  paused: { label: '已暂停', color: 'bg-amber-400/20 text-amber-400' },
};

const DEFAULT_COLORS: ChartColors = {
  trainLoss: '#e879f9',
  valLoss: '#fbbf24',
  trainAccuracy: '#4ade80',
  valAccuracy: '#60a5fa',
  learningRate: '#f472b6',
  gradientNorm: '#fbbf24',
  weightNorm: '#c084fc',
  confusionMatrix: '#4ade80',
  activationHistogram: ['#4ade80', '#fbbf24', '#e879f9', '#60a5fa'],
  cnnLayers: ['#4ade80', '#fbbf24', '#e879f9', '#60a5fa'],
  featureMaps: '#c084fc',
  success: '#4ade80',
  warning: '#fbbf24',
  danger: '#f87171',
  info: '#60a5fa',
  primary: '#4ade80',
  secondary: '#64748b',
  accent: '#e879f9',
};

type ChartTab = 'curves' | 'confusion' | 'perclass' | 'gradients' | 'roc' | 'pr';
type ExportFormat = 'svg' | 'png' | 'pdf' | 'metrics-csv' | 'json' | 'batch-csv' | 'excel' | 'cm-csv' | 'cm-json' | 'cm-png';

const CHART_TABS: { key: ChartTab; label: string; icon: string }[] = [
  { key: 'curves', label: '训练曲线', icon: 'M3 3v18h18 M7 14l4-6 4 3 5-8' },
  { key: 'confusion', label: '混淆矩阵', icon: 'M3 3h18v18H3z M3 9h18 M9 3v18' },
  { key: 'perclass', label: '每类指标', icon: 'M18 20V10 M12 20V4 M6 20v-6' },
  { key: 'roc', label: 'ROC曲线', icon: 'M3 3v18h18 M7 17l4-8 4 2 6-9' },
  { key: 'pr', label: 'PR曲线', icon: 'M3 21V3l18 0 M7 17c4-8 8-6 14-12' },
  { key: 'gradients', label: '梯度/权重', icon: 'M22 12h-4l-3 9L9 3l-3 9H2' },
];

function formatNumber(num: number): string {
  if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
  if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
  return num.toLocaleString();
}

function formatDate(dateStr: string): string {
  if (!dateStr) return '-';
  const d = new Date(dateStr);
  if (isNaN(d.getTime())) return '-';
  return d.toLocaleString('zh-CN', {
    year: 'numeric', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit',
  });
}

function formatDuration(seconds: number | null): string {
  if (!seconds || seconds <= 0) return '-';
  if (seconds < 60) return `${seconds.toFixed(1)} 秒`;
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  if (m < 60) return `${m} 分 ${s} 秒`;
  const h = Math.floor(m / 60);
  return `${h} 时 ${m % 60} 分`;
}

function formatShape(shape: number[] | number | undefined | null): string {
  if (shape === undefined || shape === null) return '-';
  if (typeof shape === 'number') return `[${shape}]`;
  if (!Array.isArray(shape) || shape.length === 0) return '-';
  return `[${shape.join(', ')}]`;
}

export default function ExperimentDetailPage() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const toast = useToast();
  const { reset, setTrainingData, goToStep } = useGlobalTraining();

  const [detail, setDetail] = useState<ExperimentDetailData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [datasetModalId, setDatasetModalId] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<ChartTab>('curves');
  const [showExportMenu, setShowExportMenu] = useState(false);
  const [rocPrData, setRocPrData] = useState<{ roc: any; pr: any; macro_auc?: number } | null>(null);
  const chartRef = useRef<ReactECharts | null>(null);
  const exportMenuRef = useRef<HTMLDivElement | null>(null);
  const scrollContainerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    const fetchDetail = async () => {
      setLoading(true);
      setError(null);
      try {
        const res = await apiService.getExperimentDetail(id);
        if (cancelled) return;
        if (res.code === 200 && res.data) {
          setDetail(res.data);
        } else {
          setError(res.message || '加载失败');
        }
      } catch (err: any) {
        if (!cancelled) setError(err.message || '无法连接到服务器');
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    fetchDetail();
    return () => { cancelled = true; };
  }, [id]);

  useEffect(() => {
    if (!id || !detail) return;
    if (detail.status !== 'running') return;
    const pollTimer = window.setInterval(async () => {
      try {
        const res = await apiService.getExperimentDetail(id);
        if (res.code === 200 && res.data) {
          setDetail(res.data);
          if (res.data.status !== 'running') window.clearInterval(pollTimer);
        }
      } catch {}
    }, 2000);
    return () => window.clearInterval(pollTimer);
  }, [id, detail?.status]);

  // 获取ROC/PR数据（从metrics中提取）
  useEffect(() => {
    if (!id || !detail || detail.status !== 'completed') return;
    const fetchRocPr = async () => {
      try {
        const res = await apiService.getExperimentMetrics(id, { limit: 100 });
        if (res.code === 200 && res.data.metrics?.length > 0) {
          const lastMetric = res.data.metrics[res.data.metrics.length - 1];
          const extra = lastMetric.extra_data;
          if (extra && (extra.roc_curve || extra.pr_curve)) {
            setRocPrData({
              roc: extra.roc_curve,
              pr: extra.pr_curve,
              macro_auc: extra.roc_curve?.macro_auc,
            });
          }
        }
      } catch {}
    };
    fetchRocPr();
  }, [id, detail?.status, detail?.experiment_id]);

  useEffect(() => {
    const clickHandler = (e: MouseEvent) => {
      if (exportMenuRef.current && !exportMenuRef.current.contains(e.target as Node)) {
        setShowExportMenu(false);
      }
    };
    const wheelHandler = (e: WheelEvent) => {
      // 忽略弹窗内部的滚轮事件
      if (exportMenuRef.current?.contains(e.target as Node)) return;
      setShowExportMenu(false);
    };
    const scrollHandler = () => setShowExportMenu(false);
    document.addEventListener('mousedown', clickHandler);
    document.addEventListener('wheel', wheelHandler, { passive: true });
    const scrollEl = scrollContainerRef.current;
    if (scrollEl) scrollEl.addEventListener('scroll', scrollHandler, { passive: true });
    return () => {
      document.removeEventListener('mousedown', clickHandler);
      document.removeEventListener('wheel', wheelHandler);
      if (scrollEl) scrollEl.removeEventListener('scroll', scrollHandler);
    };
  }, []);

  const trainingStepData: TrainingStepData[] = useMemo(() => {
    if (!detail?.training_history?.length) return [];
    return detail.training_history.map((h) => ({
      step: h.step,
      loss: h.train_loss,
      accuracy: h.train_acc,
      trainLoss: h.train_loss,
      valLoss: h.val_loss,
      trainAccuracy: h.train_acc,
      valAccuracy: h.val_acc,
      learningRate: h.learning_rate,
      precision: h.precision,
      recall: h.recall,
      f1Score: h.f1,
      gradientNorm: h.gradient_norm || 0,
      weightNorm: h.weight_norm || 0,
      layerActivations: { input: [], conv1: [], fc: [] },
      activationStats: {
        input: { layer: 'input', avg: 0, max: 0, min: 0, sparsity: 0, variance: 0 },
        conv1: { layer: 'conv1', avg: 0, max: 0, min: 0, sparsity: 0, variance: 0 },
        fc: { layer: 'fc', avg: 0, max: 0, min: 0, sparsity: 0, variance: 0 },
      },
      featureMaps: { conv1: [] },
      confusionMatrix: h.confusion_matrix || [],
      predictionDistribution: h.prediction_distribution || [],
    }));
  }, [detail]);

  const lastEpoch = useMemo(() => {
    if (!detail?.training_history?.length) return null;
    return detail.training_history[detail.training_history.length - 1];
  }, [detail]);

  const numClasses = useMemo(() => {
    if (lastEpoch?.confusion_matrix?.length) return lastEpoch.confusion_matrix.length;
    return 10;
  }, [lastEpoch]);

  const classLabels = useMemo(() => {
    return Array.from({ length: numClasses }, (_, i) => String(i));
  }, [numClasses]);

  const researchBg = '#ffffff';

  const curvesOption = useMemo(() => {
    if (trainingStepData.length === 0) return {};
    const steps = trainingStepData.map((d) => d.step);
    const trainLoss = trainingStepData.map((d) => d.trainLoss);
    const valLoss = trainingStepData.map((d) => d.valLoss);
    const trainAcc = trainingStepData.map((d) => d.trainAccuracy * 100);
    const valAcc = trainingStepData.map((d) => d.valAccuracy * 100);
    const lrVals = trainingStepData.map((d) => d.learningRate);
    const maxLoss = Math.max(...trainLoss, ...valLoss, 0.1);
    const stepCount = steps.length;
    const labelInterval = stepCount > 50 ? Math.ceil(stepCount / 8) : stepCount > 20 ? Math.ceil(stepCount / 10) : 0;
    const hasLr = lrVals.some(v => v > 0);

    return {
      backgroundColor: researchBg,
      title: { text: 'Training & Validation Curves', left: 'center', top: 10, textStyle: { fontSize: 14, fontWeight: 'bold', color: '#111' } },
      grid: { left: '8%', right: hasLr ? '8%' : '8%', top: '18%', bottom: '12%', containLabel: true },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(255,255,255,0.96)',
        borderColor: '#ddd',
        borderWidth: 1,
        textStyle: { color: '#333', fontSize: 12 },
        axisPointer: { type: 'cross', lineStyle: { color: '#999', type: 'dashed' } },
      },
      legend: {
        data: ['Train Loss', 'Val Loss', 'Train Acc (%)', 'Val Acc (%)'],
        top: 40,
        textStyle: { color: '#333', fontSize: 11 },
      },
      xAxis: {
        type: 'category',
        data: steps,
        name: 'Epoch',
        nameLocation: 'middle',
        nameGap: 28,
        nameTextStyle: { color: '#555', fontSize: 11 },
        axisLine: { lineStyle: { color: '#999' } },
        axisLabel: { color: '#555', fontSize: 10, interval: labelInterval },
        axisTick: { show: false },
        splitLine: { show: false },
      },
      yAxis: [
        {
          type: 'value', name: 'Loss', nameTextStyle: { color: DEFAULT_COLORS.trainLoss, fontSize: 11 },
          position: 'left', min: 0, max: maxLoss * 1.1,
          axisLine: { show: true, lineStyle: { color: DEFAULT_COLORS.trainLoss } },
          axisLabel: { color: '#555', fontSize: 10, formatter: (v: number) => v.toFixed(2) },
          splitLine: { lineStyle: { color: '#eee', type: 'dashed' } },
        },
        {
          type: 'value', name: 'Accuracy (%)', nameTextStyle: { color: DEFAULT_COLORS.trainAccuracy, fontSize: 11 },
          position: 'right', min: 0, max: 100,
          axisLine: { show: true, lineStyle: { color: DEFAULT_COLORS.trainAccuracy } },
          axisLabel: { color: '#555', fontSize: 10, formatter: (v: number) => `${v.toFixed(0)}%` },
          splitLine: { show: false },
        },
      ],
      series: [
        { name: 'Train Loss', type: 'line', data: trainLoss, smooth: 0.3, symbol: 'circle', symbolSize: 3,
          lineStyle: { color: DEFAULT_COLORS.trainLoss, width: 2 }, itemStyle: { color: DEFAULT_COLORS.trainLoss } },
        { name: 'Val Loss', type: 'line', data: valLoss, smooth: 0.3, symbol: 'diamond', symbolSize: 3,
          lineStyle: { color: DEFAULT_COLORS.valLoss, width: 2, type: 'dashed' }, itemStyle: { color: DEFAULT_COLORS.valLoss } },
        { name: 'Train Acc (%)', type: 'line', data: trainAcc, smooth: 0.3, symbol: 'circle', symbolSize: 3, yAxisIndex: 1,
          lineStyle: { color: DEFAULT_COLORS.trainAccuracy, width: 2 }, itemStyle: { color: DEFAULT_COLORS.trainAccuracy } },
        { name: 'Val Acc (%)', type: 'line', data: valAcc, smooth: 0.3, symbol: 'diamond', symbolSize: 3, yAxisIndex: 1,
          lineStyle: { color: DEFAULT_COLORS.valAccuracy, width: 2, type: 'dashed' }, itemStyle: { color: DEFAULT_COLORS.valAccuracy } },
      ],
      animation: true,
      animationDuration: 300,
    };
  }, [trainingStepData]);

  const confusionOption = useMemo(() => {
    if (!lastEpoch?.confusion_matrix?.length) return {};
    const cm = lastEpoch.confusion_matrix;
    const nc = cm.length;
    const data: [number, number, number][] = [];
    let maxVal = 0;
    for (let i = 0; i < nc; i++) {
      for (let j = 0; j < nc; j++) {
        const v = cm[i][j] || 0;
        data.push([j, i, v]);
        if (v > maxVal) maxVal = v;
      }
    }

    return {
      backgroundColor: researchBg,
      title: { text: 'Confusion Matrix (Final Epoch)', left: 'center', top: 10, textStyle: { fontSize: 14, fontWeight: 'bold', color: '#111' } },
      tooltip: {
        position: 'top',
        backgroundColor: 'rgba(255,255,255,0.96)',
        borderColor: '#ddd',
        textStyle: { color: '#333', fontSize: 12 },
        formatter: (p: any) => {
          const [x, y, v] = p.data;
          return `True: ${y}<br/>Pred: ${x}<br/>Count: <b>${v}</b>`;
        },
      },
      grid: { left: '12%', right: '12%', top: '18%', bottom: '15%', containLabel: true },
      xAxis: {
        type: 'category',
        data: classLabels,
        name: 'Predicted Label',
        nameLocation: 'middle',
        nameGap: 30,
        nameTextStyle: { color: '#555', fontSize: 11 },
        axisLine: { lineStyle: { color: '#999' } },
        axisLabel: { color: '#555', fontSize: 10 },
        splitLine: { show: false },
      },
      yAxis: {
        type: 'category',
        data: classLabels,
        name: 'True Label',
        nameLocation: 'middle',
        nameGap: 40,
        nameTextStyle: { color: '#555', fontSize: 11 },
        axisLine: { lineStyle: { color: '#999' } },
        axisLabel: { color: '#555', fontSize: 10 },
        splitLine: { show: false },
      },
      visualMap: {
        min: 0,
        max: Math.max(maxVal, 1),
        calculable: true,
        orient: 'vertical',
        right: '2%',
        top: 'center',
        inRange: { color: ['#f7fbff', '#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b'] },
        textStyle: { color: '#555', fontSize: 10 },
      },
      series: [{
        name: 'Confusion Matrix',
        type: 'heatmap',
        data,
        label: { show: true, fontSize: 10, color: '#333', formatter: (p: any) => p.data[2] > 0 ? p.data[2] : '' },
        emphasis: { itemStyle: { shadowBlur: 10, shadowColor: 'rgba(0,0,0,0.3)' } },
      }],
    };
  }, [lastEpoch, classLabels]);

  const perclassOption = useMemo(() => {
    if (!lastEpoch) return {};
    const pcp = lastEpoch.per_class_precision || [];
    const pcr = lastEpoch.per_class_recall || [];
    const pcf = lastEpoch.per_class_f1 || [];
    const labels = classLabels.slice(0, Math.max(pcp.length, pcr.length, pcf.length, 10));
    const nc = labels.length;

    return {
      backgroundColor: researchBg,
      title: { text: 'Per-class Precision / Recall / F1', left: 'center', top: 10, textStyle: { fontSize: 14, fontWeight: 'bold', color: '#111' } },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(255,255,255,0.96)',
        borderColor: '#ddd',
        textStyle: { color: '#333', fontSize: 12 },
        axisPointer: { type: 'shadow' },
      },
      legend: { top: 40, data: ['Precision', 'Recall', 'F1'], textStyle: { color: '#333', fontSize: 11 } },
      grid: { left: '8%', right: '5%', top: '20%', bottom: '12%', containLabel: true },
      xAxis: {
        type: 'category', data: labels, name: 'Class',
        nameLocation: 'middle', nameGap: 28, nameTextStyle: { color: '#555', fontSize: 11 },
        axisLine: { lineStyle: { color: '#999' } },
        axisLabel: { color: '#555', fontSize: 10 },
      },
      yAxis: {
        type: 'value', name: 'Score', min: 0, max: 1,
        nameTextStyle: { color: '#555', fontSize: 11 },
        axisLine: { lineStyle: { color: '#999' } },
        axisLabel: { color: '#555', fontSize: 10, formatter: (v: number) => v.toFixed(1) },
        splitLine: { lineStyle: { color: '#eee', type: 'dashed' } },
      },
      series: [
        { name: 'Precision', type: 'bar', data: pcp.slice(0, nc), barGap: '10%', barWidth: '20%',
          itemStyle: { color: '#60a5fa' } },
        { name: 'Recall', type: 'bar', data: pcr.slice(0, nc), barWidth: '20%',
          itemStyle: { color: '#4ade80' } },
        { name: 'F1', type: 'bar', data: pcf.slice(0, nc), barWidth: '20%',
          itemStyle: { color: '#fbbf24' } },
      ],
    };
  }, [lastEpoch, classLabels]);

  const gradientsOption = useMemo(() => {
    if (trainingStepData.length === 0) return {};
    const steps = trainingStepData.map((d) => d.step);
    const gradNorms = trainingStepData.map((d) => d.gradientNorm);
    const weightNorms = trainingStepData.map((d) => d.weightNorm);
    const hasGrad = gradNorms.some(v => v > 0);
    const hasWeight = weightNorms.some(v => v > 0);

    return {
      backgroundColor: researchBg,
      title: { text: 'Gradient Norm & Weight Norm over Epochs', left: 'center', top: 10, textStyle: { fontSize: 14, fontWeight: 'bold', color: '#111' } },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(255,255,255,0.96)',
        borderColor: '#ddd',
        textStyle: { color: '#333', fontSize: 12 },
      },
      legend: { top: 40, data: ['Gradient Norm', 'Weight Norm'], textStyle: { color: '#333', fontSize: 11 } },
      grid: { left: '8%', right: '5%', top: '20%', bottom: '12%', containLabel: true },
      xAxis: {
        type: 'category', data: steps, name: 'Epoch',
        nameLocation: 'middle', nameGap: 28, nameTextStyle: { color: '#555', fontSize: 11 },
        axisLine: { lineStyle: { color: '#999' } },
        axisLabel: { color: '#555', fontSize: 10 },
      },
      yAxis: [
        {
          type: 'value', name: 'Gradient Norm', nameTextStyle: { color: DEFAULT_COLORS.gradientNorm, fontSize: 11 },
          axisLine: { lineStyle: { color: DEFAULT_COLORS.gradientNorm } },
          axisLabel: { color: '#555', fontSize: 10 },
          splitLine: { lineStyle: { color: '#eee', type: 'dashed' } },
        },
        {
          type: 'value', name: 'Weight Norm', nameTextStyle: { color: DEFAULT_COLORS.weightNorm, fontSize: 11 },
          position: 'right',
          axisLine: { lineStyle: { color: DEFAULT_COLORS.weightNorm } },
          axisLabel: { color: '#555', fontSize: 10 },
          splitLine: { show: false },
        },
      ],
      series: [
        { name: 'Gradient Norm', type: 'line', data: hasGrad ? gradNorms : [], smooth: 0.3,
          symbol: 'none', lineStyle: { color: DEFAULT_COLORS.gradientNorm, width: 2 } },
        { name: 'Weight Norm', type: 'line', data: hasWeight ? weightNorms : [], smooth: 0.3, yAxisIndex: 1,
          symbol: 'none', lineStyle: { color: DEFAULT_COLORS.weightNorm, width: 2 } },
      ],
    };
  }, [trainingStepData]);

  const rocOption = useMemo(() => {
    if (!rocPrData?.roc?.fpr) return {
      backgroundColor: researchBg,
      title: { text: 'ROC Curve', left: 'center', top: 'center', textStyle: { fontSize: 14, color: '#999' } },
    };
    const roc = rocPrData.roc;
    const series: any[] = [];
    const legendData: string[] = [];
    const colors = ['#60a5fa', '#f87171', '#4ade80', '#fbbf24', '#c084fc', '#f472b6', '#22d3ee', '#fb923c', '#a3e635', '#e879f9'];

    Object.entries(roc.fpr).forEach(([clsName, fprArr]: [string, any], idx: number) => {
      const tprArr = roc.tpr?.[clsName] || [];
      const auc = roc.auc_scores?.[clsName];
      const points = (fprArr as number[]).map((f, i) => [f, (tprArr as number[])[i] || 0]);
      points.unshift([0, 0]);
      points.push([1, 1]);
      const shortLabel = `Class ${clsName} (AUC=${auc?.toFixed(3) ?? '-'})`;
      legendData.push(shortLabel);
      series.push({
        name: shortLabel,
        type: 'line',
        data: points,
        smooth: false,
        symbol: 'none',
        lineStyle: { color: colors[idx % colors.length], width: 1.5 },
      });
    });

    series.push({
      name: 'Random (AUC=0.5)',
      type: 'line',
      data: [[0, 0], [1, 1]],
      symbol: 'none',
      lineStyle: { color: '#999', width: 1, type: 'dashed' },
    });
    legendData.push('Random (AUC=0.5)');

    return {
      backgroundColor: researchBg,
      title: {
        text: `ROC Curve${rocPrData.macro_auc != null ? ` (Macro AUC = ${rocPrData.macro_auc.toFixed(3)})` : ''}`,
        left: 'center', top: 10, textStyle: { fontSize: 14, fontWeight: 'bold', color: '#111' }
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(255,255,255,0.96)',
        borderColor: '#ddd',
        textStyle: { color: '#333', fontSize: 11 },
        formatter: (params: any) => {
          if (!Array.isArray(params) || params.length === 0) return '';
          const p = params[0];
          return `FPR: ${(p.data[0] * 100).toFixed(1)}%<br/>TPR: ${(p.data[1] * 100).toFixed(1)}%`;
        }
      },
      legend: {
        data: legendData,
        top: 38, type: 'scroll', width: '92%',
        textStyle: { color: '#333', fontSize: 10 },
        pageTextStyle: { color: '#555' },
        pageIconColor: '#666',
        pageIconInactiveColor: '#ccc',
        pageIconSize: 12,
        itemWidth: 14,
        itemHeight: 8,
        itemGap: 8,
      },
      grid: { left: '8%', right: '5%', top: '24%', bottom: '12%', containLabel: true },
      xAxis: {
        type: 'value', name: 'False Positive Rate', min: 0, max: 1,
        nameLocation: 'middle', nameGap: 28, nameTextStyle: { color: '#555', fontSize: 11 },
        axisLine: { lineStyle: { color: '#999' } },
        axisLabel: { color: '#555', fontSize: 10, formatter: (v: number) => v.toFixed(1) },
        splitLine: { lineStyle: { color: '#eee', type: 'dashed' } },
      },
      yAxis: {
        type: 'value', name: 'True Positive Rate', min: 0, max: 1,
        nameTextStyle: { color: '#555', fontSize: 11 },
        axisLine: { lineStyle: { color: '#999' } },
        axisLabel: { color: '#555', fontSize: 10, formatter: (v: number) => v.toFixed(1) },
        splitLine: { lineStyle: { color: '#eee', type: 'dashed' } },
      },
      series,
      animationDuration: 300,
    };
  }, [rocPrData]);

  const prOption = useMemo(() => {
    if (!rocPrData?.pr?.recall) return {
      backgroundColor: researchBg,
      title: { text: 'Precision-Recall Curve', left: 'center', top: 'center', textStyle: { fontSize: 14, color: '#999' } },
    };
    const pr = rocPrData.pr;
    const series: any[] = [];
    const legendData: string[] = [];
    const colors = ['#60a5fa', '#f87171', '#4ade80', '#fbbf24', '#c084fc', '#f472b6', '#22d3ee', '#fb923c', '#a3e635', '#e879f9'];

    Object.entries(pr.recall).forEach(([clsName, recallArr]: [string, any], idx: number) => {
      const precArr = pr.precision?.[clsName] || [];
      const ap = pr.ap_scores?.[clsName];
      const points = (recallArr as number[]).map((r, i) => [r, (precArr as number[])[i] || 0]);
      const shortLabel = `Class ${clsName} (AP=${ap?.toFixed(3) ?? '-'})`;
      legendData.push(shortLabel);
      series.push({
        name: shortLabel,
        type: 'line',
        data: points,
        smooth: false,
        symbol: 'none',
        lineStyle: { color: colors[idx % colors.length], width: 1.5 },
      });
    });

    return {
      backgroundColor: researchBg,
      title: {
        text: `Precision-Recall Curve${pr.map_ap != null ? ` (mAP = ${(pr.map_ap * 100).toFixed(1)}%)` : ''}`,
        left: 'center', top: 10, textStyle: { fontSize: 14, fontWeight: 'bold', color: '#111' }
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(255,255,255,0.96)',
        borderColor: '#ddd',
        textStyle: { color: '#333', fontSize: 11 },
        formatter: (params: any) => {
          if (!Array.isArray(params) || params.length === 0) return '';
          const p = params[0];
          return `Recall: ${(p.data[0] * 100).toFixed(1)}%<br/>Precision: ${(p.data[1] * 100).toFixed(1)}%`;
        }
      },
      legend: {
        data: legendData,
        top: 38, type: 'scroll', width: '92%',
        textStyle: { color: '#333', fontSize: 10 },
        pageTextStyle: { color: '#555' },
        pageIconColor: '#666',
        pageIconInactiveColor: '#ccc',
        pageIconSize: 12,
        itemWidth: 14,
        itemHeight: 8,
        itemGap: 8,
      },
      grid: { left: '8%', right: '5%', top: '24%', bottom: '12%', containLabel: true },
      xAxis: {
        type: 'value', name: 'Recall', min: 0, max: 1,
        nameLocation: 'middle', nameGap: 28, nameTextStyle: { color: '#555', fontSize: 11 },
        axisLine: { lineStyle: { color: '#999' } },
        axisLabel: { color: '#555', fontSize: 10, formatter: (v: number) => v.toFixed(1) },
        splitLine: { lineStyle: { color: '#eee', type: 'dashed' } },
      },
      yAxis: {
        type: 'value', name: 'Precision', min: 0, max: 1,
        nameTextStyle: { color: '#555', fontSize: 11 },
        axisLine: { lineStyle: { color: '#999' } },
        axisLabel: { color: '#555', fontSize: 10, formatter: (v: number) => v.toFixed(1) },
        splitLine: { lineStyle: { color: '#eee', type: 'dashed' } },
      },
      series,
      animationDuration: 300,
    };
  }, [rocPrData]);

  const currentChartOption = useMemo(() => {
    switch (activeTab) {
      case 'confusion': return confusionOption;
      case 'perclass': return perclassOption;
      case 'gradients': return gradientsOption;
      case 'roc': return rocOption;
      case 'pr': return prOption;
      default: return curvesOption;
    }
  }, [activeTab, curvesOption, confusionOption, perclassOption, gradientsOption, rocOption, prOption]);

  const downloadBlob = useCallback((blob: Blob, filename: string) => {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(() => URL.revokeObjectURL(url), 100);
  }, []);

  const handleExportChart = useCallback((format: 'svg' | 'png') => {
    const chartInst = chartRef.current?.getEchartsInstance();
    if (!chartInst) { toast.showError('图表尚未就绪'); return; }
    const name = (detail?.name || 'experiment').replace(/[\\/:*?"<>|]/g, '_');
    const tabName = CHART_TABS.find(t => t.key === activeTab)?.label || 'chart';
    const ts = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
    const filename = `${name}_${tabName}_${ts}.${format}`;

    if (format === 'svg') {
      const svgStr = chartInst.renderToSVGString();
      const blob = new Blob([svgStr], { type: 'image/svg+xml;charset=utf-8' });
      downloadBlob(blob, filename);
      toast.showSuccess('SVG 矢量图导出成功');
    } else {
      const dataUrl = chartInst.getDataURL({ type: 'png', pixelRatio: 3, backgroundColor: researchBg });
      const arr = dataUrl.split(',');
      const mime = arr[0].match(/:(.*?);/)?.[1] || 'image/png';
      const bstr = atob(arr[1]);
      let n = bstr.length;
      const u8 = new Uint8Array(n);
      while (n--) u8[n] = bstr.charCodeAt(n);
      downloadBlob(new Blob([u8], { type: mime }), filename);
      toast.showSuccess('PNG 高清图导出成功');
    }
    setShowExportMenu(false);
  }, [detail, activeTab, downloadBlob, toast]);

  const handleExport = useCallback(async (fmt: ExportFormat) => {
    if (!detail) return;
    setExporting(true);
    setShowExportMenu(false);
    try {
      const name = detail.name.replace(/[\\/:*?"<>|]/g, '_');
      const ts = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');

      if (fmt === 'svg' || fmt === 'png') {
        handleExportChart(fmt);
        setExporting(false);
        return;
      }

      if (fmt === 'pdf') {
        // 批量导出所有图表为PDF
        const chartDefs: { title: string; option: any; width: number; height: number }[] = [];
        if (trainingStepData.length > 0) {
          chartDefs.push({ title: '训练曲线 (Loss & Accuracy)', option: curvesOption, width: 1100, height: 480 });
        }
        if (lastEpoch?.confusion_matrix?.length) {
          chartDefs.push({ title: '混淆矩阵', option: confusionOption, width: 900, height: 520 });
        }
        if (lastEpoch) {
          chartDefs.push({ title: '每类指标 (Precision/Recall/F1)', option: perclassOption, width: 1100, height: 480 });
        }
        if (trainingStepData.length > 0) {
          chartDefs.push({ title: '梯度范数 & 权重范数', option: gradientsOption, width: 1100, height: 480 });
        }
        if (rocPrData?.roc?.fpr) {
          chartDefs.push({ title: 'ROC 曲线', option: rocOption, width: 1100, height: 480 });
        }
        if (rocPrData?.pr?.recall) {
          chartDefs.push({ title: 'PR 曲线', option: prOption, width: 1100, height: 480 });
        }

        if (chartDefs.length === 0) {
          toast.showError('没有可导出的图表数据');
          setExporting(false);
          return;
        }

        const container = document.createElement('div');
        container.style.cssText = 'position:fixed;left:-9999px;top:0;';
        document.body.appendChild(container);

        try {
          const pdf = new jsPDF({ orientation: 'landscape', unit: 'px', format: [1200, 850] });
          const pageW = 1200;
          const pageH = 850;
          const marginX = 40;
          const marginY = 40;
          const titleH = 24;

          let isFirstPage = true;

          for (let i = 0; i < chartDefs.length; i++) {
            const { title, option, width: cw, height: ch } = chartDefs[i];

            const chartDiv = document.createElement('div');
            chartDiv.style.cssText = `width:${cw}px;height:${ch}px;`;
            container.appendChild(chartDiv);

            const chartInstance = echarts.init(chartDiv);
            chartInstance.setOption({ ...option, animation: false });

            await new Promise<void>((resolve) => {
              // 使用 requestAnimationFrame 确保渲染完成
              const tryCapture = () => {
                const dataUrl = chartInstance.getDataURL({
                  type: 'png',
                  pixelRatio: 2,
                  backgroundColor: '#ffffff',
                });

                if (!isFirstPage) pdf.addPage();
                isFirstPage = false;

                // 标题
                pdf.setFontSize(14);
                pdf.setTextColor(30, 30, 30);
                pdf.text(title, marginX, marginY + 14);

                // 图表图片 — 自适应缩放适配页面
                const availW = pageW - marginX * 2;
                const availH = pageH - marginY * 2 - titleH;
                const scale = Math.min(availW / cw, availH / ch);
                const imgW = cw * scale;
                const imgH = ch * scale;
                const imgX = marginX + (availW - imgW) / 2;
                const imgY = marginY + titleH + (availH - imgH) / 2;

                pdf.addImage(dataUrl, 'PNG', imgX, imgY, imgW, imgH);

                chartInstance.dispose();
                chartDiv.remove();
                resolve();
              };

              // 延迟一帧确保 ECharts 渲染完成
              requestAnimationFrame(() => {
                requestAnimationFrame(tryCapture);
              });
            });
          }

          pdf.save(`${name}_charts_${ts}.pdf`);
          toast.showSuccess('PDF 导出成功（包含全部图表）');
        } finally {
          document.body.removeChild(container);
        }
        setExporting(false);
        return;
      }

      let blob: Blob;
      let filename: string;
      if (fmt === 'metrics-csv') {
        blob = await apiService.exportMetricsCSV(detail.experiment_id);
        filename = `${name}_metrics_${ts}.csv`;
      } else if (fmt === 'json') {
        blob = await apiService.exportExperimentJSON(detail.experiment_id);
        filename = `${name}_full_${ts}.json`;
      } else if (fmt === 'excel') {
        blob = await apiService.exportExperimentExcel(detail.experiment_id);
        filename = `${name}_report_${ts}.xlsx`;
      } else if (fmt === 'batch-csv') {
        blob = await apiService.exportExperimentsCSV([detail.experiment_id]);
        filename = `${name}_summary_${ts}.csv`;
      } else if (fmt === 'cm-csv') {
        blob = await apiService.exportConfusionMatrix(detail.experiment_id, 'csv');
        filename = `${name}_confusion_matrix_${ts}.csv`;
      } else if (fmt === 'cm-json') {
        blob = await apiService.exportConfusionMatrix(detail.experiment_id, 'json');
        filename = `${name}_confusion_matrix_${ts}.json`;
      } else if (fmt === 'cm-png') {
        blob = await apiService.exportConfusionMatrix(detail.experiment_id, 'png', 300);
        filename = `${name}_confusion_matrix_${ts}.png`;
      } else {
        blob = await apiService.exportExperimentsCSV([detail.experiment_id]);
        filename = `${name}_summary_${ts}.csv`;
      }
      downloadBlob(blob, filename);
      toast.showSuccess('导出成功');
    } catch (err: any) {
      toast.showError('导出失败', err.message);
    } finally {
      setExporting(false);
    }
  }, [detail, activeTab, handleExportChart, downloadBlob, toast]);

  const handleLoadToWorkbench = () => {
    if (!detail) return;
    reset();
    setTrainingData(trainingStepData);
    if (trainingStepData.length > 0) goToStep(trainingStepData.length - 1);
    toast.showSuccess('已加载到工作台', '训练历史已同步');
    navigate('/workbench');
  };

  const handleDelete = async () => {
    if (!detail) return;
    if (!confirm(`确定要删除实验「${detail.name}」吗？此操作不可恢复。`)) return;
    try {
      await apiService.deleteExperiment(detail.experiment_id);
      toast.showSuccess('实验已删除');
      navigate('/experiments');
    } catch (err: any) {
      toast.showError('删除失败', err.message);
    }
  };

  if (loading) {
    return (
      <div className="flex h-full w-full flex-col overflow-hidden bg-[#0a0c14] text-white">
        <div className="flex flex-1 items-center justify-center">
          <div className="flex flex-col items-center gap-4">
            <div className="h-8 w-8 animate-spin rounded-full border-2 border-slate-600 border-t-slate-300" />
            <p className="text-sm text-muted-foreground">加载实验数据中...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error || !detail) {
    return (
      <div className="flex h-full w-full flex-col overflow-hidden bg-[#0a0c14] text-white">
        <div className="flex flex-1 items-center justify-center">
          <div className="flex flex-col items-center justify-center py-20">
            <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full border border-red-400/20 bg-red-400/10">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-red-400">
                <circle cx="12" cy="12" r="10" /><path d="M12 8v4M12 16h.01" />
              </svg>
            </div>
            <h3 className="text-sm font-semibold text-foreground">加载实验详情失败</h3>
            <p className="mt-1 text-xs text-muted-foreground">{error || '实验不存在'}</p>
            <button onClick={() => navigate('/experiments')} className="mt-4 flex items-center gap-1.5 rounded-md border border-white/[0.08] bg-white/[0.02] px-3 py-1.5 text-xs text-muted-foreground transition hover:text-foreground">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M19 12H5M12 19l-7-7 7-7" /></svg>
              返回实验列表
            </button>
          </div>
        </div>
      </div>
    );
  }

  const statusInfo = STATUS_LABELS[detail.status] || { label: detail.status, color: 'bg-gray-400/20 text-gray-400' };
  const ms = detail.metrics_summary || {};
  const hp = detail.hyperparams || {};
  const mc = detail.model_config || {};
  const bi = detail.basic_info || {};
  const hasData = trainingStepData.length > 0;

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-[#0a0c14] text-white">
      <div className="flex-shrink-0 border-b border-white/[0.06] bg-[#0c0e17]/95 backdrop-blur-sm">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <button onClick={() => navigate('/experiments')} className="group flex h-9 w-9 items-center justify-center rounded-lg border border-white/[0.08] bg-white/[0.02] text-muted-foreground transition-all hover:border-primary/30 hover:bg-primary/[0.06] hover:text-primary" title="返回实验列表">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="transition-transform group-hover:-translate-x-0.5">
                <path d="M19 12H5M12 19l-7-7 7-7" />
              </svg>
            </button>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-lg font-bold">{detail.name}</h1>
                <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${statusInfo.color}`}>{statusInfo.label}</span>
              </div>
              <p className="text-xs text-muted-foreground">
                创建于 {formatDate(detail.created_at)} · ID: {detail.experiment_id.slice(0, 8)}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button onClick={handleLoadToWorkbench} className="flex items-center gap-1.5 rounded-md border border-primary/30 bg-primary/10 px-3 py-1.5 text-xs font-semibold text-primary transition-all hover:bg-primary/15" title="加载到工作台继续调试">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M5 3l14 9-14 9V3z" /></svg>
              加载到工作台
            </button>
            {detail.config?.model_path_best && (
              <button onClick={() => window.open(`/api/model/download?path=${encodeURIComponent(detail.config.model_path_best)}`, '_blank')}
                className="flex items-center gap-1.5 rounded-md border border-emerald-400/30 bg-emerald-400/10 px-3 py-1.5 text-xs font-semibold text-emerald-400 transition-all hover:bg-emerald-400/15" title="下载最佳模型权重 (best.pt)">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><path d="M7 10l5 5 5-5" /><path d="M12 15V3" /></svg>
                best.pt
              </button>
            )}
            {detail.config?.model_path_last && (
              <button onClick={() => window.open(`/api/model/download?path=${encodeURIComponent(detail.config.model_path_last)}`, '_blank')}
                className="flex items-center gap-1.5 rounded-md border border-white/[0.08] bg-white/[0.02] px-3 py-1.5 text-xs font-medium text-muted-foreground transition-all hover:text-foreground" title="下载最终模型权重 (last.pt)">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><path d="M7 10l5 5 5-5" /><path d="M12 15V3" /></svg>
                last.pt
              </button>
            )}

            <div className="relative" ref={exportMenuRef}>
              <button onClick={() => setShowExportMenu(!showExportMenu)} disabled={exporting}
                className="flex items-center gap-1.5 rounded-md border border-white/[0.08] bg-white/[0.02] px-3 py-1.5 text-xs font-medium text-muted-foreground transition-all hover:text-foreground disabled:opacity-50">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><path d="M7 10l5 5 5-5" /><path d="M12 15V3" /></svg>
                {exporting ? '导出中...' : '科研导出'}
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M6 9l6 6 6-6" /></svg>
              </button>
              {showExportMenu && (
                <div className="absolute right-0 top-full z-[9999] mt-1 w-64 overflow-hidden rounded-lg border border-white/[0.08] bg-[#151822] shadow-xl">
                  <div className="border-b border-white/[0.06] px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">图表导出</div>
                  <button onClick={() => handleExport('svg')} className="flex w-full items-center gap-2 px-3 py-2 text-xs transition hover:bg-white/[0.04]">
                    <span className="w-12 text-primary font-semibold">SVG</span>
                    <span className="text-muted-foreground">矢量图（论文级，可无限缩放）</span>
                  </button>
                  <button onClick={() => handleExport('png')} className="flex w-full items-center gap-2 px-3 py-2 text-xs transition hover:bg-white/[0.04]">
                    <span className="w-12 text-blue-400 font-semibold">PNG</span>
                    <span className="text-muted-foreground">高清位图（300DPI）</span>
                  </button>
                  <button onClick={() => handleExport('pdf')} className="flex w-full items-center gap-2 px-3 py-2 text-xs transition hover:bg-white/[0.04]">
                    <span className="w-12 text-red-400 font-semibold">PDF</span>
                    <span className="text-muted-foreground">PDF文档（适合直接插入论文）</span>
                  </button>
                  <div className="border-t border-white/[0.06] px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">数据导出</div>
                  <button onClick={() => handleExport('excel')} className="flex w-full items-center gap-2 px-3 py-2 text-xs transition hover:bg-white/[0.04]">
                    <span className="w-12 text-emerald-400 font-semibold">Excel</span>
                    <span className="text-muted-foreground">实验报告 (.xlsx, 多Sheet)</span>
                  </button>
                  <button onClick={() => handleExport('metrics-csv')} className="flex w-full items-center gap-2 px-3 py-2 text-xs transition hover:bg-white/[0.04]">
                    <span className="w-12 text-emerald-400 font-semibold">CSV</span>
                    <span className="text-muted-foreground">逐Epoch指标 (Loss/Acc/P/R/F1)</span>
                  </button>
                  <button onClick={() => handleExport('json')} className="flex w-full items-center gap-2 px-3 py-2 text-xs transition hover:bg-white/[0.04]">
                    <span className="w-12 text-amber-400 font-semibold">JSON</span>
                    <span className="text-muted-foreground">完整数据 (配置+指标+日志)</span>
                  </button>
                  <button onClick={() => handleExport('batch-csv')} className="flex w-full items-center gap-2 px-3 py-2 text-xs transition hover:bg-white/[0.04]">
                    <span className="w-12 text-purple-400 font-semibold">CSV</span>
                    <span className="text-muted-foreground">实验汇总表</span>
                  </button>
                  <div className="border-t border-white/[0.06] px-3 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">混淆矩阵导出</div>
                  <button onClick={() => handleExport('cm-csv')} className="flex w-full items-center gap-2 px-3 py-2 text-xs transition hover:bg-white/[0.04]">
                    <span className="w-12 text-cyan-400 font-semibold">CSV</span>
                    <span className="text-muted-foreground">数值矩阵（可导入Excel）</span>
                  </button>
                  <button onClick={() => handleExport('cm-json')} className="flex w-full items-center gap-2 px-3 py-2 text-xs transition hover:bg-white/[0.04]">
                    <span className="w-12 text-cyan-400 font-semibold">JSON</span>
                    <span className="text-muted-foreground">结构化数据</span>
                  </button>
                  <button onClick={() => handleExport('cm-png')} className="flex w-full items-center gap-2 px-3 py-2 text-xs transition hover:bg-white/[0.04]">
                    <span className="w-12 text-cyan-400 font-semibold">PNG</span>
                    <span className="text-muted-foreground">热力图（300DPI）</span>
                  </button>
                </div>
              )}
            </div>

            <button onClick={handleDelete} className="flex items-center gap-1.5 rounded-md border border-white/[0.08] bg-white/[0.02] px-3 py-1.5 text-xs font-medium text-muted-foreground transition-all hover:border-red-400/30 hover:bg-red-400/10 hover:text-red-400" title="删除实验">
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" /></svg>
              删除
            </button>
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto" ref={scrollContainerRef}>
        <div className="mx-auto max-w-6xl px-6 py-6">
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
          <section className="rounded-xl border border-white/[0.06] bg-[#0f1119] p-5">
            <h2 className="mb-4 flex items-center gap-2 text-sm font-bold">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary"><circle cx="12" cy="12" r="10" /><path d="M12 16v-4M12 8h.01" /></svg>
              基础信息
            </h2>
            <dl className="space-y-2.5 text-xs">
              <div className="flex"><dt className="w-24 text-muted-foreground">描述</dt><dd className="flex-1 text-foreground">{bi.description || '-'}</dd></div>
              <div className="flex"><dt className="w-24 text-muted-foreground">标签</dt>
                <dd className="flex-1">{bi.tags && bi.tags.length > 0 ? (
                  <div className="flex flex-wrap gap-1">{bi.tags.map((tag, i) => (<span key={i} className="rounded px-1.5 py-0.5 text-[10px] bg-white/[0.04] text-muted-foreground">{tag}</span>))}</div>
                ) : '-'}</dd>
              </div>
              <div className="flex"><dt className="w-24 text-muted-foreground">数据集</dt>
                <dd className="flex-1">{hp.dataset_name ? (
                  <button onClick={async () => {
                    try {
                      const res = await apiService.listDatasets({ search: hp.dataset_name, page: 1, page_size: 1 });
                      if (res.data.items.length > 0) setDatasetModalId(res.data.items[0].dataset_id);
                      else toast.showInfo('未找到数据集', '该数据集可能已被删除');
                    } catch { toast.showError('查询数据集失败'); }
                  }} className="inline-flex items-center gap-1 text-primary transition-all hover:underline">
                    {hp.dataset_name}{hp.dataset_version ? ` (${hp.dataset_version})` : ''}
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M7 17L17 7M7 7h10v10" /></svg>
                  </button>
                ) : '-'}</dd>
              </div>
              <div className="flex"><dt className="w-24 text-muted-foreground">备注</dt><dd className="flex-1 text-foreground">{bi.remark || '-'}</dd></div>
              <div className="flex"><dt className="w-24 text-muted-foreground">Precision</dt><dd className="flex-1 font-mono text-foreground">{detail.config?.precision != null ? (detail.config.precision * 100).toFixed(2) + '%' : '-'}</dd></div>
              <div className="flex"><dt className="w-24 text-muted-foreground">Recall</dt><dd className="flex-1 font-mono text-foreground">{detail.config?.recall != null ? (detail.config.recall * 100).toFixed(2) + '%' : '-'}</dd></div>
              <div className="flex"><dt className="w-24 text-muted-foreground">F1 Score</dt><dd className="flex-1 font-mono text-foreground">{detail.config?.f1 != null ? (detail.config.f1 * 100).toFixed(2) + '%' : '-'}</dd></div>
              <div className="flex"><dt className="w-24 text-muted-foreground">创建时间</dt><dd className="flex-1 text-foreground">{formatDate(bi.created_at)}</dd></div>
              <div className="flex"><dt className="w-24 text-muted-foreground">更新时间</dt><dd className="flex-1 text-foreground">{formatDate(bi.updated_at)}</dd></div>
            </dl>
          </section>

          <section className="rounded-xl border border-white/[0.06] bg-[#0f1119] p-5">
            <h2 className="mb-4 flex items-center gap-2 text-sm font-bold">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary"><path d="M12 2a10 10 0 1 0 10 10A10 10 0 0 0 12 2z" /><path d="M12 16v-4M12 8h.01" /></svg>
              超参数
            </h2>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="rounded-lg border border-white/[0.04] bg-white/[0.01] px-3 py-2"><div className="text-muted-foreground">学习率</div><div className="mt-0.5 font-mono text-foreground">{(hp?.learning_rate ?? 0).toFixed(4)}</div></div>
              <div className="rounded-lg border border-white/[0.04] bg-white/[0.01] px-3 py-2"><div className="text-muted-foreground">批次大小</div><div className="mt-0.5 font-mono text-foreground">{hp.batch_size}</div></div>
              <div className="rounded-lg border border-white/[0.04] bg-white/[0.01] px-3 py-2"><div className="text-muted-foreground">优化器</div><div className="mt-0.5 text-foreground">{hp.optimizer || '-'}</div></div>
              <div className="rounded-lg border border-white/[0.04] bg-white/[0.01] px-3 py-2"><div className="text-muted-foreground">训练轮次</div><div className="mt-0.5 font-mono text-foreground">{hp.total_epochs}</div></div>
              <div className="rounded-lg border border-white/[0.04] bg-white/[0.01] px-3 py-2"><div className="text-muted-foreground">随机种子</div><div className="mt-0.5 font-mono text-foreground">{hp.random_seed}</div></div>
              <div className="rounded-lg border border-white/[0.04] bg-white/[0.01] px-3 py-2"><div className="text-muted-foreground">损失函数</div><div className="mt-0.5 text-foreground">{hp.loss_function || '-'}</div></div>
              {(hp as any).val_split != null && (
                <div className="rounded-lg border border-white/[0.04] bg-white/[0.01] px-3 py-2"><div className="text-muted-foreground">验证集比例</div><div className="mt-0.5 font-mono text-foreground">{((hp as any).val_split * 100).toFixed(0)}%</div></div>
              )}
            </div>
          </section>

          <section className="rounded-xl border border-white/[0.06] bg-[#0f1119] p-5 lg:col-span-2">
            <h2 className="mb-4 flex items-center gap-2 text-sm font-bold">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary"><rect x="3" y="3" width="7" height="7" /><rect x="14" y="3" width="7" height="7" /><rect x="14" y="14" width="7" height="7" /><rect x="3" y="14" width="7" height="7" /></svg>
              模型配置
            </h2>
            <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div className="rounded-lg border border-white/[0.04] bg-white/[0.01] px-3 py-2"><div className="text-xs text-muted-foreground">模型类型</div><div className="mt-0.5 text-sm text-foreground">{mc.model_type || detail.model_name || '-'}</div></div>
              <div className="rounded-lg border border-white/[0.04] bg-white/[0.01] px-3 py-2"><div className="text-xs text-muted-foreground">总参数量</div><div className="mt-0.5 font-mono text-sm text-foreground">{formatNumber(mc?.total_params ?? 0)}</div></div>
              <div className="rounded-lg border border-white/[0.04] bg-white/[0.01] px-3 py-2"><div className="text-xs text-muted-foreground">总层数</div><div className="mt-0.5 font-mono text-sm text-foreground">{mc.total_layers}</div></div>
              <div className="rounded-lg border border-white/[0.04] bg-white/[0.01] px-3 py-2"><div className="text-xs text-muted-foreground">输入 / 输出</div><div className="mt-0.5 font-mono text-xs text-foreground">{formatShape(mc.input_shape)} → {formatShape(mc.output_shape)}</div></div>
            </div>
            {detail.layers && detail.layers.length > 0 ? (
              <div className="overflow-hidden rounded-lg border border-white/[0.04]">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-white/[0.06] bg-white/[0.01]">
                      <th className="px-3 py-2 text-left font-semibold text-muted-foreground">层名称</th>
                      <th className="px-3 py-2 text-left font-semibold text-muted-foreground">类型</th>
                      <th className="px-3 py-2 text-right font-semibold text-muted-foreground">参数量</th>
                      <th className="px-3 py-2 text-left font-semibold text-muted-foreground">输入形状</th>
                      <th className="px-3 py-2 text-left font-semibold text-muted-foreground">输出形状</th>
                      <th className="px-3 py-2 text-left font-semibold text-muted-foreground">激活函数</th>
                    </tr>
                  </thead>
                  <tbody>
                    {detail.layers.map((layer, i) => (
                      <tr key={i} className="border-b border-white/[0.04] transition-colors hover:bg-white/[0.02]">
                        <td className="px-3 py-2 font-medium text-foreground">{layer.name}</td>
                        <td className="px-3 py-2 text-muted-foreground">{layer.type}</td>
                        <td className="px-3 py-2 text-right font-mono text-muted-foreground">{formatNumber(layer.params)}</td>
                        <td className="px-3 py-2 font-mono text-muted-foreground">{formatShape(layer.input_shape)}</td>
                        <td className="px-3 py-2 font-mono text-muted-foreground">{formatShape(layer.output_shape)}</td>
                        <td className="px-3 py-2 text-muted-foreground">{layer.activation || '-'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <p className="py-4 text-center text-xs text-muted-foreground">暂无层详情数据</p>}
          </section>

          {/* 训练指标 — 多Tab图表 */}
          <section className="rounded-xl border border-white/[0.06] bg-[#0f1119] p-5 lg:col-span-2">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="flex items-center gap-2 text-sm font-bold">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary"><path d="M3 3v18h18" /><path d="M7 14l4-6 4 3 5-8" /></svg>
                训练指标
              </h2>
            </div>

            <div className="mb-4 grid grid-cols-2 gap-3 sm:grid-cols-4">
              <div className="rounded-lg border border-emerald-400/20 bg-emerald-400/[0.04] px-3 py-2"><div className="text-xs text-muted-foreground">最佳准确率</div><div className="mt-0.5 font-mono text-sm font-semibold text-emerald-400">{((ms?.best_accuracy ?? 0) * 100).toFixed(2)}%</div></div>
              <div className="rounded-lg border border-red-400/20 bg-red-400/[0.04] px-3 py-2"><div className="text-xs text-muted-foreground">最终 Loss</div><div className="mt-0.5 font-mono text-sm font-semibold text-red-400">{(ms?.final_loss ?? 0).toFixed(4)}</div></div>
              <div className="rounded-lg border border-primary/20 bg-primary/[0.04] px-3 py-2"><div className="text-xs text-muted-foreground">最佳 Epoch</div><div className="mt-0.5 font-mono text-sm font-semibold text-primary">{ms?.best_epoch != null ? ms.best_epoch : '-'}</div></div>
              <div className="rounded-lg border border-amber-400/20 bg-amber-400/[0.04] px-3 py-2"><div className="text-xs text-muted-foreground">训练耗时</div><div className="mt-0.5 font-mono text-sm font-semibold text-amber-400">{formatDuration(ms?.training_duration ?? null)}</div></div>
            </div>

            {/* Tab 切换 */}
            {hasData ? (
              <>
                <div className="mb-3 flex gap-1 rounded-lg border border-white/[0.06] bg-[#0c0e17] p-1">
                  {CHART_TABS.map((tab) => (
                    <button key={tab.key} onClick={() => setActiveTab(tab.key)}
                      className={`flex flex-1 items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-all ${
                        activeTab === tab.key ? 'bg-primary/15 text-primary' : 'text-muted-foreground hover:text-foreground'
                      }`}>
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d={tab.icon} />
                      </svg>
                      {tab.label}
                    </button>
                  ))}
                </div>
                <div className="rounded-lg border border-white/[0.04] bg-white p-2">
                  <div style={{ height: activeTab === 'confusion' ? '380px' : '340px' }}>
                    <ReactECharts
                      ref={chartRef}
                      option={currentChartOption}
                      style={{ height: '100%', width: '100%' }}
                      opts={{ renderer: 'canvas' }}
                      notMerge={true}
                    />
                  </div>
                  <div className="mt-2 flex items-center justify-between px-2 text-[10px] text-gray-400">
                    <span>
                      {activeTab === 'curves' && `共 ${trainingStepData.length} 个 epoch · 白色背景适合直接粘贴到论文/报告`}
                      {activeTab === 'confusion' && `基于最后一个epoch (epoch ${lastEpoch?.epoch}) 的验证集结果`}
                      {activeTab === 'perclass' && '每类 Precision/Recall/F1 — 类别性能诊断'}
                      {activeTab === 'gradients' && '梯度范数和权重范数变化趋势 — 诊断训练稳定性'}
                    </span>
                    <span>使用"科研导出"按钮可下载 SVG 矢量图</span>
                  </div>
                </div>
              </>
            ) : (
              <div className="flex h-[280px] flex-col items-center justify-center rounded-lg border border-white/[0.04] bg-[#0c0e17]">
                <svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="mb-2 text-muted-foreground/30">
                  <path d="M3 3v18h18" /><path d="M7 14l4-6 4 3 5-8" />
                </svg>
                <p className="text-xs text-muted-foreground/70">暂无训练时序数据</p>
                <p className="mt-1 text-[10px] text-muted-foreground/50">如果实验已完成训练但看不到数据，可能是旧版本实验数据格式不兼容，重新训练即可生成完整数据</p>
              </div>
            )}
          </section>

          {/* 训练日志 */}
          <section className="rounded-xl border border-white/[0.06] bg-[#0f1119] p-5 lg:col-span-2">
            <h2 className="mb-4 flex items-center gap-2 text-sm font-bold">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" /><path d="M14 2v6h6M16 13H8M16 17H8M10 9H8" /></svg>
              训练日志
              {detail.training_logs?.length > 0 && <span className="text-[10px] text-muted-foreground">({detail.training_logs.length} 条)</span>}
            </h2>
            {detail.training_logs && detail.training_logs.length > 0 ? (
              <div className="max-h-[400px] overflow-y-auto rounded-lg border border-white/[0.04] bg-[#0c0e17] p-3">
                <div className="space-y-1 font-mono text-xs">
                  {detail.training_logs.map((log: any, i: number) => {
                    const text = typeof log === 'string' ? log : JSON.stringify(log);
                    const isError = /error|fail|失败|错误/i.test(text);
                    const isSuccess = /完成|best|新最佳|saved/i.test(text);
                    return (
                      <div key={i} className={`flex gap-2 border-b border-white/[0.02] py-1 last:border-0 ${
                        isError ? 'text-red-400' : isSuccess ? 'text-emerald-400' : 'text-muted-foreground/80'
                      }`}>
                        <span className="shrink-0 text-muted-foreground/40">{String(i + 1).padStart(3, '0')}</span>
                        <span>{text}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              <div className="flex h-[120px] flex-col items-center justify-center rounded-lg border border-white/[0.04] bg-[#0c0e17]">
                <p className="text-xs text-muted-foreground/70">暂无训练日志</p>
                <p className="mt-1 text-[10px] text-muted-foreground/50">训练过程中的分步日志将在此显示</p>
              </div>
            )}
          </section>
        </div>
        </div>
      </div>

      <DatasetDetailModal datasetId={datasetModalId} onClose={() => setDatasetModalId(null)} />
    </div>
  );
}
