import { useState, useEffect, useMemo, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import ReactECharts from 'echarts-for-react';
import { apiService, type ExperimentData } from '../services/api';
import { useToast } from '../contexts/ToastContext';

/* ============================================
   Mock数据 - 当API不可用时使用
   ============================================ */
const MOCK_EXPERIMENTS: ExperimentData[] = [
  {
    experiment_id: 'exp_001',
    name: 'ResNet50 图像分类训练',
    description: '使用ResNet50在CIFAR-10数据集上进行图像分类训练',
    model_id: 'resnet50',
    model_name: 'ResNet50',
    model_architecture: { type: 'resnet', depth: 50 },
    status: 'completed',
    total_params: 25600000,
    layer_count: 50,
    best_accuracy: 0.9523,
    final_loss: 0.1567,
    total_epochs: 100,
    current_step: 100,
    hyperparams: { learning_rate: 0.001, batch_size: 64 },
    config: {},
    tags: ['图像分类', 'ResNet', 'CIFAR-10'],
    created_at: '2024-03-15T10:30:00Z',
    updated_at: '2024-03-15T18:45:00Z',
  },
  {
    experiment_id: 'exp_002',
    name: 'BERT 文本分类实验',
    description: '使用BERT进行情感分析任务',
    model_id: 'bert-base',
    model_name: 'BERT-base',
    model_architecture: { type: 'transformer', layers: 12 },
    status: 'running',
    total_params: 110000000,
    layer_count: 12,
    best_accuracy: 0.8934,
    final_loss: 0.2876,
    total_epochs: 50,
    current_step: 35,
    hyperparams: { learning_rate: 0.00005, batch_size: 32 },
    config: {},
    tags: ['NLP', 'BERT', '情感分析'],
    created_at: '2024-03-18T08:00:00Z',
    updated_at: '2024-03-18T14:20:00Z',
  },
  {
    experiment_id: 'exp_003',
    name: 'YOLO 目标检测',
    description: '使用YOLOv8进行实时目标检测训练',
    model_id: 'yolov8',
    model_name: 'YOLOv8m',
    model_architecture: { type: 'yolo', version: 'v8' },
    status: 'running',
    total_params: 25900000,
    layer_count: 168,
    best_accuracy: 0.8756,
    final_loss: 0.3421,
    total_epochs: 200,
    current_step: 78,
    hyperparams: { learning_rate: 0.01, batch_size: 16 },
    config: {},
    tags: ['目标检测', 'YOLO', '计算机视觉'],
    created_at: '2024-03-20T09:15:00Z',
    updated_at: '2024-03-20T16:30:00Z',
  },
  {
    experiment_id: 'exp_004',
    name: 'LSTM 时间序列预测',
    description: '使用LSTM进行股票价格预测',
    model_id: 'lstm_ts',
    model_name: 'LSTM-256',
    model_architecture: { type: 'lstm', hidden_size: 256 },
    status: 'completed',
    total_params: 1350000,
    layer_count: 3,
    best_accuracy: 0.8234,
    final_loss: 0.2134,
    total_epochs: 80,
    current_step: 80,
    hyperparams: { learning_rate: 0.001, batch_size: 128 },
    config: {},
    tags: ['时间序列', 'LSTM', '金融'],
    created_at: '2024-03-10T11:00:00Z',
    updated_at: '2024-03-10T19:30:00Z',
  },
  {
    experiment_id: 'exp_005',
    name: 'GAN 图像生成',
    description: '使用DCGAN生成手写数字',
    model_id: 'dcgan',
    model_name: 'DCGAN',
    model_architecture: { type: 'gan', latent_dim: 100 },
    status: 'failed',
    total_params: 8500000,
    layer_count: 8,
    best_accuracy: 0.0,
    final_loss: 0.0,
    total_epochs: 50,
    current_step: 23,
    hyperparams: { learning_rate: 0.0002, batch_size: 64 },
    config: {},
    tags: ['GAN', '图像生成', '无监督学习'],
    created_at: '2024-03-12T14:00:00Z',
    updated_at: '2024-03-12T17:45:00Z',
  },
  {
    experiment_id: 'exp_006',
    name: 'CNN 手写数字识别',
    description: '使用LeNet进行MNIST手写数字识别',
    model_id: 'lenet5',
    model_name: 'LeNet-5',
    model_architecture: { type: 'lenet', layers: 7 },
    status: 'completed',
    total_params: 60000,
    layer_count: 7,
    best_accuracy: 0.9912,
    final_loss: 0.0287,
    total_epochs: 20,
    current_step: 20,
    hyperparams: { learning_rate: 0.01, batch_size: 256 },
    config: {},
    tags: ['MNIST', 'LeNet', '入门'],
    created_at: '2024-03-08T10:00:00Z',
    updated_at: '2024-03-08T12:30:00Z',
  },
  {
    experiment_id: 'exp_007',
    name: 'Transformer 机器翻译',
    description: '使用Transformer进行中英翻译',
    model_id: 'transformer',
    model_name: 'Transformer-Base',
    model_architecture: { type: 'transformer', layers: 6 },
    status: 'paused',
    total_params: 65000000,
    layer_count: 6,
    best_accuracy: 0.7845,
    final_loss: 0.4567,
    total_epochs: 100,
    current_step: 45,
    hyperparams: { learning_rate: 0.0001, batch_size: 48 },
    config: {},
    tags: ['NLP', 'Transformer', '机器翻译'],
    created_at: '2024-03-14T08:30:00Z',
    updated_at: '2024-03-16T11:20:00Z',
  },
  {
    experiment_id: 'exp_008',
    name: 'VGG16 花卉分类',
    description: '使用VGG16进行花卉种类识别',
    model_id: 'vgg16',
    model_name: 'VGG16',
    model_architecture: { type: 'vgg', depth: 16 },
    status: 'draft',
    total_params: 138000000,
    layer_count: 16,
    best_accuracy: 0.0,
    final_loss: 0.0,
    total_epochs: 0,
    current_step: 0,
    hyperparams: { learning_rate: 0.001, batch_size: 32 },
    config: {},
    tags: ['图像分类', 'VGG', '花卉识别'],
    created_at: '2024-03-19T15:00:00Z',
    updated_at: '2024-03-19T15:00:00Z',
  },
];

const STATUS_LABELS: Record<string, { label: string; color: string }> = {
  draft: { label: '草稿', color: 'bg-gray-400/20 text-gray-400' },
  running: { label: '训练中', color: 'bg-blue-400/20 text-blue-400' },
  completed: { label: '已完成', color: 'bg-emerald-400/20 text-emerald-400' },
  failed: { label: '失败', color: 'bg-red-400/20 text-red-400' },
  paused: { label: '已暂停', color: 'bg-amber-400/20 text-amber-400' },
};

function formatNumber(num: number | null | undefined): string {
  if (num == null || isNaN(num)) return '-';
  if (num >= 1e6) return (num / 1e6).toFixed(2) + 'M';
  if (num >= 1e3) return (num / 1e3).toFixed(1) + 'K';
  return num.toLocaleString();
}

function formatPct(value: number | null | undefined, digits: number = 1): string {
  if (value == null || isNaN(value)) return '-';
  return (value * 100).toFixed(digits) + '%';
}

function formatLoss(value: number | null | undefined, digits: number = 4): string {
  if (value == null || isNaN(value)) return '-';
  return value.toFixed(digits);
}

function formatDate(dateStr: string): string {
  if (!dateStr) return '-';
  const d = new Date(dateStr);
  if (isNaN(d.getTime())) return '-';
  return d.toLocaleString('zh-CN', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export default function ExperimentsPage() {
  const navigate = useNavigate();
  const toast = useToast();
  const [experiments, setExperiments] = useState<ExperimentData[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [showCompare, setShowCompare] = useState(false);
  const [compareData, setCompareData] = useState<ExperimentData[]>([]);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [exportingCSV, setExportingCSV] = useState(false);
  const pageSize = 20;
  const compareModalRef = useRef<HTMLDivElement>(null);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // 搜索防抖
  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(search);
    }, 300);
    return () => clearTimeout(timer);
  }, [search]);

  // 点击弹窗外部关闭对比弹窗
  useEffect(() => {
    if (!showCompare) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (compareModalRef.current && !compareModalRef.current.contains(e.target as Node)) {
        setShowCompare(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [showCompare]);

  // ESC键关闭弹窗
  useEffect(() => {
    if (!showCompare) return;
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setShowCompare(false);
    };
    document.addEventListener('keydown', handleEsc);
    return () => document.removeEventListener('keydown', handleEsc);
  }, [showCompare]);

  const fetchExperiments = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiService.listExperiments({
        page,
        page_size: pageSize,
        status: statusFilter === 'all' ? undefined : statusFilter,
        search: debouncedSearch || undefined,
      });
      setExperiments(res.data.items);
      setTotal(res.data.total);
    } catch (err: any) {
      // API不可用时使用Mock数据
      console.warn('API不可用，使用Mock数据:', err.message);
      let filteredData = [...MOCK_EXPERIMENTS];
      
      // 应用状态筛选
      if (statusFilter !== 'all') {
        filteredData = filteredData.filter(e => e.status === statusFilter);
      }
      
      // 应用搜索筛选
      if (debouncedSearch) {
        const searchLower = debouncedSearch.toLowerCase();
        filteredData = filteredData.filter(e => 
          e.name.toLowerCase().includes(searchLower) ||
          (e.description && e.description.toLowerCase().includes(searchLower)) ||
          (e.tags && e.tags.some(t => t.toLowerCase().includes(searchLower)))
        );
      }
      
      setExperiments(filteredData);
      setTotal(filteredData.length);
      setError(null); // 不显示错误，因为有Mock数据
    } finally {
      setLoading(false);
    }
  }, [page, statusFilter, debouncedSearch]);

  useEffect(() => {
    fetchExperiments();
  }, [fetchExperiments]);

  // 自动轮询：有运行中的实验时，每5秒刷新状态，确保与消融实验/对比实验状态同步
  useEffect(() => {
    const hasRunning = experiments.some(e => e.status === 'running');
    if (hasRunning) {
      pollTimerRef.current = setInterval(() => {
        fetchExperiments();
      }, 5000);
    }
    return () => {
      if (pollTimerRef.current) {
        clearInterval(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [experiments, fetchExperiments]);

  const handleToggleSelect = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const handleSelectAll = () => {
    if (selectedIds.size === experiments.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(experiments.map((e) => e.experiment_id)));
    }
  };

  const handleCompare = async () => {
    if (selectedIds.size < 2) return;
    try {
      const res = await apiService.getExperimentsSummary(Array.from(selectedIds));
      // 确保返回的数据是数组
      const summaryData = Array.isArray(res.data) ? res.data : (res.data as any)?.items || [];
      if (summaryData.length >= 2) {
        setCompareData(summaryData);
        setShowCompare(true);
      } else {
        // API返回数据不足，使用Mock数据
        const selected = MOCK_EXPERIMENTS.filter(e => selectedIds.has(e.experiment_id));
        if (selected.length >= 2) {
          setCompareData(selected);
          setShowCompare(true);
        } else {
          setError('未能获取足够的对比数据');
        }
      }
    } catch {
      // API不可用时从Mock数据中获取对比数据
      const selected = MOCK_EXPERIMENTS.filter(e => selectedIds.has(e.experiment_id));
      if (selected.length >= 2) {
        setCompareData(selected);
        setShowCompare(true);
      } else {
        setError('API不可用，且Mock数据中未找到足够的实验');
      }
    }
  };

  const handleDelete = async (id: string, name: string) => {
    if (!confirm(`确定要删除实验「${name}」吗？`)) return;
    try {
      await apiService.deleteExperiment(id);
    } catch {
      // API不可用时从本地状态中删除（Mock模式）
      console.warn('API不可用，从本地列表中删除');
    }
    // 无论API是否成功，都从本地列表中移除
    setExperiments(prev => prev.filter(e => e.experiment_id !== id));
    setTotal(prev => prev - 1);
    setSelectedIds((prev) => {
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
  };

  const handleBatchDelete = async () => {
    if (selectedIds.size === 0) return;
    if (!confirm(`确定要删除选中的 ${selectedIds.size} 个实验吗？此操作不可撤销。`)) return;
    const ids = Array.from(selectedIds);
    try {
      const res = await apiService.batchDeleteExperiments({ experimentIds: ids });
      toast.showSuccess(`成功删除 ${res.data?.deleted ?? ids.length} 个实验`);
    } catch (err: any) {
      toast.showError('批量删除失败', err.message);
      // API不可用，仍从本地删除
      console.warn('API不可用，从本地列表中删除');
    }
    setExperiments(prev => prev.filter(e => !selectedIds.has(e.experiment_id)));
    setTotal(prev => prev - selectedIds.size);
    setSelectedIds(new Set());
  };

  const handleDeleteAll = async () => {
    if (!confirm(`确定要删除全部 ${total} 个实验吗？此操作不可撤销！\n\n提示：如需保留部分实验，请先取消全选，再使用多选删除。`)) return;
    try {
      const res = await apiService.batchDeleteExperiments({ deleteAll: true });
      toast.showSuccess(`成功删除 ${res.data?.deleted ?? total} 个实验`);
    } catch (err: any) {
      toast.showError('全部删除失败', err.message);
      console.warn('API不可用，从本地列表中删除');
    }
    setExperiments([]);
    setTotal(0);
    setSelectedIds(new Set());
  };

  const totalPages = Math.ceil(total / pageSize);

  const barChartOption = useMemo(() => {
    if (!compareData.length) return {};
    const colors = ['#22d3ee', '#a78bfa', '#34d399', '#fbbf24', '#f87171'];

    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        backgroundColor: 'rgba(15, 17, 25, 0.95)',
        borderColor: 'rgba(255,255,255,0.1)',
        textStyle: { color: '#fff', fontSize: 12 },
        appendToBody: true,
      },
      legend: {
        data: compareData.map((e) => e.name),
        textStyle: { color: '#94a3b8', fontSize: 11 },
        top: 0,
      },
      grid: { left: '3%', right: '4%', bottom: '3%', top: '15%', containLabel: true },
      xAxis: {
        type: 'category',
        data: ['参数量(K)', '层数', '准确率(%)', 'Loss', 'Epoch'],
        axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
        axisLabel: { color: '#94a3b8', fontSize: 11 },
      },
      yAxis: {
        type: 'value',
        axisLine: { lineStyle: { color: 'rgba(255,255,255,0.1)' } },
        axisLabel: { color: '#94a3b8', fontSize: 11 },
        splitLine: { lineStyle: { color: 'rgba(255,255,255,0.05)' } },
      },
      series: compareData.map((exp, idx) => ({
        name: exp.name,
        type: 'bar',
        data: [
          (exp.total_params || 0) / 1000,
          exp.layer_count || 0,
          (exp.best_accuracy || 0) * 100,
          exp.final_loss || 0,
          exp.total_epochs || 0,
        ],
        itemStyle: {
          color: colors[idx % colors.length],
          borderRadius: [4, 4, 0, 0],
        },
        barWidth: Math.max(8, 40 / compareData.length),
      })),
    };
  }, [compareData]);

  // 对接后端全量字段导出接口（支持单实验/批量导出）
  const exportCSV = async () => {
    if (!compareData.length) return;
    setExportingCSV(true);
    try {
      const ids = compareData.map((e) => e.experiment_id);
      const blob = await apiService.exportExperimentsCSV(ids);
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      const ts = new Date().toLocaleString('zh-CN', {
        year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit',
      }).replace(/[\/\s:]/g, '');
      // 多实验导出用 comparison 命名，单实验用实验名
      const fileName = compareData.length === 1
        ? `${compareData[0].name}_${ts}.csv`
        : `experiment_comparison_${ts}.csv`;
      link.download = fileName;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      toast.showSuccess('CSV 导出成功');
    } catch (err: any) {
      toast.showError('导出失败', err.message);
    } finally {
      setExportingCSV(false);
    }
  };

  // 跳转到实验详情页
  const handleRowClick = (e: React.MouseEvent, experimentId: string) => {
    // 点击复选框或删除按钮时不跳转
    const target = e.target as HTMLElement;
    if (target.closest('input[type="checkbox"]') || target.closest('button')) {
      return;
    }
    navigate(`/experiments/${experimentId}`);
  };

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-[#0a0c14] text-white">
      {/* 顶部栏 */}
      <div className="flex-shrink-0 border-b border-white/[0.06] bg-[#0c0e17]/95 backdrop-blur-sm">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-primary/60 shadow-lg shadow-primary/20">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-white">
                <path d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
              </svg>
            </div>
            <div>
              <h1 className="text-lg font-bold">我的实验</h1>
              <p className="text-xs text-muted-foreground">共 {total} 个实验</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {selectedIds.size > 0 && (
              <>
                <span className="text-xs text-muted-foreground">
                  已选 {selectedIds.size} 个
                </span>
                <button
                  onClick={() => setSelectedIds(new Set())}
                  className="text-xs text-muted-foreground hover:text-foreground"
                >
                  清除
                </button>
                <button
                  onClick={handleBatchDelete}
                  className="flex items-center gap-1.5 rounded-md border border-red-400/30 bg-red-400/10 px-3 py-1.5 text-xs font-semibold text-red-400 transition-all hover:bg-red-400/15"
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                  </svg>
                  删除选中
                </button>
                <button
                  onClick={handleCompare}
                  disabled={selectedIds.size < 2}
                  className="flex items-center gap-1.5 rounded-md border border-primary/30 bg-primary/10 px-3 py-1.5 text-xs font-semibold text-primary transition-all hover:bg-primary/15 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 3v18h18" />
                    <path d="M7 14l4-6 4 3 5-8" />
                  </svg>
                  对比实验
                </button>
              </>
            )}
            {experiments.length > 0 && (
              <button
                onClick={handleDeleteAll}
                className="flex items-center gap-1.5 rounded-md border border-red-400/20 bg-transparent px-3 py-1.5 text-xs text-red-400/70 transition-all hover:bg-red-400/10 hover:text-red-400"
              >
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2M10 11v6M14 11v6" />
                </svg>
                全部删除
              </button>
            )}
          </div>
        </div>

        {/* 筛选栏 */}
        <div className="mx-auto flex max-w-7xl items-center gap-4 px-6 py-3">
          <div className="relative flex-1 max-w-md">
            <svg
              className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground"
              width="14"
              height="14"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
            >
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.35-4.35" />
            </svg>
            <input
              type="text"
              placeholder="搜索实验名称、描述、标签..."
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setPage(1);
              }}
              className="w-full rounded-lg border border-white/[0.08] bg-white/[0.02] py-2 pl-9 pr-3 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary/40 focus:outline-none"
            />
          </div>

          <div className="flex items-center gap-1 rounded-lg border border-white/[0.08] bg-white/[0.02] p-1">
            {['all', 'draft', 'running', 'completed', 'failed'].map((s) => (
              <button
                key={s}
                onClick={() => {
                  setStatusFilter(s);
                  setPage(1);
                }}
                className={`rounded-md px-3 py-1.5 text-xs font-medium transition-all ${
                  statusFilter === s
                    ? 'bg-primary/15 text-primary'
                    : 'text-muted-foreground hover:text-foreground'
                }`}
              >
                {s === 'all' ? '全部' : STATUS_LABELS[s]?.label || s}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* 主内容 */}
      <div className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-7xl px-6 py-6">
        {error && (
          <div className="mb-4 rounded-lg border border-red-400/20 bg-red-400/10 px-4 py-3 text-sm text-red-400">
            {error}
          </div>
        )}

        {loading ? (
          <div className="flex py-20 items-center justify-center">
            <div className="text-sm text-muted-foreground">加载中...</div>
          </div>
        ) : experiments.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full border border-white/[0.08] bg-white/[0.02]">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-muted-foreground">
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <path d="M14 2v6h6M16 13H8M16 17H8M10 9H8" />
              </svg>
            </div>
            <h3 className="text-sm font-semibold text-foreground">暂无实验</h3>
            <p className="mt-1 text-xs text-muted-foreground">
              在工作台中保存实验后，会在这里显示
            </p>
          </div>
        ) : (
          <>
            {/* 实验列表表格 */}
            <div className="overflow-hidden rounded-xl border border-white/[0.06] bg-[#0f1119]">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/[0.06] bg-white/[0.01]">
                    <th className="w-10 px-4 py-3">
                      <input
                        type="checkbox"
                        checked={selectedIds.size === experiments.length && experiments.length > 0}
                        onChange={handleSelectAll}
                        className="h-4 w-4 cursor-pointer rounded border-white/20 bg-white/5 accent-primary"
                      />
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      实验名称
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      状态
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      模型
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      参数量
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      准确率
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      Loss
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                      创建时间
                    </th>
                    <th className="w-20 px-4 py-3"></th>
                  </tr>
                </thead>
                <tbody>
                  {experiments.map((exp) => {
                    const statusInfo = STATUS_LABELS[exp.status] || {
                      label: exp.status,
                      color: 'bg-gray-400/20 text-gray-400',
                    };
                    return (
                      <tr
                        key={exp.experiment_id}
                        onClick={(e) => handleRowClick(e, exp.experiment_id)}
                        className={`cursor-pointer border-b border-white/[0.04] transition-colors hover:bg-white/[0.04] ${
                          selectedIds.has(exp.experiment_id) ? 'bg-primary/[0.04]' : ''
                        }`}
                      >
                        <td className="px-4 py-3" onClick={(e) => e.stopPropagation()}>
                          <input
                            type="checkbox"
                            checked={selectedIds.has(exp.experiment_id)}
                            onChange={() => handleToggleSelect(exp.experiment_id)}
                            className="h-4 w-4 cursor-pointer rounded border-white/20 bg-white/5 accent-primary"
                          />
                        </td>
                        <td className="px-4 py-3">
                          <div className="font-medium text-foreground transition-colors group-hover:text-primary">
                            {exp.name}
                          </div>
                          {exp.description && (
                            <div className="mt-0.5 truncate text-xs text-muted-foreground max-w-xs">
                              {exp.description}
                            </div>
                          )}
                          {exp.tags && exp.tags.length > 0 && (
                            <div className="mt-1 flex flex-wrap gap-1">
                              {exp.tags.slice(0, 3).map((tag, i) => (
                                <span
                                  key={i}
                                  className="rounded px-1.5 py-0.5 text-[10px] bg-white/[0.04] text-muted-foreground"
                                >
                                  {tag}
                                </span>
                              ))}
                            </div>
                          )}
                        </td>
                        <td className="px-4 py-3">
                          <span
                            className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${statusInfo.color}`}
                          >
                            {statusInfo.label}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-muted-foreground">
                          {exp.model_name || '-'}
                        </td>
                        <td className="px-4 py-3 text-right font-mono text-muted-foreground">
                          {formatNumber(exp.total_params)}
                        </td>
                        <td className="px-4 py-3 text-right font-mono text-emerald-400">
                          {formatPct(exp.best_accuracy, 1)}
                        </td>
                        <td className="px-4 py-3 text-right font-mono text-red-400">
                          {formatLoss(exp.final_loss, 4)}
                        </td>
                        <td className="px-4 py-3 text-xs text-muted-foreground">
                          {formatDate(exp.created_at)}
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center justify-end gap-1">
                            <button
                              onClick={() => handleDelete(exp.experiment_id, exp.name)}
                              className="rounded p-1.5 text-muted-foreground transition-colors hover:bg-red-400/10 hover:text-red-400"
                              title="删除"
                            >
                              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                              </svg>
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* 分页 */}
            {totalPages > 1 && (
              <div className="mt-4 flex items-center justify-center gap-2">
                <button
                  onClick={() => setPage((p) => Math.max(1, p - 1))}
                  disabled={page === 1}
                  className="rounded-md border border-white/[0.08] bg-white/[0.02] px-3 py-1.5 text-xs text-muted-foreground transition-all hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
                >
                  上一页
                </button>
                <span className="text-xs text-muted-foreground">
                  第 {page} / {totalPages} 页
                </span>
                <button
                  onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                  disabled={page === totalPages}
                  className="rounded-md border border-white/[0.08] bg-white/[0.02] px-3 py-1.5 text-xs text-muted-foreground transition-all hover:text-foreground disabled:cursor-not-allowed disabled:opacity-50"
                >
                  下一页
                </button>
              </div>
            )}
          </>
        )}
        </div>
      </div>

      {/* 对比视图弹窗 */}
      {showCompare && compareData.length > 0 && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div ref={compareModalRef} className="flex h-[85vh] w-full max-w-5xl flex-col overflow-hidden rounded-2xl border border-white/[0.08] bg-[#0f1119] shadow-2xl">
            {/* 弹窗头部 */}
            <div className="flex items-center justify-between border-b border-white/[0.06] px-6 py-4">
              <div>
                <h2 className="text-base font-bold">实验对比</h2>
                <p className="text-xs text-muted-foreground">
                  {compareData.length} 个实验 · 核心指标横向对比
                </p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={exportCSV}
                  disabled={exportingCSV}
                  className="flex items-center gap-1.5 rounded-md border border-white/[0.08] bg-white/[0.02] px-3 py-1.5 text-xs font-medium text-muted-foreground transition-all hover:text-foreground disabled:opacity-50"
                >
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                    <path d="M7 10l5 5 5-5" />
                    <path d="M12 15V3" />
                  </svg>
                  {exportingCSV ? '导出中...' : '导出 CSV'}
                </button>
                <button
                  onClick={() => setShowCompare(false)}
                  className="flex h-8 w-8 items-center justify-center rounded-md border border-white/[0.08] bg-white/[0.02] text-muted-foreground transition-all hover:text-foreground"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M18 6 6 18M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>

            {/* 弹窗内容 */}
            <div className="flex-1 overflow-y-auto p-6">
              {/* 柱状图 */}
              <div className="mb-6 rounded-xl border border-white/[0.06] bg-[#0c0e17] p-4">
                <h3 className="mb-3 text-sm font-semibold">指标对比图</h3>
                <div style={{ height: '280px' }}>
                  <ReactECharts
                    option={barChartOption}
                    style={{ height: '100%', width: '100%' }}
                    opts={{ renderer: 'canvas' }}
                  />
                </div>
              </div>

              {/* 对比表格 */}
              <div className="overflow-hidden rounded-xl border border-white/[0.06]">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-white/[0.06] bg-white/[0.01]">
                      <th className="w-32 px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                        指标
                      </th>
                      {compareData.map((exp) => (
                        <th
                          key={exp.experiment_id}
                          className="px-4 py-3 text-left text-xs font-semibold text-foreground"
                        >
                          {exp.name}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr className="border-b border-white/[0.04]">
                      <td className="px-4 py-3 text-xs text-muted-foreground">状态</td>
                      {compareData.map((exp) => {
                        const s = STATUS_LABELS[exp.status] || {
                          label: exp.status,
                          color: 'bg-gray-400/20 text-gray-400',
                        };
                        return (
                          <td key={exp.experiment_id} className="px-4 py-3">
                            <span
                              className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${s.color}`}
                            >
                              {s.label}
                            </span>
                          </td>
                        );
                      })}
                    </tr>
                    <tr className="border-b border-white/[0.04]">
                      <td className="px-4 py-3 text-xs text-muted-foreground">模型</td>
                      {compareData.map((exp) => (
                        <td key={exp.experiment_id} className="px-4 py-3 font-mono text-xs">
                          {exp.model_name || '-'}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-white/[0.04]">
                      <td className="px-4 py-3 text-xs text-muted-foreground">参数量</td>
                      {compareData.map((exp) => (
                        <td key={exp.experiment_id} className="px-4 py-3 font-mono">
                          {formatNumber(exp.total_params)}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-white/[0.04]">
                      <td className="px-4 py-3 text-xs text-muted-foreground">层数</td>
                      {compareData.map((exp) => (
                        <td key={exp.experiment_id} className="px-4 py-3 font-mono">
                          {exp.layer_count} 层
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-white/[0.04]">
                      <td className="px-4 py-3 text-xs text-muted-foreground">最佳准确率</td>
                      {compareData.map((exp) => (
                        <td key={exp.experiment_id} className="px-4 py-3 font-mono text-emerald-400">
                          {formatPct(exp.best_accuracy, 2)}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-white/[0.04]">
                      <td className="px-4 py-3 text-xs text-muted-foreground">最终 Loss</td>
                      {compareData.map((exp) => (
                        <td key={exp.experiment_id} className="px-4 py-3 font-mono text-red-400">
                          {formatLoss(exp.final_loss, 4)}
                        </td>
                      ))}
                    </tr>
                    <tr className="border-b border-white/[0.04]">
                      <td className="px-4 py-3 text-xs text-muted-foreground">总 Epoch</td>
                      {compareData.map((exp) => (
                        <td key={exp.experiment_id} className="px-4 py-3 font-mono">
                          {exp.total_epochs}
                        </td>
                      ))}
                    </tr>
                    <tr>
                      <td className="px-4 py-3 text-xs text-muted-foreground">创建时间</td>
                      {compareData.map((exp) => (
                        <td key={exp.experiment_id} className="px-4 py-3 text-xs text-muted-foreground">
                          {formatDate(exp.created_at)}
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
