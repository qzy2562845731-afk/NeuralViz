import { useState, useEffect, useRef } from 'react';
import ReactECharts from 'echarts-for-react';
import { apiService, type DatasetData } from '../../services/api';
import { useToast } from '../../contexts/ToastContext';

/* ============================================
   DatasetDetailModal — 数据集详情弹窗
   - 展示完整信息：基本信息、样本统计、类别分布
   - 类别分布用 ECharts 柱状图可视化
   - 支持失败状态重新解析
   ============================================ */

interface DatasetDetailModalProps {
  datasetId: string | null;
  onClose: () => void;
}

// 数据集类型显示名称映射
const DATASET_TYPE_LABELS: Record<string, string> = {
  mnist_idx: 'MNIST IDX 二进制',
  numpy: 'NumPy 数组',
  csv: 'CSV/TSV 表格',
  image_folder: '图片目录',
};

interface PreviewSample {
  image: string;
  label: string | number;
  index: number;
}

interface PreviewData {
  info: DatasetData;
  stats: {
    sample_count: number;
    class_count: number;
    feature_shape: string;
    class_distribution: Record<string, number>;
    pixel_mean: number;
    pixel_std: number;
    value_range: [number, number];
  };
  samples: PreviewSample[];
}

export function DatasetDetailModal({ datasetId, onClose }: DatasetDetailModalProps) {
  const toast = useToast();
  const modalRef = useRef<HTMLDivElement>(null);
  const [dataset, setDataset] = useState<DatasetData | null>(null);
  const [loading, setLoading] = useState(false);
  const [reparsing, setReparsing] = useState(false);
  const [preview, setPreview] = useState<PreviewData | null>(null);
  const [previewLoading, setPreviewLoading] = useState(false);

  const fetchDataset = () => {
    if (!datasetId) {
      setDataset(null);
      setPreview(null);
      return;
    }
    setLoading(true);
    setPreview(null);
    apiService
      .getDataset(datasetId)
      .then((res) => {
        if (res.code === 200) setDataset(res.data);
      })
      .catch(() => setDataset(null))
      .finally(() => setLoading(false));
  };

  const loadPreview = () => {
    if (!datasetId || !dataset || dataset.status !== 'ready') return;
    setPreviewLoading(true);
    apiService
      .getDatasetPreview(datasetId, 16)
      .then((res) => {
        if (res.code === 200) setPreview(res.data);
      })
      .catch((err) => {
        toast.showError('预览加载失败', err.message);
      })
      .finally(() => setPreviewLoading(false));
  };

  useEffect(() => {
    fetchDataset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetId]);

  // 轮询：processing 状态时自动刷新
  useEffect(() => {
    if (!dataset || dataset.status !== 'processing') return;
    const timer = setInterval(fetchDataset, 2000);
    return () => clearInterval(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataset?.status, datasetId]);

  const handleReparse = async () => {
    if (!dataset) return;
    setReparsing(true);
    try {
      await apiService.reparseDataset(dataset.dataset_id);
      toast.showSuccess('重新解析已启动', '正在后台重新解析数据集');
      fetchDataset();
    } catch (err: any) {
      toast.showError('重试解析失败', err.message);
    } finally {
      setReparsing(false);
    }
  };

  // ESC 关闭
  useEffect(() => {
    if (!datasetId) return;
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', handleEsc);
    return () => document.removeEventListener('keydown', handleEsc);
  }, [datasetId, onClose]);

  // 点击外部关闭
  useEffect(() => {
    if (!datasetId) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (modalRef.current && !modalRef.current.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [datasetId, onClose]);

  if (!datasetId) return null;

  const classEntries = dataset?.class_distribution
    ? Object.entries(dataset.class_distribution).sort((a, b) => b[1] - a[1])
    : [];

  const chartOption = {
    grid: { left: '8%', right: '5%', bottom: '10%', top: '8%' },
    tooltip: {
      trigger: 'axis',
      backgroundColor: '#0f1119',
      borderColor: 'rgba(255,255,255,0.08)',
      textStyle: { color: '#e4e4e7', fontSize: 12 },
    },
    xAxis: {
      type: 'category',
      data: classEntries.map(([name]) => name.length > 12 ? name.slice(0, 10) + '...' : name),
      axisLabel: { color: '#71717a', fontSize: 10, rotate: classEntries.length > 8 ? 35 : 0 },
      axisLine: { lineStyle: { color: 'rgba(255,255,255,0.08)' } },
    },
    yAxis: {
      type: 'value',
      axisLabel: { color: '#71717a', fontSize: 10 },
      splitLine: { lineStyle: { color: 'rgba(255,255,255,0.04)' } },
    },
    series: [
      {
        type: 'bar',
        data: classEntries.map(([, count]) => count),
        itemStyle: {
          color: {
            type: 'linear',
            x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: 'rgba(16,185,129,0.8)' },
              { offset: 1, color: 'rgba(16,185,129,0.2)' },
            ],
          },
          borderRadius: [3, 3, 0, 0],
        },
        barMaxWidth: 40,
      },
    ],
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div
        ref={modalRef}
        className="flex max-h-[85vh] w-full max-w-2xl flex-col overflow-hidden rounded-2xl border border-white/[0.08] bg-[#0f1119] shadow-2xl"
      >
        {/* 头部 */}
        <div className="flex items-center justify-between border-b border-white/[0.06] px-6 py-4">
          <div>
            <h2 className="text-base font-bold">
              {loading ? '加载中...' : dataset?.name || '数据集详情'}
              {dataset && (
                <span className="ml-2 rounded px-1.5 py-0.5 text-[10px] font-medium bg-white/[0.04] text-muted-foreground">
                  {dataset.version}
                </span>
              )}
            </h2>
            <p className="mt-0.5 text-xs text-muted-foreground">
              {dataset ? `ID: ${dataset.dataset_id.slice(0, 8)}...` : ''}
            </p>
          </div>
          <button
            onClick={onClose}
            className="flex h-8 w-8 items-center justify-center rounded-md border border-white/[0.08] bg-white/[0.02] text-muted-foreground transition-all hover:text-foreground"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M18 6 6 18M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* 可滚动内容 */}
        <div className="flex-1 overflow-y-auto p-6">
          {loading ? (
            <div className="flex items-center justify-center py-20">
              <svg className="animate-spin text-muted-foreground" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 12a9 9 0 1 1-6.219-8.56" />
              </svg>
            </div>
          ) : dataset ? (
            <div className="space-y-5">
              {/* 统计卡片网格 */}
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
                <div className="rounded-lg border border-primary/20 bg-primary/[0.04] px-3 py-2">
                  <div className="text-xs text-muted-foreground">总样本数</div>
                  <div className="mt-0.5 font-mono text-sm font-semibold text-primary">{dataset.sample_count.toLocaleString()}</div>
                </div>
                <div className="rounded-lg border border-emerald-400/20 bg-emerald-400/[0.04] px-3 py-2">
                  <div className="text-xs text-muted-foreground">类别数</div>
                  <div className="mt-0.5 font-mono text-sm font-semibold text-emerald-400">{dataset.class_count}</div>
                </div>
                <div className="rounded-lg border border-amber-400/20 bg-amber-400/[0.04] px-3 py-2">
                  <div className="text-xs text-muted-foreground">特征/尺寸</div>
                  <div className="mt-0.5 font-mono text-sm font-semibold text-amber-400">{dataset.feature_shape || dataset.image_size || '-'}</div>
                </div>
                <div className="rounded-lg border border-purple-400/20 bg-purple-400/[0.04] px-3 py-2">
                  <div className="text-xs text-muted-foreground">数据集类型</div>
                  <div className="mt-0.5 text-xs font-semibold text-purple-400">
                    {dataset.dataset_type ? (DATASET_TYPE_LABELS[dataset.dataset_type] || dataset.dataset_type) : '-'}
                  </div>
                </div>
                <div className="rounded-lg border border-blue-400/20 bg-blue-400/[0.04] px-3 py-2">
                  <div className="text-xs text-muted-foreground">状态</div>
                  <div className="mt-0.5 text-sm font-semibold text-blue-400">
                    {dataset.status === 'ready' ? '就绪' : dataset.status === 'processing' ? '解析中' : dataset.status === 'failed' ? '失败' : dataset.status}
                  </div>
                </div>
              </div>

              {/* 失败状态错误信息醒目展示 + 重新解析按钮 */}
              {dataset.status === 'failed' && dataset.error_message && (
                <section className="rounded-xl border border-red-400/30 bg-red-400/[0.08] p-4">
                  <div className="flex items-start gap-3">
                    <svg className="mt-0.5 flex-shrink-0 text-red-400" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10" />
                      <line x1="12" y1="8" x2="12" y2="12" />
                      <line x1="12" y1="16" x2="12.01" y2="16" />
                    </svg>
                    <div className="flex-1">
                      <h3 className="text-sm font-semibold text-red-400">解析失败</h3>
                      <p className="mt-1 text-xs text-red-300/90 break-all">{dataset.error_message}</p>
                      <button
                        onClick={handleReparse}
                        disabled={reparsing}
                        className="mt-3 flex items-center gap-1.5 rounded-md border border-amber-400/30 bg-amber-400/10 px-3 py-1.5 text-xs font-semibold text-amber-400 transition-all hover:bg-amber-400/15 disabled:opacity-50"
                      >
                        {reparsing ? (
                          <>
                            <svg className="animate-spin" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                              <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                            </svg>
                            重新解析中...
                          </>
                        ) : (
                          <>
                            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                              <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                              <path d="M3 3v5h5" />
                            </svg>
                            重新解析
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                </section>
              )}

              {/* 基本信息 */}
              <section className="rounded-xl border border-white/[0.06] bg-[#0c0e17] p-4">
                <h3 className="mb-3 text-xs font-bold uppercase tracking-wider text-muted-foreground">基本信息</h3>
                <dl className="space-y-2 text-xs">
                  <div className="flex">
                    <dt className="w-24 text-muted-foreground">描述</dt>
                    <dd className="flex-1 text-foreground">{dataset.description || '-'}</dd>
                  </div>
                  <div className="flex">
                    <dt className="w-24 text-muted-foreground">标签</dt>
                    <dd className="flex-1">
                      {dataset.tags.length > 0 ? (
                        <div className="flex flex-wrap gap-1">
                          {dataset.tags.map((tag, i) => (
                            <span key={i} className="rounded px-1.5 py-0.5 text-[10px] bg-white/[0.04] text-muted-foreground">{tag}</span>
                          ))}
                        </div>
                      ) : '-'}
                    </dd>
                  </div>
                  <div className="flex">
                    <dt className="w-24 text-muted-foreground">文件哈希</dt>
                    <dd className="flex-1 font-mono text-[10px] text-foreground/70">{dataset.file_hash?.slice(0, 24) || '-'}...</dd>
                  </div>
                  <div className="flex">
                    <dt className="w-24 text-muted-foreground">创建时间</dt>
                    <dd className="flex-1 text-foreground">{new Date(dataset.created_at).toLocaleString('zh-CN')}</dd>
                  </div>
                </dl>
              </section>

              {/* 类别分布图 */}
              {classEntries.length > 0 && (
                <section className="rounded-xl border border-white/[0.06] bg-[#0c0e17] p-4">
                  <h3 className="mb-3 text-xs font-bold uppercase tracking-wider text-muted-foreground">类别分布</h3>
                  <ReactECharts option={chartOption} style={{ height: '240px' }} />
                </section>
              )}

              {/* 类别列表 */}
              {classEntries.length > 0 && (
                <section className="rounded-xl border border-white/[0.06] bg-[#0c0e17] p-4">
                  <h3 className="mb-3 text-xs font-bold uppercase tracking-wider text-muted-foreground">类别明细</h3>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                    {classEntries.map(([name, count]) => (
                      <div key={name} className="flex items-center justify-between rounded-lg border border-white/[0.04] bg-white/[0.02] px-3 py-1.5">
                        <span className="truncate text-xs text-foreground">{name}</span>
                        <span className="ml-2 font-mono text-xs font-semibold text-muted-foreground">{count}</span>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {/* 数据集预览 - 样本图片 */}
              {dataset.status === 'ready' && (
                <section className="rounded-xl border border-white/[0.06] bg-[#0c0e17] p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <h3 className="text-xs font-bold uppercase tracking-wider text-muted-foreground">数据预览</h3>
                    <button
                      onClick={loadPreview}
                      disabled={previewLoading}
                      className="flex items-center gap-1 rounded-md border border-white/[0.08] bg-white/[0.02] px-2.5 py-1 text-[10px] font-medium text-muted-foreground transition-all hover:bg-primary/10 hover:text-primary hover:border-primary/20 disabled:opacity-50"
                    >
                      {previewLoading ? (
                        <>
                          <svg className="animate-spin" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                            <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                          </svg>
                          加载中
                        </>
                      ) : preview ? (
                        <>
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                            <path d="M3 3v5h5" />
                          </svg>
                          刷新
                        </>
                      ) : (
                        <>
                          <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                            <circle cx="12" cy="12" r="3" />
                          </svg>
                          查看样本
                        </>
                      )}
                    </button>
                  </div>

                  {preview && preview.stats && (
                    <div className="mb-3 grid grid-cols-3 gap-2">
                      <div className="rounded-lg border border-white/[0.04] bg-white/[0.02] px-2 py-1.5">
                        <div className="text-[10px] text-muted-foreground">像素均值</div>
                        <div className="font-mono text-xs font-semibold text-foreground">{preview.stats.pixel_mean.toFixed(2)}</div>
                      </div>
                      <div className="rounded-lg border border-white/[0.04] bg-white/[0.02] px-2 py-1.5">
                        <div className="text-[10px] text-muted-foreground">像素标准差</div>
                        <div className="font-mono text-xs font-semibold text-foreground">{preview.stats.pixel_std.toFixed(2)}</div>
                      </div>
                      <div className="rounded-lg border border-white/[0.04] bg-white/[0.02] px-2 py-1.5">
                        <div className="text-[10px] text-muted-foreground">值域范围</div>
                        <div className="font-mono text-xs font-semibold text-foreground">[{preview.stats.value_range[0].toFixed(0)}, {preview.stats.value_range[1].toFixed(0)}]</div>
                      </div>
                    </div>
                  )}

                  {preview && preview.samples && preview.samples.length > 0 && (
                    <div className="grid grid-cols-4 gap-2 sm:grid-cols-8">
                      {preview.samples.map((s, i) => (
                        <div key={i} className="group relative overflow-hidden rounded-lg border border-white/[0.06] bg-black/30">
                          <img
                            src={s.image}
                            alt={`sample-${s.index}`}
                            className="aspect-square w-full object-contain p-1 transition-transform group-hover:scale-110"
                            style={{ imageRendering: 'pixelated' }}
                          />
                          <div className="absolute bottom-0 left-0 right-0 bg-black/70 px-1 py-0.5 text-center">
                            <span className="font-mono text-[10px] font-semibold text-white">{s.label}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  {!preview && !previewLoading && (
                    <div className="flex items-center justify-center rounded-lg border border-dashed border-white/[0.08] py-8">
                      <p className="text-xs text-muted-foreground">点击「查看样本」加载数据集预览</p>
                    </div>
                  )}
                </section>
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-20">
              <p className="text-sm text-muted-foreground">数据集不存在或已删除</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
