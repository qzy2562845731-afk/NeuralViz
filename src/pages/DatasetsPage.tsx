import { useState, useEffect, useCallback } from 'react';
import { apiService, type DatasetData } from '../services/api';
import { useToast } from '../contexts/ToastContext';
import { DatasetUploader } from '../components/dataset/DatasetUploader';
import { DatasetDetailModal } from '../components/dataset/DatasetDetailModal';

/* ============================================
   DatasetsPage — 数据集管理页面
   - 路由：/datasets
   - 列表、搜索、筛选、上传、详情、删除
   ============================================ */

const STATUS_LABELS: Record<string, { label: string; color: string }> = {
  uploading: { label: '上传中', color: 'bg-blue-400/20 text-blue-400' },
  processing: { label: '解析中', color: 'bg-amber-400/20 text-amber-400' },
  ready: { label: '就绪', color: 'bg-emerald-400/20 text-emerald-400' },
  failed: { label: '失败', color: 'bg-red-400/20 text-red-400' },
};

export default function DatasetsPage() {
  const toast = useToast();
  const [datasets, setDatasets] = useState<DatasetData[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(0);
  const [search, setSearch] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showUploader, setShowUploader] = useState(false);
  const [detailDatasetId, setDetailDatasetId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [reparsingId, setReparsingId] = useState<string | null>(null);
  const [downloadingBuiltin, setDownloadingBuiltin] = useState<Record<string, { progress: number; status: string; message: string }>>({});

  const pageSize = 50;

  // 搜索防抖
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedSearch(search), 300);
    return () => clearTimeout(timer);
  }, [search]);

  const fetchDatasets = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiService.listDatasets({
        page,
        page_size: pageSize,
        search: debouncedSearch || undefined,
        status: statusFilter === 'all' ? undefined : statusFilter,
      });
      setDatasets(res.data.items);
      setTotal(res.data.total);
      setTotalPages(res.data.total_pages);
    } catch (err: any) {
      setError(err.message);
      setDatasets([]);
      setTotal(0);
    } finally {
      setLoading(false);
    }
  }, [page, debouncedSearch, statusFilter]);

  useEffect(() => {
    fetchDatasets();
  }, [fetchDatasets]);

  // 轮询：有 processing 状态的数据集时自动刷新
  useEffect(() => {
    const hasProcessing = datasets.some((d) => d.status === 'processing' || d.status === 'uploading');
    if (!hasProcessing) return;
    const timer = setInterval(() => fetchDatasets(), 3000);
    return () => clearInterval(timer);
  }, [datasets, fetchDatasets]);

  const handleDelete = async (datasetId: string, name: string) => {
    if (!confirm(`确定删除数据集「${name}」吗？`)) return;
    setDeletingId(datasetId);
    try {
      await apiService.deleteDataset(datasetId);
      toast.showSuccess('删除成功');
      fetchDatasets();
    } catch (err: any) {
      toast.showError('删除失败', err.message);
    } finally {
      setDeletingId(null);
    }
  };

  const handleReparse = async (datasetId: string, name: string) => {
    setReparsingId(datasetId);
    try {
      await apiService.reparseDataset(datasetId);
      toast.showSuccess('重新解析已启动', `数据集「${name}」正在重新解析`);
      // 立即刷新列表，轮询机制会自动跟踪 processing 状态
      fetchDatasets();
    } catch (err: any) {
      toast.showError('重试解析失败', err.message);
    } finally {
      setReparsingId(null);
    }
  };

  const handleDownloadBuiltin = async (name: 'mnist' | 'cifar10') => {
    try {
      setDownloadingBuiltin(prev => ({ ...prev, [name]: { progress: 0, status: 'starting', message: '启动下载...' } }));
      const res = await apiService.downloadBuiltinDataset(name);
      if (res.data.status === 'completed') {
        toast.showSuccess('下载完成', `${name === 'mnist' ? 'MNIST' : 'CIFAR-10'} 数据集已就绪`);
        setDownloadingBuiltin(prev => {
          const next = { ...prev };
          delete next[name];
          return next;
        });
        fetchDatasets();
      } else {
        toast.showInfo('下载已开始', `${name === 'mnist' ? 'MNIST' : 'CIFAR-10'} 正在后台下载，请稍候`);
      }
    } catch (err: any) {
      toast.showError('下载失败', err.message);
      setDownloadingBuiltin(prev => {
        const next = { ...prev };
        delete next[name];
        return next;
      });
    }
  };

  // 轮询下载进度
  useEffect(() => {
    const activeDownloads = Object.keys(downloadingBuiltin).filter(
      k => downloadingBuiltin[k].status !== 'completed' && downloadingBuiltin[k].status !== 'failed'
    );
    if (activeDownloads.length === 0) return;

    const timer = setInterval(async () => {
      for (const name of activeDownloads) {
        try {
          const res = await apiService.getBuiltinDownloadStatus(name as 'mnist' | 'cifar10');
          const task = res.data;
          if (task) {
            setDownloadingBuiltin(prev => ({ ...prev, [name]: task }));
            if (task.status === 'completed') {
              toast.showSuccess('下载完成', `${name === 'mnist' ? 'MNIST' : 'CIFAR-10'} 数据集已就绪`);
              setDownloadingBuiltin(prev => {
                const next = { ...prev };
                delete next[name];
                return next;
              });
              fetchDatasets();
            } else if (task.status === 'failed') {
              toast.showError('下载失败', task.message || '未知错误');
              setDownloadingBuiltin(prev => {
                const next = { ...prev };
                delete next[name];
                return next;
              });
            }
          }
        } catch {
          // 忽略轮询错误
        }
      }
    }, 2000);

    return () => clearInterval(timer);
  }, [downloadingBuiltin, fetchDatasets]);

  const formatDate = (dateStr: string) => {
    if (!dateStr) return '-';
    const d = new Date(dateStr);
    return d.toLocaleString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-[#0a0c14] text-white">
      {/* 顶部栏 */}
      <div className="flex-shrink-0 border-b border-white/[0.06] bg-[#0c0e17]/95 backdrop-blur-sm">
        {/* 标题行 */}
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-primary/60 shadow-lg shadow-primary/20">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-white">
                <ellipse cx="12" cy="5" rx="9" ry="3" />
                <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" />
                <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" />
              </svg>
            </div>
            <div>
              <h1 className="text-lg font-bold tracking-tight">数据集管理</h1>
              <p className="text-xs text-muted-foreground">共 {total} 个数据集</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            {/* 内置数据集快速下载 */}
            <div className="flex items-center gap-2 rounded-lg border border-white/[0.08] bg-white/[0.02] px-2 py-1">
              <span className="text-[10px] text-muted-foreground px-1">标准数据集</span>
              {(['mnist', 'cifar10'] as const).map(name => {
                const dl = downloadingBuiltin[name];
                const isDownloading = dl && dl.status !== 'completed' && dl.status !== 'failed';
                return (
                  <button
                    key={name}
                    onClick={() => !isDownloading && handleDownloadBuiltin(name)}
                    disabled={!!isDownloading}
                    className="relative flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium transition-all disabled:opacity-60 bg-white/[0.03] text-muted-foreground hover:bg-primary/10 hover:text-primary border border-transparent hover:border-primary/20"
                    title={name === 'mnist' ? '下载MNIST手写数字数据集 (约11MB)' : '下载CIFAR-10彩色图像数据集 (约170MB)'}
                  >
                    {isDownloading ? (
                      <>
                        <svg className="animate-spin" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                          <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                        </svg>
                        <span>{name.toUpperCase()}</span>
                        <span className="text-[10px] opacity-70">{Math.round(dl.progress)}%</span>
                      </>
                    ) : (
                      <>
                        <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                          <polyline points="7 10 12 15 17 10" />
                          <line x1="12" y1="15" x2="12" y2="3" />
                        </svg>
                        <span>{name.toUpperCase()}</span>
                      </>
                    )}
                    {isDownloading && (
                      <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-white/10 rounded-full overflow-hidden">
                        <div className="h-full bg-primary transition-all duration-300" style={{ width: `${dl.progress}%` }} />
                      </div>
                    )}
                  </button>
                );
              })}
            </div>

            <button
              onClick={() => setShowUploader(true)}
              className="flex items-center gap-1.5 rounded-lg border border-emerald-400/30 bg-emerald-400/10 px-3 py-1.5 text-xs font-semibold text-emerald-400 transition-all hover:bg-emerald-400/15"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              上传数据集
            </button>
          </div>
        </div>

        {/* 筛选栏 */}
        <div className="mx-auto flex max-w-7xl items-center gap-4 px-6 py-3">
          {/* 搜索框 */}
          <div className="relative flex-1 max-w-md">
            <svg className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="11" cy="11" r="8" />
              <path d="m21 21-4.35-4.35" />
            </svg>
            <input
              type="text"
              placeholder="搜索数据集名称、描述..."
              value={search}
              onChange={(e) => { setSearch(e.target.value); setPage(1); }}
              className="w-full rounded-lg border border-white/[0.08] bg-white/[0.02] py-2 pl-9 pr-3 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary/40 focus:outline-none"
            />
          </div>

          {/* 状态筛选 */}
          <div className="flex items-center gap-1 rounded-lg border border-white/[0.08] bg-white/[0.02] p-1">
            {['all', 'ready', 'processing', 'failed'].map((s) => (
              <button
                key={s}
                onClick={() => { setStatusFilter(s); setPage(1); }}
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
        {error ? (
          <div className="rounded-lg border border-red-400/20 bg-red-400/10 px-4 py-3 text-sm text-red-400">
            {error}
          </div>
        ) : loading ? (
          <div className="flex items-center justify-center py-20">
            <svg className="animate-spin text-muted-foreground" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M21 12a9 9 0 1 1-6.219-8.56" />
            </svg>
          </div>
        ) : datasets.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full border border-white/[0.08] bg-white/[0.02]">
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-muted-foreground">
                <path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V7" />
                <path d="M21 7l-9 6-9-6" />
                <path d="M3 7l9-4 9 4" />
              </svg>
            </div>
            <h3 className="text-sm font-semibold text-foreground">暂无数据集</h3>
            <p className="mt-1 text-xs text-muted-foreground">点击「上传数据集」导入你的数据</p>
          </div>
        ) : (
          <>
            {/* 表格 */}
            <div className="overflow-hidden rounded-xl border border-white/[0.06] bg-[#0f1119]">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/[0.06] bg-white/[0.01]">
                    <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">名称</th>
                    <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">版本</th>
                    <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">样本数</th>
                    <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">类别数</th>
                    <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">特征/尺寸</th>
                    <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">状态</th>
                    <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wider text-muted-foreground">创建时间</th>
                    <th className="px-4 py-3 text-right text-xs font-semibold uppercase tracking-wider text-muted-foreground">操作</th>
                  </tr>
                </thead>
                <tbody>
                  {datasets.map((ds) => {
                    const statusInfo = STATUS_LABELS[ds.status] || { label: ds.status, color: 'bg-gray-400/20 text-gray-400' };
                    return (
                      <tr
                        key={ds.dataset_id}
                        onClick={() => setDetailDatasetId(ds.dataset_id)}
                        className="cursor-pointer border-b border-white/[0.04] transition-colors hover:bg-white/[0.04]"
                      >
                        <td className="px-4 py-3">
                          <div className="font-medium text-foreground">{ds.name}</div>
                          {ds.description && (
                            <div className="mt-0.5 truncate text-xs text-muted-foreground max-w-[200px]">{ds.description}</div>
                          )}
                          {ds.tags?.length > 0 && (
                            <div className="mt-1 flex flex-wrap gap-1">
                              {ds.tags.slice(0, 3).map((tag, i) => (
                                <span key={i} className="rounded px-1.5 py-0.5 text-[10px] bg-white/[0.04] text-muted-foreground">{tag}</span>
                              ))}
                            </div>
                          )}
                        </td>
                        <td className="px-4 py-3">
                          <span className="rounded px-1.5 py-0.5 text-[10px] font-medium bg-white/[0.04] text-muted-foreground">{ds.version}</span>
                        </td>
                        <td className="px-4 py-3 font-mono text-foreground">{(ds.sample_count ?? 0).toLocaleString()}</td>
                        <td className="px-4 py-3 font-mono text-foreground">{ds.class_count}</td>
                        <td className="px-4 py-3 font-mono text-xs text-muted-foreground">{ds.feature_shape || ds.image_size || '-'}</td>
                        <td className="px-4 py-3">
                          <span
                            className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${statusInfo.color} ${ds.status === 'failed' && ds.error_message ? 'cursor-help' : ''}`}
                            title={ds.status === 'failed' && ds.error_message ? ds.error_message : undefined}
                          >
                            {ds.status === 'processing' && (
                              <svg className="mr-1 animate-spin" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                              </svg>
                            )}
                            {statusInfo.label}
                          </span>
                        </td>
                        <td className="px-4 py-3 text-xs text-muted-foreground">{formatDate(ds.created_at)}</td>
                        <td className="px-4 py-3 text-right" onClick={(e) => e.stopPropagation()}>
                          <div className="flex items-center justify-end gap-1.5">
                            {ds.status === 'failed' && (
                              <button
                                onClick={() => handleReparse(ds.dataset_id, ds.name)}
                                disabled={reparsingId === ds.dataset_id}
                                className="rounded-md border border-amber-400/30 bg-amber-400/10 px-2.5 py-1 text-xs font-medium text-amber-400 transition-all hover:bg-amber-400/15 disabled:opacity-50"
                              >
                                {reparsingId === ds.dataset_id ? (
                                  <span className="flex items-center gap-1">
                                    <svg className="animate-spin" width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                                    </svg>
                                    重试中
                                  </span>
                                ) : (
                                  <span className="flex items-center gap-1">
                                    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                      <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                                      <path d="M3 3v5h5" />
                                    </svg>
                                    重试解析
                                  </span>
                                )}
                              </button>
                            )}
                            <button
                              onClick={() => handleDelete(ds.dataset_id, ds.name)}
                              disabled={deletingId === ds.dataset_id}
                              className="rounded-md border border-white/[0.08] bg-white/[0.02] px-2.5 py-1 text-xs font-medium text-muted-foreground transition-all hover:border-red-400/30 hover:bg-red-400/10 hover:text-red-400 disabled:opacity-50"
                            >
                              {deletingId === ds.dataset_id ? '删除中...' : '删除'}
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
                <span className="text-xs text-muted-foreground">第 {page} / {totalPages} 页</span>
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

      {/* 上传弹窗 */}
      <DatasetUploader
        isOpen={showUploader}
        onClose={() => setShowUploader(false)}
        onUploaded={() => {
          fetchDatasets();
        }}
      />

      {/* 详情弹窗 */}
      <DatasetDetailModal
        datasetId={detailDatasetId}
        onClose={() => setDetailDatasetId(null)}
      />
    </div>
  );
}
