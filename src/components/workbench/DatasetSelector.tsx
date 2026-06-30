import { useState, useEffect, useRef } from 'react';
import { apiService, type DatasetData } from '../../services/api';
import { useToast } from '../../contexts/ToastContext';

/* ============================================
   DatasetSelector — 工作台数据集选择弹窗
   - 展示所有 ready 状态的数据集
   - 搜索筛选，选中后回调
   ============================================ */

interface DatasetSelectorProps {
  selectedDataset: DatasetData | null;
  onSelect: (dataset: DatasetData | null) => void;
}

export function DatasetSelector({ selectedDataset, onSelect }: DatasetSelectorProps) {
  const toast = useToast();
  const modalRef = useRef<HTMLDivElement>(null);
  const [isOpen, setIsOpen] = useState(false);
  const [datasets, setDatasets] = useState<DatasetData[]>([]);
  const [loading, setLoading] = useState(false);
  const [search, setSearch] = useState('');

  const fetchDatasets = async () => {
    setLoading(true);
    try {
      const res = await apiService.listDatasets({
        page: 1,
        page_size: 100,
        status: 'ready',
        search: search || undefined,
      });
      setDatasets(res.data.items);
    } catch {
      setDatasets([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) fetchDatasets();
  }, [isOpen, search]);

  // ESC 关闭
  useEffect(() => {
    if (!isOpen) return;
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setIsOpen(false);
    };
    document.addEventListener('keydown', handleEsc);
    return () => document.removeEventListener('keydown', handleEsc);
  }, [isOpen]);

  // 点击外部关闭
  useEffect(() => {
    if (!isOpen) return;
    const handleClickOutside = (e: MouseEvent) => {
      if (modalRef.current && !modalRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  const handleSelect = (ds: DatasetData) => {
    onSelect(ds);
    toast.showSuccess('数据集已关联', ds.name);
    setIsOpen(false);
  };

  return (
    <>
      {/* 触发按钮 */}
      <button
        onClick={() => setIsOpen(true)}
        className={`flex items-center gap-1.5 rounded-md border px-2.5 py-1.5 transition-all ${
          selectedDataset
            ? 'border-primary/30 bg-primary/10 text-primary'
            : 'border-white/[0.08] bg-white/[0.02] text-muted-foreground/80 hover:text-foreground'
        }`}
        title="选择训练数据集"
      >
        <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M3 7v10a2 2 0 002 2h14a2 2 0 002-2V7" />
          <path d="M21 7l-9 6-9-6" />
          <path d="M3 7l9-4 9 4" />
        </svg>
        <span className="text-[10px] font-semibold">
          {selectedDataset ? selectedDataset.name : '数据集'}
        </span>
      </button>

      {/* 弹窗 */}
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
          <div
            ref={modalRef}
            className="flex max-h-[80vh] w-full max-w-lg flex-col overflow-hidden rounded-2xl border border-white/[0.08] bg-[#0f1119] shadow-2xl"
          >
            {/* 头部 */}
            <div className="flex items-center justify-between border-b border-white/[0.06] px-6 py-4">
              <div>
                <h2 className="text-base font-bold">选择数据集</h2>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  {selectedDataset ? `当前: ${selectedDataset.name} (${selectedDataset.version})` : '未选择数据集'}
                </p>
              </div>
              <button
                onClick={() => setIsOpen(false)}
                className="flex h-8 w-8 items-center justify-center rounded-md border border-white/[0.08] bg-white/[0.02] text-muted-foreground transition-all hover:text-foreground"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M18 6 6 18M6 6l12 12" />
                </svg>
              </button>
            </div>

            {/* 搜索栏 */}
            <div className="border-b border-white/[0.06] px-6 py-3">
              <div className="relative">
                <svg className="absolute left-3 top-1/2 -translate-y-1/2 text-muted-foreground" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="11" cy="11" r="8" />
                  <path d="m21 21-4.35-4.35" />
                </svg>
                <input
                  type="text"
                  placeholder="搜索数据集..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  className="w-full rounded-lg border border-white/[0.08] bg-white/[0.02] py-2 pl-9 pr-3 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary/40 focus:outline-none"
                />
              </div>
            </div>

            {/* 列表 */}
            <div className="flex-1 overflow-y-auto p-4">
              {loading ? (
                <div className="flex items-center justify-center py-12">
                  <svg className="animate-spin text-muted-foreground" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                  </svg>
                </div>
              ) : datasets.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12">
                  <p className="text-sm text-muted-foreground">暂无可用数据集</p>
                  <p className="mt-1 text-xs text-muted-foreground">请先在数据集管理页面上传数据</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {datasets.map((ds) => (
                    <div
                      key={ds.dataset_id}
                      onClick={() => handleSelect(ds)}
                      className={`cursor-pointer rounded-lg border p-3 transition-all ${
                        selectedDataset?.dataset_id === ds.dataset_id
                          ? 'border-primary/40 bg-primary/[0.06]'
                          : 'border-white/[0.06] bg-white/[0.02] hover:border-white/10 hover:bg-white/[0.04]'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-medium text-foreground">{ds.name}</span>
                            <span className="rounded px-1.5 py-0.5 text-[10px] font-medium bg-white/[0.04] text-muted-foreground">{ds.version}</span>
                          </div>
                          <div className="mt-1 flex items-center gap-3 text-xs text-muted-foreground">
                            <span>{ds.sample_count.toLocaleString()} 样本</span>
                            <span>{ds.class_count} 类别</span>
                            {ds.image_size && <span>{ds.image_size}</span>}
                          </div>
                        </div>
                        {selectedDataset?.dataset_id === ds.dataset_id && (
                          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" className="text-primary">
                            <path d="M20 6L9 17l-5-5" />
                          </svg>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* 底部 */}
            {selectedDataset && (
              <div className="border-t border-white/[0.06] px-6 py-3">
                <button
                  onClick={() => {
                    onSelect(null);
                    toast.showInfo('已取消数据集关联');
                    setIsOpen(false);
                  }}
                  className="text-xs text-muted-foreground transition-all hover:text-red-400"
                >
                  取消关联当前数据集
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
