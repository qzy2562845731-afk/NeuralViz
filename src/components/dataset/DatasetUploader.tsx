import { useState, useRef, useEffect } from 'react';
import { apiService } from '../../services/api';
import { useToast } from '../../contexts/ToastContext';

/* ============================================
   DatasetUploader — 数据集上传弹窗
   - 拖拽/点击上传 zip 文件
   - 填写名称、描述、标签
   - 上传进度显示
   ============================================ */

interface DatasetUploaderProps {
  isOpen: boolean;
  onClose: () => void;
  onUploaded: () => void;
}

export function DatasetUploader({ isOpen, onClose, onUploaded }: DatasetUploaderProps) {
  const toast = useToast();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [tags, setTags] = useState('');
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  // ESC 关闭
  useEffect(() => {
    if (!isOpen) return;
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !uploading) onClose();
    };
    document.addEventListener('keydown', handleEsc);
    return () => document.removeEventListener('keydown', handleEsc);
  }, [isOpen, uploading, onClose]);

  // 重置表单
  useEffect(() => {
    if (!isOpen) {
      setFile(null);
      setName('');
      setDescription('');
      setTags('');
      setUploading(false);
      setDragOver(false);
    }
  }, [isOpen]);

  const handleFileSelect = (selected: File | null) => {
    if (!selected) return;
    if (!selected.name.toLowerCase().endsWith('.zip')) {
      toast.showError('格式不支持', '请上传 zip 格式压缩包，内部可包含图片目录、MNIST IDX、NumPy 或 CSV 数据');
      return;
    }
    if (selected.size > 500 * 1024 * 1024) {
      toast.showError('文件过大', '最大支持 500MB');
      return;
    }
    setFile(selected);
    // 自动填充名称
    if (!name) {
      setName(selected.name.replace(/\.zip$/i, ''));
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) handleFileSelect(droppedFile);
  };

  const handleUpload = async () => {
    if (!file) {
      toast.showError('请选择文件');
      return;
    }
    if (!name.trim()) {
      toast.showError('请填写数据集名称');
      return;
    }

    setUploading(true);
    try {
      const res = await apiService.uploadDataset(file, name.trim(), description.trim(), tags.trim());
      if (res.code === 200) {
        toast.showSuccess('数据集上传成功', '正在后台解析，请稍后刷新查看');
        onUploaded();
        onClose();
      } else {
        toast.showError('上传失败', res.message);
      }
    } catch (err: any) {
      const errMsg = err.message || '';
      // 格式检查引导：常见错误提示
      if (errMsg.includes('415') || errMsg.toLowerCase().includes('unsupported')) {
        toast.showError('上传失败', `${errMsg}\n请检查：仅支持 .zip 格式压缩包，内部可包含图片目录/MNIST IDX/NumPy/CSV 数据`);
      } else if (errMsg.includes('413') || errMsg.toLowerCase().includes('large') || errMsg.toLowerCase().includes('size')) {
        toast.showError('上传失败', `${errMsg}\n请检查：文件大小超过 500MB 限制`);
      } else {
        toast.showError('上传失败', errMsg);
      }
    } finally {
      setUploading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-md overflow-hidden rounded-2xl border border-white/[0.08] bg-[#0f1119] shadow-2xl">
        {/* 头部 */}
        <div className="border-b border-white/[0.06] px-6 py-4">
          <h2 className="text-base font-bold">上传数据集</h2>
          <p className="mt-0.5 text-xs text-muted-foreground">上传 zip 格式压缩包，自动识别数据集格式并解析</p>
        </div>

        {/* 内容区 */}
        <div className="space-y-4 p-6">
          {/* 拖拽上传区 */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed py-8 transition-all ${
              dragOver
                ? 'border-primary/50 bg-primary/[0.06]'
                : file
                ? 'border-emerald-400/40 bg-emerald-400/[0.04]'
                : 'border-white/[0.1] bg-white/[0.02] hover:border-white/20 hover:bg-white/[0.03]'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".zip"
              className="hidden"
              onChange={(e) => handleFileSelect(e.target.files?.[0] || null)}
            />
            {file ? (
              <>
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="mb-2 text-emerald-400">
                  <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                  <path d="M22 4L12 14.01l-3-3" />
                </svg>
                <p className="text-sm font-medium text-foreground">{file.name}</p>
                <p className="mt-0.5 text-xs text-muted-foreground">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
              </>
            ) : (
              <>
                <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="mb-2 text-muted-foreground">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                <p className="text-sm font-medium text-foreground/80">拖拽文件到此处或点击选择</p>
                <p className="mt-0.5 text-xs text-muted-foreground">支持 .zip 格式，最大 500MB</p>
              </>
            )}
          </div>

          {/* 支持的数据集格式说明 */}
          <div className="rounded-lg border border-white/[0.06] bg-white/[0.02] p-3">
            <div className="mb-2 flex items-center gap-1.5">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-primary">
                <circle cx="12" cy="12" r="10" />
                <path d="M12 16v-4M12 8h.01" />
              </svg>
              <span className="text-xs font-semibold text-foreground">支持的数据集格式</span>
            </div>
            <ul className="space-y-1.5 text-[11px] text-muted-foreground">
              <li className="flex gap-2">
                <span className="font-mono text-emerald-400/80">图片目录</span>
                <span>按类别分子文件夹的图片集（jpg/png/bmp/webp）</span>
              </li>
              <li className="flex gap-2">
                <span className="font-mono text-emerald-400/80">MNIST IDX</span>
                <span>-images-idx3-ubyte(.gz) + -labels-idx1-ubyte(.gz)</span>
              </li>
              <li className="flex gap-2">
                <span className="font-mono text-emerald-400/80">NumPy 数组</span>
                <span>.npy / .npz（键名 X/data 为特征，y/label 为标签）</span>
              </li>
              <li className="flex gap-2">
                <span className="font-mono text-emerald-400/80">CSV/TSV 表格</span>
                <span>结构化特征数据，默认最后一列为标签</span>
              </li>
            </ul>
          </div>

          {/* 名称 */}
          <div>
            <label className="mb-1.5 block text-xs font-medium text-muted-foreground">
              数据集名称 <span className="text-red-400">*</span>
            </label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="如：MNIST、CIFAR-10"
              className="w-full rounded-lg border border-white/[0.08] bg-white/[0.02] px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary/40 focus:outline-none"
            />
          </div>

          {/* 描述 */}
          <div>
            <label className="mb-1.5 block text-xs font-medium text-muted-foreground">描述</label>
            <textarea
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="数据集简要描述..."
              rows={2}
              className="w-full resize-none rounded-lg border border-white/[0.08] bg-white/[0.02] px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary/40 focus:outline-none"
            />
          </div>

          {/* 标签 */}
          <div>
            <label className="mb-1.5 block text-xs font-medium text-muted-foreground">标签（逗号分隔）</label>
            <input
              type="text"
              value={tags}
              onChange={(e) => setTags(e.target.value)}
              placeholder="如：图像分类, MNIST"
              className="w-full rounded-lg border border-white/[0.08] bg-white/[0.02] px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground focus:border-primary/40 focus:outline-none"
            />
          </div>
        </div>

        {/* 底部按钮 */}
        <div className="flex items-center justify-end gap-2 border-t border-white/[0.06] px-6 py-3">
          <button
            onClick={onClose}
            disabled={uploading}
            className="rounded-md border border-white/[0.08] bg-white/[0.02] px-4 py-1.5 text-xs font-medium text-muted-foreground transition-all hover:text-foreground disabled:opacity-50"
          >
            取消
          </button>
          <button
            onClick={handleUpload}
            disabled={uploading || !file || !name.trim()}
            className="flex items-center gap-1.5 rounded-md border border-emerald-400/30 bg-emerald-400/10 px-4 py-1.5 text-xs font-semibold text-emerald-400 transition-all hover:bg-emerald-400/15 disabled:opacity-50"
          >
            {uploading ? (
              <>
                <svg className="animate-spin" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                  <path d="M21 12a9 9 0 1 1-6.219-8.56" />
                </svg>
                上传中...
              </>
            ) : (
              <>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                上传数据集
              </>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
