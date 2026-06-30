import { useState, useRef, useCallback, useEffect } from 'react';
import { analyzeModel, type ModelAnalysisResult, generateSampleModel } from '../cnn3d/modelAnalyzer';
import { apiService } from '../../services/api';
import { useWorkbench } from './WorkbenchContext';

interface ModelImporterProps {
  onModelLoad: (result: ModelAnalysisResult) => void;
}

// 支持的文件扩展名
const SUPPORTED_EXTENSIONS = [
  '.json', '.js', '.onnx', '.h5', '.hdf5',
  '.pt', '.pth', '.pickle', '.pkl', '.pb', '.keras'
];

// 文件格式说明
const FORMAT_INFO: Record<string, { framework: string; note: string; color: string }> = {
  '.json': { framework: '通用 JSON', note: '直接解析（PyTorch/Keras 导出）', color: 'text-emerald-400' },
  '.js': { framework: '通用 JSON', note: '直接解析', color: 'text-emerald-400' },
  '.onnx': { framework: 'ONNX', note: '二进制格式，需本地服务解析', color: 'text-blue-400' },
  '.h5': { framework: 'Keras / HDF5', note: '二进制格式，需本地服务解析', color: 'text-blue-400' },
  '.hdf5': { framework: 'HDF5', note: '二进制格式，需本地服务解析', color: 'text-blue-400' },
  '.pt': { framework: 'PyTorch', note: '二进制格式，需本地服务解析', color: 'text-amber-400' },
  '.pth': { framework: 'PyTorch', note: '二进制格式，需本地服务解析', color: 'text-amber-400' },
  '.pickle': { framework: 'Python Pickle', note: '二进制格式，需本地服务解析', color: 'text-amber-400' },
  '.pkl': { framework: 'Python Pickle', note: '二进制格式，需本地服务解析', color: 'text-amber-400' },
  '.pb': { framework: 'TensorFlow', note: '二进制格式，需本地服务解析', color: 'text-amber-400' },
  '.keras': { framework: 'Keras', note: '二进制格式，需本地服务解析', color: 'text-blue-400' },
};

type TabType = 'upload' | 'server' | 'python';

export function ModelImporter({ onModelLoad }: ModelImporterProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<ModelAnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<TabType>('upload');
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dropZoneRef = useRef<HTMLDivElement>(null);

  const {
    serverStatus,
    serverError,
    setServerStatus,
    setCurrentModel,
  } = useWorkbench();

  const serverUrl = 'http://localhost:8000';
  const [serverModels, setServerModels] = useState<Array<{ id: string; name: string }>>([]);

  useEffect(() => {
    if (dragOver && dropZoneRef.current) {
      const el = dropZoneRef.current;
      const handleDragLeave = () => setDragOver(false);
      const handleDrop = (e: DragEvent) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer?.files?.[0];
        if (file) {
          processFile(file);
        }
      };
      el.addEventListener('dragleave', handleDragLeave);
      el.addEventListener('drop', handleDrop);
      return () => {
        el.removeEventListener('dragleave', handleDragLeave);
        el.removeEventListener('drop', handleDrop);
      };
    }
  }, [dragOver]);

  const processFile = useCallback(async (file: File) => {
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();

    if (!SUPPORTED_EXTENSIONS.includes(ext)) {
      setError(`不支持的文件格式: ${ext}。支持: ${SUPPORTED_EXTENSIONS.join(', ')}`);
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    try {
      if (ext === '.json' || ext === '.js') {
        const text = await file.text();
        const jsonData = JSON.parse(text);
        const result = analyzeModel(jsonData);
        setAnalysisResult(result);
        const modelName = (jsonData as any).model_name || (jsonData as any).name || file.name.replace(/\.[^.]+$/, '');
        setCurrentModel(`json_${Date.now()}`, modelName);
      } else if (serverStatus === 'connected') {
        const result = await apiService.parseModel(file);
        const modelData = result.data;
        const analysis = analyzeModel(modelData);
        setAnalysisResult(analysis);
        setCurrentModel(modelData.model_id || null, modelData.model_name || file.name);
      } else {
        setError(
          `${FORMAT_INFO[ext]?.framework || '二进制'} 格式需要本地 Python 服务解析。` +
          `请切换到"本地服务"标签页连接服务，或参考"Python 模板"启动服务。`
        );
        setAnalysisResult(null);
      }
    } catch (err) {
      setError(`解析失败: ${err instanceof Error ? err.message : '未知错误'}`);
      setAnalysisResult(null);
    } finally {
      setIsAnalyzing(false);
    }
  }, [serverStatus, setCurrentModel]);

  const handleFileSelect = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) processFile(file);
  }, [processFile]);

  const handleLoadModel = useCallback(() => {
    if (analysisResult) {
      onModelLoad(analysisResult);
      setIsOpen(false);
      setAnalysisResult(null);
    }
  }, [analysisResult, onModelLoad]);

  const handleLoadSample = useCallback(() => {
    const sample = generateSampleModel();
    setAnalysisResult(sample);
    setCurrentModel('sample_cnn', 'SampleCNN');
  }, [setCurrentModel]);

  const connectServer = useCallback(async () => {
    setServerStatus('connecting', null);
    setServerModels([]);

    try {
      const result = await apiService.getServiceStatus();
      
      if (result.code === 200 && result.data.status === 'online') {
        setServerStatus('connected', null);

        try {
          const modelsResp = await fetch(`${serverUrl}/models`, {
            method: 'GET',
            signal: AbortSignal.timeout(3000),
          });
          if (modelsResp.ok) {
            const modelsData = await modelsResp.json();
            const rawModels = modelsData.models;
            if (Array.isArray(rawModels)) {
              const parsed = rawModels.map((m: any) => ({
                id: m.model_id || m.id || m.name || String(m),
                name: m.model_name || m.name || m.model_id || String(m),
              }));
              setServerModels(parsed);
            }
          }
        } catch {
        }
      } else {
        throw new Error(result.message || '服务响应异常');
      }
    } catch (err) {
      setServerStatus('error', 
        `无法连接到服务: ${err instanceof Error ? err.message : '未知错误'}. ` +
        `请确认在本地运行 Python 解析服务，并在"Python 模板"标签页查看如何启动。`
      );
    }
  }, [serverUrl, setServerStatus]);

  const loadFromServer = useCallback(async (modelId: string) => {
    setIsAnalyzing(true);
    setError(null);

    try {
      const response = await fetch(`${serverUrl}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelId }),
        signal: AbortSignal.timeout(15000),
      });

      if (!response.ok) throw new Error(`服务错误: ${response.status}`);

      const result = await response.json();
      // 服务端返回的格式应与 ModelAnalysisResult 兼容
      const analysis = analyzeModel(result);
      setAnalysisResult(analysis);
      const modelName = result.model_name || result.name || modelId;
      setCurrentModel(modelId, modelName);
    } catch (err) {
      setError(`从服务端加载模型失败: ${err instanceof Error ? err.message : '未知错误'}`);
    } finally {
      setIsAnalyzing(false);
    }
  }, [serverUrl, setCurrentModel]);

  const formatParams = (params: number): string => {
    if (params >= 1000000) return `${(params / 1000000).toFixed(2)}M`;
    if (params >= 1000) return `${(params / 1000).toFixed(1)}K`;
    return params.toString();
  };

  const handleOpenToggle = () => {
    setIsOpen(!isOpen);
    if (isOpen) {
      setAnalysisResult(null);
      setError(null);
    }
  };

  // 监听外部打开弹窗的事件（例如推理面板的"去导入"引导）
  useEffect(() => {
    const handleOpen = () => setIsOpen(true);
    window.addEventListener('open-model-importer', handleOpen);
    return () => window.removeEventListener('open-model-importer', handleOpen);
  }, []);

  // Bug3修复：组件挂载时自动探活，已连接则复用，异常则标记断开
  useEffect(() => {
    if (serverStatus !== 'connected') return;
    let cancelled = false;
    const probe = async () => {
      try {
        const result = await apiService.getServiceStatus();
        if (cancelled) return;
        if (result.code !== 200 || result.data?.status !== 'online') {
          setServerStatus('error', '服务已离线');
        }
        // 探活成功：保持 connected 状态，不强制重连
      } catch {
        if (!cancelled) {
          setServerStatus('error', '无法连接到服务');
        }
      }
    };
    probe();
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // 仅挂载时执行一次

  return (
    <>
      <button
        onClick={handleOpenToggle}
        className="flex items-center gap-2 rounded-lg border border-white/[0.08] bg-white/[0.02] px-3 py-1.5 text-xs font-semibold text-muted-foreground transition-all hover:border-primary/30 hover:text-foreground"
      >
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="17 8 12 3 7 8" />
          <line x1="12" y1="3" x2="12" y2="15" />
        </svg>
        导入模型
      </button>

      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={handleOpenToggle} />

          <div className="relative w-full max-w-2xl overflow-hidden rounded-xl border border-border bg-card shadow-2xl">
            {/* 头部 */}
            <div className="flex items-center justify-between border-b border-border px-5 py-4">
              <div>
                <h2 className="text-base font-semibold">模型导入与分析</h2>
                <p className="text-[10px] text-muted-foreground mt-0.5">支持 JSON 直接导入，或通过本地 Python 服务解析 ONNX/H5/PyTorch 模型</p>
              </div>
              <button
                onClick={handleOpenToggle}
                className="rounded-full p-1 text-muted-foreground hover:bg-muted hover:text-foreground transition-colors"
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <line x1="18" y1="6" x2="6" y2="18" />
                  <line x1="6" y1="6" x2="18" y2="18" />
                </svg>
              </button>
            </div>

            {/* Tab 切换 */}
            <div className="flex gap-1 px-5 pt-3 border-b border-border">
              {[
                { id: 'upload' as TabType, name: '文件导入', icon: '📁' },
                { id: 'server' as TabType, name: '本地服务', icon: '🔌' },
                { id: 'python' as TabType, name: 'Python 模板', icon: '🐍' },
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-1.5 rounded-t-lg px-3 py-2 text-[11px] font-semibold transition-all ${
                    activeTab === tab.id
                      ? 'bg-muted/50 text-foreground border-b-2 border-primary'
                      : 'text-muted-foreground hover:text-foreground'
                  }`}
                >
                  <span className="text-xs">{tab.icon}</span>
                  {tab.name}
                </button>
              ))}
            </div>

            {/* 内容 */}
            <div className="p-5">
              {!analysisResult ? (
                <>
                  {activeTab === 'upload' && (
                    <div className="space-y-4">
                      {/* 文件上传区 */}
                      <div
                        ref={dropZoneRef}
                        onClick={() => fileInputRef.current?.click()}
                        onDragOver={(e) => {
                          e.preventDefault();
                          setDragOver(true);
                        }}
                        className={`group relative flex min-h-[140px] cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed transition-all ${
                          dragOver
                            ? 'border-primary bg-primary/10 scale-[1.01]'
                            : 'border-border bg-muted/20 hover:border-primary/50 hover:bg-muted/30'
                        }`}
                      >
                        <input
                          ref={fileInputRef}
                          type="file"
                          accept={SUPPORTED_EXTENSIONS.join(',')}
                          onChange={handleFileSelect}
                          className="hidden"
                        />
                        <svg
                          width="36"
                          height="36"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="1.5"
                          className={`mb-3 transition-colors ${dragOver ? 'text-primary' : 'text-muted-foreground group-hover:text-primary'}`}
                        >
                          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                          <polyline points="17 8 12 3 7 8" />
                          <line x1="12" y1="3" x2="12" y2="15" />
                        </svg>
                        <p className="text-sm font-semibold text-foreground">
                          {dragOver ? '松开以上传文件' : '点击或拖拽上传模型文件'}
                        </p>
                        <p className="text-[10px] text-muted-foreground mt-1">
                          支持: {SUPPORTED_EXTENSIONS.join(', ')}
                        </p>
                      </div>

                      {/* 支持格式说明 */}
                      <div className="rounded-lg border border-border bg-muted/10 p-3">
                        <p className="text-[10px] font-semibold uppercase text-muted-foreground mb-2">格式说明</p>
                        <div className="grid grid-cols-2 gap-x-3 gap-y-1.5">
                          {Object.entries(FORMAT_INFO).map(([ext, info]) => (
                            <div key={ext} className="grid grid-cols-[3rem_1fr] items-center gap-2 text-[10px]">
                              <span className={`font-mono ${info.color}`}>{ext}</span>
                              <span className="text-muted-foreground truncate">{info.framework}</span>
                            </div>
                          ))}
                        </div>
                        <p className="text-[9px] text-muted-foreground/80 mt-2 leading-relaxed">
                          ✨ JSON/JS 格式可直接在浏览器解析；其他二进制格式需通过本地 Python 服务解析
                        </p>
                      </div>

                      {error && (
                        <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-3">
                          <p className="text-sm text-destructive">{error}</p>
                        </div>
                      )}

                      {isAnalyzing && (
                        <div className="flex items-center justify-center py-4">
                          <div className="mr-3 h-5 w-5 animate-spin rounded-full border-2 border-primary/30 border-t-primary" />
                          <span className="text-sm text-muted-foreground">正在分析模型...</span>
                        </div>
                      )}

                      <button
                        onClick={handleLoadSample}
                        className="group relative flex w-full items-center justify-center gap-2 overflow-hidden rounded-lg border border-white/[0.08] bg-gradient-to-b from-white/[0.06] to-white/[0.02] py-3 text-sm font-semibold text-foreground shadow-sm transition-all hover:border-primary/40 hover:shadow-[0_0_20px_rgba(124,58,237,0.15)] hover:from-white/[0.08] hover:to-primary/[0.08] active:scale-[0.99]"
                      >
                        <span className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/[0.08] to-transparent transition-transform duration-700 group-hover:translate-x-full" />
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="relative z-10 transition-transform group-hover:rotate-12">
                          <circle cx="12" cy="12" r="10" />
                          <polyline points="12 6 12 12 16 14" />
                        </svg>
                        <span className="relative z-10">加载示例模型（含 12 层网络）</span>
                      </button>
                    </div>
                  )}

                  {activeTab === 'server' && (
                    <div className="space-y-4">
                      {/* 服务配置 */}
                      <div className="rounded-lg border border-border bg-muted/20 p-4">
                        <p className="text-[10px] font-semibold uppercase text-muted-foreground mb-3">本地服务配置</p>
                        <div className="flex gap-2">
                          <div
                            className="flex-1 rounded-lg border border-border bg-muted/30 px-3 py-2 text-sm font-mono text-muted-foreground"
                          >
                            {serverUrl}
                          </div>
                          <button
                            onClick={connectServer}
                            disabled={serverStatus === 'connecting'}
                            className="rounded-lg bg-primary px-4 py-2 text-sm font-semibold text-primary-foreground transition-all hover:bg-primary/90 disabled:opacity-50"
                          >
                            {serverStatus === 'connecting' ? '连接中...' : '检测连接'}
                          </button>
                        </div>

                        <div className="mt-3 flex items-center gap-2">
                          <span
                            className={`inline-block h-2 w-2 rounded-full ${
                              serverStatus === 'connected'
                                ? 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.6)]'
                                : serverStatus === 'error'
                                ? 'bg-red-400 shadow-[0_0_8px_rgba(248,113,113,0.6)]'
                                : serverStatus === 'connecting'
                                ? 'bg-amber-400 animate-pulse'
                                : 'bg-slate-500'
                            }`}
                          />
                          <span className="text-xs text-muted-foreground">
                            {serverStatus === 'connected'
                              ? '服务已连接'
                              : serverStatus === 'error'
                              ? '连接失败'
                              : serverStatus === 'connecting'
                              ? '尝试连接...'
                              : '未连接'}
                          </span>
                        </div>

                        {serverError && (
                          <div className="mt-2 rounded border border-destructive/30 bg-destructive/5 p-2.5">
                            <p className="text-xs text-destructive">{serverError}</p>
                          </div>
                        )}
                      </div>

                      {/* 可用模型列表 */}
                      {serverStatus === 'connected' && serverModels.length > 0 && (
                        <div className="rounded-lg border border-border bg-muted/20 p-4">
                          <p className="text-[10px] font-semibold uppercase text-muted-foreground mb-3">可用模型</p>
                          <div className="space-y-2 max-h-[200px] overflow-y-auto">
                            {serverModels.map((model) => (
                              <button
                                key={model.id}
                                onClick={() => loadFromServer(model.id)}
                                className="flex w-full items-center justify-between rounded-md bg-background px-3 py-2.5 text-xs font-medium transition-all hover:border-primary/30 hover:bg-muted/30 border border-border"
                              >
                                <span className="font-mono">{model.name}</span>
                                <span className="text-[10px] text-primary">→ 分析并加载</span>
                              </button>
                            ))}
                          </div>
                        </div>
                      )}

                      {serverStatus === 'connected' && serverModels.length === 0 && (
                        <div className="rounded-lg border border-emerald-400/30 bg-emerald-400/5 p-4">
                          <p className="text-xs text-emerald-400">✓ 服务已连接</p>
                          <p className="text-xs text-muted-foreground mt-1.5">
                            可通过 POST /analyze 发送模型文件路径或内容来解析模型。切换到"Python 模板"标签页查看如何上传自定义模型。
                          </p>
                        </div>
                      )}

                      {isAnalyzing && (
                        <div className="flex items-center justify-center py-4">
                          <div className="mr-3 h-5 w-5 animate-spin rounded-full border-2 border-primary/30 border-t-primary" />
                          <span className="text-sm text-muted-foreground">正在从服务端分析模型...</span>
                        </div>
                      )}

                      {error && (
                        <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-3">
                          <p className="text-sm text-destructive">{error}</p>
                        </div>
                      )}
                    </div>
                  )}

                  {activeTab === 'python' && (
                    <div className="space-y-4">
                      <div className="rounded-lg border border-border bg-muted/10 p-4">
                        <p className="text-[10px] font-semibold uppercase text-muted-foreground mb-2">快速启动本地服务</p>
                        <div className="space-y-2">
                          <p className="text-xs text-foreground/90">① 保存以下代码为 <span className="font-mono text-primary">model_server.py</span></p>
                          <p className="text-xs text-foreground/90">② 安装依赖: <span className="font-mono text-amber-400">pip install torch tensorflow onnx fastapi uvicorn pydantic</span></p>
                          <p className="text-xs text-foreground/90">③ 启动服务: <span className="font-mono text-emerald-400">python model_server.py</span></p>
                        </div>
                      </div>

                      {/* Python 模板代码 */}
                      <div className="rounded-lg border border-border bg-black/40 overflow-hidden">
                        <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-black/20">
                          <span className="text-[10px] font-mono text-slate-400">model_server.py</span>
                          <button
                            onClick={() => navigator.clipboard?.writeText(PYTHON_SERVER_CODE)}
                            className="text-[10px] font-semibold text-primary hover:text-primary/80"
                          >
                            复制到剪贴板
                          </button>
                        </div>
                        <pre className="p-4 overflow-x-auto max-h-[300px] text-[10px] leading-relaxed font-mono text-slate-300">
                          <code>{PYTHON_SERVER_CODE}</code>
                        </pre>
                      </div>

                      <div className="rounded-lg border border-amber-400/20 bg-amber-400/5 p-3">
                        <p className="text-[10px] font-semibold text-amber-400 mb-1.5">⚠ 使用说明</p>
                        <p className="text-[10px] text-muted-foreground leading-relaxed">
                          1. 默认端口 8000，可在脚本中修改<br />
                          2. 将模型文件放在与脚本相同目录，或提供绝对路径<br />
                          3. 支持 PyTorch (.pt/.pth)、Keras (.h5/.keras)、ONNX (.onnx) 模型<br />
                          4. 浏览器调用 /analyze 接口获取模型架构的 JSON 描述<br />
                          5. 也支持将 model 参数设为模型文件路径
                        </p>
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <div className="space-y-4">
                  {/* 分析结果摘要 */}
                  <div className="rounded-lg border border-border bg-muted/20 p-4">
                    <h3 className="mb-3 text-sm font-semibold">模型分析结果</h3>

                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <p className="text-[10px] font-medium uppercase text-muted-foreground">总层数</p>
                        <p className="text-xl font-bold text-foreground">{analysisResult.summary.totalLayers}</p>
                      </div>
                      <div>
                        <p className="text-[10px] font-medium uppercase text-muted-foreground">总参数</p>
                        <p className="text-xl font-bold text-foreground">{formatParams(analysisResult.summary.totalParams)}</p>
                      </div>
                      <div>
                        <p className="text-[10px] font-medium uppercase text-muted-foreground">输入形状</p>
                        <p className="font-mono text-sm text-foreground">{analysisResult.summary.inputShape.join('×')}</p>
                      </div>
                      <div>
                        <p className="text-[10px] font-medium uppercase text-muted-foreground">输出形状</p>
                        <p className="font-mono text-sm text-foreground">{analysisResult.summary.outputShape.join('×')}</p>
                      </div>
                    </div>

                    {/* 层类型分布 */}
                    <div className="mt-4">
                      <p className="mb-2 text-[10px] font-medium uppercase text-muted-foreground">层类型分布</p>
                      <div className="flex flex-wrap gap-2">
                        {Object.entries(analysisResult.summary.layerTypes).map(([type, count]) => (
                          count > 0 && (
                            <span
                              key={type}
                              className="rounded-full border border-border bg-muted/30 px-2 py-1 font-mono text-[10px] capitalize"
                            >
                              {type}: {count}
                            </span>
                          )
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* 警告 */}
                  {analysisResult.warnings.length > 0 && (
                    <div className="space-y-2">
                      <p className="text-[10px] font-medium uppercase text-muted-foreground">分析警告</p>
                      {analysisResult.warnings.map((warning) => (
                        <div
                          key={warning.id}
                          className={`rounded-lg border p-3 text-sm ${
                            warning.severity === 'error'
                              ? 'border-destructive/30 bg-destructive/5 text-destructive'
                              : warning.severity === 'warning'
                              ? 'border-amber-400/30 bg-amber-400/5 text-amber-400'
                              : 'border-primary/30 bg-primary/5 text-primary'
                          }`}
                        >
                          <p>{warning.message}</p>
                          {warning.suggestion && <p className="mt-1 text-xs opacity-80">建议: {warning.suggestion}</p>}
                        </div>
                      ))}
                    </div>
                  )}

                  {/* 层列表预览 */}
                  <div className="rounded-lg border border-border bg-muted/20 p-4">
                    <p className="mb-3 text-[10px] font-medium uppercase text-muted-foreground">层列表预览</p>
                    <div className="max-h-[150px] overflow-y-auto space-y-1">
                      {analysisResult.architecture.layers.slice(0, 20).map((layer, index) => (
                        <div key={layer.id} className="flex items-center gap-2 rounded-md bg-muted/30 px-2 py-1.5 text-xs">
                          <span className="w-5 flex-shrink-0 text-center font-mono text-muted-foreground">
                            {String(index + 1).padStart(2, '0')}
                          </span>
                          <span
                            className={`w-2 h-2 flex-shrink-0 rounded-full`}
                            style={{
                              backgroundColor: {
                                input: '#76b900',
                                conv: '#7c3aed',
                                pool: '#06b6d4',
                                fc: '#f97316',
                                output: '#ef4444',
                                norm: '#eab308',
                                dropout: '#9ca3af',
                              }[layer.type],
                            }}
                          />
                          <span className="flex-1 truncate font-medium">{layer.name}</span>
                          <span className="flex-shrink-0 font-mono text-muted-foreground capitalize">{layer.type}</span>
                        </div>
                      ))}
                      {analysisResult.architecture.layers.length > 20 && (
                        <p className="text-center text-xs text-muted-foreground">
                          ... 还有 {analysisResult.architecture.layers.length - 20} 层
                        </p>
                      )}
                    </div>
                  </div>

                  {/* 操作按钮 */}
                  <div className="flex gap-3">
                    <button
                      onClick={() => setAnalysisResult(null)}
                      className="flex-1 rounded-lg border border-border bg-muted/20 py-2.5 text-sm font-medium text-foreground transition-all hover:bg-muted/30"
                    >
                      返回
                    </button>
                    <button
                      onClick={handleLoadModel}
                      className="flex-1 rounded-lg bg-primary py-2.5 text-sm font-medium text-primary-foreground transition-all hover:bg-primary/90"
                    >
                      加载模型
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </>
  );
}

// Python 服务端代码模板 - 支持图片推理和激活值提取
const PYTHON_SERVER_CODE = `# CNN Visualizer - Inference Server v2.0
# 运行: python model_server.py
# 访问: http://localhost:8000
# 
# 功能：
# 1. 模型结构分析
# 2. 图片推理 + 每层激活值提取
# 3. 支持 PyTorch / Keras / ONNX 模型

import io
import json
import os
import base64
import tempfile
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np

app = FastAPI(title="CNN Visualizer - Inference Server", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== 配置 =====
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)

# 缓存已加载的模型
loaded_models: Dict[str, Any] = {}


# ===== 工具函数 =====
def preprocess_image_pil(image_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    """使用 PIL 预处理图片"""
    try:
        from PIL import Image
        import torchvision.transforms as transforms

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        tensor = transform(img)
        return tensor.numpy()
    except ImportError:
        raise HTTPException(status_code=500, detail="需要安装 Pillow 和 torchvision: pip install pillow torchvision")


def preprocess_image_cv2(image_bytes: bytes, target_size=(224, 224)) -> np.ndarray:
    """使用 OpenCV 预处理图片（备选方案）"""
    import cv2

    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="无法解码图片")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return img


def extract_activations_pytorch(model, input_tensor, layer_names: List[str]) -> Dict[str, List[float]]:
    """使用 PyTorch hooks 提取每层激活值"""
    import torch

    activations = {}
    hooks = []

    def get_activation(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                # 对输出取均值作为激活强度
                act = output.detach().cpu().numpy()
                # 如果是多维，取全局平均池化的值
                if len(act.shape) > 2:
                    act = np.mean(act, axis=tuple(range(2, len(act.shape))))
                # 确保是一维数组
                act = act.flatten()
                # 取前 N 个神经元
                activations[name] = act[:min(32, len(act))].tolist()
            elif isinstance(output, tuple):
                for i, o in enumerate(output):
                    if isinstance(o, torch.Tensor):
                        act = o.detach().cpu().numpy()
                        if len(act.shape) > 2:
                            act = np.mean(act, axis=tuple(range(2, len(act.shape))))
                        act = act.flatten()
                        activations[f"{name}_{i}"] = act[:min(32, len(act))].tolist()
        return hook

    # 注册 hooks 到常见层
    for name, module in model.named_modules():
        if any(skip in name.lower() for skip in ["input", "dropout", "identity"]):
            continue
        hooks.append(module.register_forward_hook(get_activation(name)))

    with torch.no_grad():
        if isinstance(input_tensor, np.ndarray):
            input_tensor = torch.from_numpy(input_tensor).float()
        model(input_tensor)

    # 移除 hooks
    for hook in hooks:
        hook.remove()

    return activations


def infer_pytorch(model_path: str, image_bytes: bytes) -> Dict[str, Any]:
    """PyTorch 模型推理"""
    import torch
    from PIL import Image

    # 加载模型
    if model_path not in loaded_models:
        model = torch.load(model_path, map_location="cpu")
        if hasattr(model, "eval"):
            model.eval()
        loaded_models[model_path] = model
    else:
        model = loaded_models[model_path]

    # 预处理图片
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    import torchvision.transforms as transforms

    # 尝试自动检测输入尺寸
    input_size = (224, 224)  # 默认值
    try:
        if hasattr(model, "input_size"):
            input_size = model.input_size
        elif hasattr(model, "config"):
            if isinstance(model.config, dict):
                input_size = model.config.get("input_size", (224, 224))
    except:
        pass

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensor = transform(img).unsqueeze(0)  # 添加 batch 维度

    # 提取激活值
    activations = extract_activations_pytorch(model, tensor, [])

    # 运行推理
    with torch.no_grad():
        output = model(tensor)

    output_data = output.cpu().numpy()
    if len(output_data.shape) > 2:
        output_data = output_data.flatten()
    else:
        output_data = output_data[0]

    # 获取预测类别（Top-5）
    probs = np.exp(output_data) / np.sum(np.exp(output_data))
    top5_idx = np.argsort(probs)[-5:][::-1]
    predictions = [{"class_id": int(i), "probability": float(probs[i])} for i in top5_idx]

    return {
        "activations": activations,
        "predictions": predictions,
        "input_size": input_size,
        "success": True,
    }


def infer_keras(model_path: str, image_bytes: bytes) -> Dict[str, Any]:
    """Keras 模型推理"""
    try:
        import tensorflow as tf
    except ImportError:
        raise HTTPException(status_code=500, detail="需要安装 TensorFlow: pip install tensorflow")

    # 加载模型
    if model_path not in loaded_models:
        model = tf.keras.models.load_model(model_path)
        loaded_models[model_path] = model
    else:
        model = loaded_models[model_path]

    # 预处理图片
    img = tf.keras.preprocessing.image.load_img(io.BytesIO(image_bytes), target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    # 提取中间层激活
    layer_outputs = {}
    for layer in model.layers:
        if "conv" in layer.name.lower() or "dense" in layer.name.lower():
            try:
                intermediate_model = tf.keras.Model(
                    inputs=model.input,
                    outputs=layer.output
                )
                act = intermediate_model.predict(img_array, verbose=0)
                if len(act.shape) > 2:
                    act = np.mean(act, axis=(1, 2))
                act = act.flatten()[:32]
                layer_outputs[layer.name] = act.tolist()
            except:
                pass

    # 运行推理
    predictions = model.predict(img_array, verbose=0)
    if len(predictions.shape) > 1:
        predictions = predictions[0]

    probs = predictions / np.sum(predictions) if np.sum(predictions) > 0 else predictions
    top5_idx = np.argsort(probs)[-5:][::-1]
    top5_predictions = [{"class_id": int(i), "probability": float(probs[i])} for i in top5_idx]

    return {
        "activations": layer_outputs,
        "predictions": top5_predictions,
        "input_size": (224, 224),
        "success": True,
    }


def infer_onnx(model_path: str, image_bytes: bytes) -> Dict[str, Any]:
    """ONNX 模型推理"""
    try:
        import onnxruntime as ort
    except ImportError:
        raise HTTPException(status_code=500, detail="需要安装 onnxruntime: pip install onnxruntime")

    if model_path not in loaded_models:
        session = ort.InferenceSession(model_path)
        loaded_models[model_path] = session
    else:
        session = loaded_models[model_path]

    # 预处理
    img = preprocess_image_cv2(image_bytes, (224, 224))
    img = img.astype(np.float32)
    img = np.expand_dims(img, 0)

    inputs = session.get_inputs()
    input_name = inputs[0].name
    outputs = session.get_outputs()

    # 推理
    result = session.run([o.name for o in outputs], {input_name: img})
    predictions = result[0]

    if len(predictions.shape) > 1:
        predictions = predictions[0]
    probs = predictions / np.sum(predictions) if np.sum(predictions) > 0 else predictions
    top5_idx = np.argsort(probs)[-5:][::-1]
    top5_predictions = [{"class_id": int(i), "probability": float(probs[i])} for i in top5_idx]

    return {
        "activations": {},  # ONNX 不易提取中间层激活
        "predictions": top5_predictions,
        "input_size": (224, 224),
        "success": True,
    }


def auto_infer(model_path: str, image_bytes: bytes) -> Dict[str, Any]:
    """根据模型类型自动选择推理器"""
    ext = os.path.splitext(model_path)[1].lower()
    if ext in (".pt", ".pth"):
        return infer_pytorch(model_path, image_bytes)
    elif ext in (".h5", ".keras"):
        return infer_keras(model_path, image_bytes)
    elif ext == ".onnx":
        return infer_onnx(model_path, image_bytes)
    else:
        raise HTTPException(status_code=400, detail=f"不支持的模型格式: {ext}")


# ===== API 端点 =====

@app.get("/health")
def health():
    """健康检查"""
    return {
        "status": "ok",
        "version": "2.0.0",
        "models_dir": str(MODEL_DIR.absolute()),
        "loaded_models": list(loaded_models.keys()),
    }


@app.get("/models")
def list_models():
    """列出可用模型"""
    files = []
    for f in MODEL_DIR.iterdir():
        if f.is_file() and f.suffix.lower() in (".pt", ".pth", ".h5", ".keras", ".onnx"):
            files.append(f.name)
    return {"models": files}


class AnalyzeRequest(BaseModel):
    model: Optional[str] = None
    model_data: Optional[dict] = None


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    """分析模型结构"""
    if req.model_data:
        return req.model_data

    if not req.model:
        raise HTTPException(status_code=400, detail="需要提供 model 参数")

    candidate = MODEL_DIR / req.model
    model_path = None
    if candidate.exists():
        model_path = str(candidate)
    elif os.path.isabs(req.model) and os.path.exists(req.model):
        model_path = req.model
    else:
        raise HTTPException(status_code=404, detail=f"找不到模型: {req.model}")

    ext = os.path.splitext(model_path)[1].lower()
    if ext == ".json":
        with open(model_path, "r") as f:
            return json.load(f)

    # 返回基本信息
    return {
        "message": f"模型 {req.model} 已准备好推理",
        "model": req.model,
        "type": ext.replace(".", ""),
        "ready": True,
    }


@app.post("/infer")
async def infer(
    model: str = Form(...),
    file: UploadFile = File(...),
):
    """图片推理 + 激活值提取"""
    # 读取图片
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="图片为空")

    # 查找模型
    candidate = MODEL_DIR / model
    model_path = None
    if candidate.exists():
        model_path = str(candidate)
    elif os.path.isabs(model) and os.path.exists(model):
        model_path = model
    else:
        raise HTTPException(status_code=404, detail=f"找不到模型: {model}")

    try:
        result = auto_infer(model_path, image_bytes)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


@app.post("/infer_base64")
def infer_base64(data: dict):
    """Base64 图片推理（备选接口）"""
    if "model" not in data or "image" not in data:
        raise HTTPException(status_code=400, detail="需要提供 model 和 image 字段")

    # 解码 base64 图片
    try:
        image_bytes = base64.b64decode(data["image"])
    except Exception:
        raise HTTPException(status_code=400, detail="无效的 Base64 图片数据")

    model = data["model"]

    # 查找模型
    candidate = MODEL_DIR / model
    model_path = None
    if candidate.exists():
        model_path = str(candidate)
    elif os.path.isabs(model) and os.path.exists(model):
        model_path = model
    else:
        raise HTTPException(status_code=404, detail=f"找不到模型: {model}")

    try:
        result = auto_infer(model_path, image_bytes)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("🚀 CNN Visualizer - Inference Server v2.0")
    print("=" * 50)
    print(f"📁 模型目录: {MODEL_DIR.absolute()}")
    print("   支持: PyTorch(.pt/.pth) | Keras(.h5/.keras) | ONNX(.onnx)")
    print()
    print("📌 API 接口:")
    print("   GET  /health         - 健康检查")
    print("   GET  /models         - 列出可用模型")
    print("   POST /analyze        - 分析模型结构")
    print("   POST /infer          - 图片推理（form-data）")
    print("   POST /infer_base64   - Base64 图片推理")
    print()
    print("💡 使用示例:")
    print('   curl -X POST "http://localhost:8000/infer" \\\\')
    print('     -F "model=resnet18.pth" \\\\')
    print('     -F "file=@test.jpg"')
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
`;

export default ModelImporter;