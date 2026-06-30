import { useState } from 'react';
import { useAISettings } from '../hooks/useAISettings';
import { AI_PROVIDER_PRESETS, QUICK_PROVIDERS } from '../types/ai';
import type { AIProvider } from '../types/ai';
import { useToast } from '../contexts/ToastContext';
import { DarkSelect } from '../components/ui/DarkSelect';

/* ============================================
   AISettingsPage — AI 分析接入设置
   - 主流服务商一键快速选择
   - API Key 显隐切换
   - 输入格式实时校验
   - 模型自定义输入
   ============================================ */

function isValidUrl(url: string): boolean {
  if (!url) return false;
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

export default function AISettingsPage() {
  const {
    settings,
    updateProvider,
    updateApiKey,
    updateEndpoint,
    updateModel,
    updateTemperature,
    updateMaxTokens,
    toggleEnabled,
    reset,
    isConfigured,
  } = useAISettings();

  const [showApiKey, setShowApiKey] = useState(false);
  const preset = AI_PROVIDER_PRESETS[settings.provider];
  const toast = useToast();

  const endpointValid = isValidUrl(settings.apiEndpoint);
  const showEndpointError = settings.apiEndpoint.length > 0 && !endpointValid;

  // 快速选择服务商
  const handleQuickSelect = (provider: AIProvider) => {
    updateProvider(provider);
    toast.showInfo('已切换服务商', `已填入 ${AI_PROVIDER_PRESETS[provider].label} 官方接口`);
  };

  // 保存配置
  const handleSave = () => {
    if (!endpointValid) {
      toast.showError('保存失败', '请输入有效的接口地址');
      return;
    }
    if (!settings.apiKey) {
      toast.showError('保存失败', 'API Key 不能为空');
      return;
    }
    toast.showSuccess('配置已保存', `${preset.label} · ${settings.model}`);
  };

  // 重置配置
  const handleReset = () => {
    reset();
    toast.showInfo('已重置为默认配置');
  };

  // 切换启用状态
  const handleToggleEnabled = () => {
    toggleEnabled();
    toast.showInfo(settings.enabled ? '已禁用 AI 分析' : '已启用 AI 分析');
  };

  return (
    <div className="flex h-full w-full flex-col overflow-hidden bg-[#0a0c14] text-white">
      {/* 顶部栏 */}
      <header className="flex items-center justify-between border-b border-white/[0.06] bg-[#0c0e17] px-5 py-3">
        <div className="flex items-center gap-3">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-gradient-to-br from-primary to-primary/60 shadow-lg shadow-primary/20">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="text-white">
              <circle cx="12" cy="12" r="3" />
              <path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41" />
            </svg>
          </div>
          <div>
            <h1 className="text-[13px] font-bold tracking-tight">AI 分析设置</h1>
            <p className="text-[10px] text-muted-foreground">配置外部 AI 供应商与 API 调用参数</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleSave}
            className="flex h-8 items-center gap-1.5 rounded-lg border border-primary/30 bg-primary/10 px-3 text-[11px] font-semibold text-primary transition-all hover:bg-primary/15"
          >
            <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <path d="M20 6L9 17l-5-5" />
            </svg>
            保存配置
          </button>
          <button
            onClick={handleToggleEnabled}
            className={`flex h-8 items-center gap-1.5 rounded-lg border px-3 text-[11px] font-semibold transition-all ${
              settings.enabled
                ? 'border-emerald-400/30 bg-emerald-400/10 text-emerald-400 hover:bg-emerald-400/15'
                : 'border-white/[0.08] bg-white/[0.02] text-foreground/85 hover:bg-white/[0.05]'
            }`}
          >
            <span className={`h-2 w-2 rounded-full ${settings.enabled ? 'bg-emerald-400' : 'bg-muted-foreground/40'}`} />
            {settings.enabled ? '已启用' : '已禁用'}
          </button>
          <button
            onClick={handleReset}
            className="flex h-8 items-center gap-1.5 rounded-lg border border-white/[0.08] bg-white/[0.02] px-3 text-[11px] font-semibold text-foreground/85 transition-all hover:bg-white/[0.05]"
          >
            重置
          </button>
        </div>
      </header>

      {/* 主内容 */}
      <main className="flex-1 overflow-y-auto p-5">
        <div className="mx-auto max-w-3xl">
          {/* 状态卡 */}
          <div className={`mb-5 rounded-xl border p-4 ${isConfigured ? 'border-emerald-400/20 bg-emerald-400/5' : 'border-amber-400/20 bg-amber-400/5'}`}>
            <div className="flex items-center gap-2 text-sm font-bold">
              <span className={`h-2 w-2 rounded-full ${isConfigured ? 'bg-emerald-400' : 'bg-amber-400'}`} />
              {isConfigured ? 'AI 分析已就绪' : 'AI 分析未配置'}
            </div>
            <p className="mt-1 text-[11px] text-muted-foreground">
              {isConfigured
                ? `已选择 ${preset.label}，模型 ${settings.model}。诊断面板将调用该端点进行分析。`
                : '请填写 API Key 并启用 AI 分析，以在 3D 工作台中使用智能诊断功能。'}
            </p>
          </div>

          <div className="space-y-4 rounded-xl border border-white/[0.06] bg-[#0c0e17] p-5">
            {/* 快速选择服务商 */}
            <div>
              <label className="mb-2 block text-[11px] font-semibold text-foreground/90">快速选择服务商</label>
              <div className="flex flex-wrap gap-2">
                {QUICK_PROVIDERS.map((provider) => {
                  const p = AI_PROVIDER_PRESETS[provider];
                  const isActive = settings.provider === provider;
                  return (
                    <button
                      key={provider}
                      onClick={() => handleQuickSelect(provider)}
                      className={`flex items-center gap-1.5 rounded-lg border px-3 py-2 text-[11px] font-medium transition-all ${
                        isActive
                          ? 'border-primary/30 bg-primary/10 text-primary'
                          : 'border-white/[0.06] bg-white/[0.02] text-muted-foreground hover:bg-white/[0.04] hover:text-foreground/80'
                      }`}
                    >
                      {p.label}
                    </button>
                  );
                })}
              </div>
              <p className="mt-1.5 text-[10px] text-muted-foreground/60">点击即可自动填入官方接口地址和推荐模型，仍可手动修改。</p>
            </div>

            <div className="border-t border-white/[0.04] pt-4">
              {/* 供应商（完整列表） */}
              <div className="mb-4">
                <label className="mb-1.5 block text-[11px] font-semibold text-foreground/90">AI 供应商</label>
                <div className="grid grid-cols-4 gap-2">
                  {(Object.keys(AI_PROVIDER_PRESETS) as AIProvider[]).map((provider) => (
                    <button
                      key={provider}
                      onClick={() => updateProvider(provider)}
                      className={`rounded-lg border px-3 py-2 text-[11px] font-medium transition-all ${
                        settings.provider === provider
                          ? 'border-primary/30 bg-primary/10 text-primary'
                          : 'border-white/[0.06] bg-white/[0.02] text-muted-foreground hover:bg-white/[0.04] hover:text-foreground/80'
                      }`}
                    >
                      {AI_PROVIDER_PRESETS[provider].label}
                    </button>
                  ))}
                </div>
              </div>

              {/* API Key */}
              <div className="mb-4">
                <label className="mb-1.5 block text-[11px] font-semibold text-foreground/90">API Key</label>
                <div className="relative">
                  <input
                    type={showApiKey ? 'text' : 'password'}
                    value={settings.apiKey}
                    onChange={(e) => updateApiKey(e.target.value)}
                    placeholder={preset.keyPlaceholder || 'sk-...'}
                    className={`w-full rounded-lg border bg-white/[0.02] px-3 py-2 pr-10 text-[11px] text-foreground outline-none transition placeholder:text-muted-foreground/40 focus:bg-white/[0.03] ${
                      settings.apiKey.length === 0 && settings.provider !== 'custom'
                        ? 'border-amber-400/30'
                        : 'border-white/[0.08] focus:border-primary/30'
                    }`}
                  />
                  <button
                    onClick={() => setShowApiKey(v => !v)}
                    className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground transition hover:text-foreground"
                    title={showApiKey ? '隐藏密钥' : '显示密钥'}
                  >
                    {showApiKey ? (
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M9.88 9.88a3 3 0 1 0 4.24 4.24" />
                        <path d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68" />
                        <path d="M6.61 6.61A13.526 13.526 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61" />
                        <line x1="2" y1="2" x2="22" y2="22" />
                      </svg>
                    ) : (
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z" />
                        <circle cx="12" cy="12" r="3" />
                      </svg>
                    )}
                  </button>
                </div>
                <div className="mt-1 flex items-center justify-between">
                  <p className="text-[10px] text-muted-foreground/60">仅在浏览器本地存储，不会上传到任何服务器。</p>
                  {preset.docsUrl && (
                    <a
                      href={preset.docsUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-[10px] text-primary/70 transition hover:text-primary"
                    >
                      获取 API Key →
                    </a>
                  )}
                </div>
                {settings.apiKey.length === 0 && settings.provider !== 'custom' && (
                  <p className="mt-1 text-[10px] text-amber-400/80">⚠ 请填写 API Key 以启用 AI 分析</p>
                )}
              </div>

              {/* Endpoint */}
              <div className="mb-4">
                <label className="mb-1.5 block text-[11px] font-semibold text-foreground/90">API Endpoint</label>
                <input
                  type="text"
                  value={settings.apiEndpoint}
                  onChange={(e) => updateEndpoint(e.target.value)}
                  placeholder="https://api.example.com/v1/chat/completions"
                  className={`w-full rounded-lg border bg-white/[0.02] px-3 py-2 text-[11px] text-foreground outline-none transition placeholder:text-muted-foreground/40 focus:bg-white/[0.03] ${
                    showEndpointError
                      ? 'border-red-400/50 focus:border-red-400/50'
                      : 'border-white/[0.08] focus:border-primary/30'
                  }`}
                />
                {showEndpointError ? (
                  <p className="mt-1 text-[10px] text-red-400/80">⚠ 请输入有效的接口地址（需以 http:// 或 https:// 开头）</p>
                ) : (
                  <p className="mt-1 text-[10px] text-muted-foreground/60">支持手动修改，兼容本地私有化部署、代理转发等场景。</p>
                )}
              </div>

              {/* Model */}
              <div className="mb-4">
                <label className="mb-1.5 block text-[11px] font-semibold text-foreground/90">模型</label>
                <div className="flex items-center gap-2">
                  <DarkSelect
                    options={preset.models.map((m: string) => ({ value: m, label: m }))}
                    value={settings.model}
                    onChange={updateModel}
                    className="flex-1"
                  />
                  <input
                    type="text"
                    value={settings.model}
                    onChange={(e) => updateModel(e.target.value)}
                    placeholder="或输入自定义模型名"
                    className="flex-1 rounded-lg border border-white/[0.08] bg-white/[0.02] px-3 py-2 text-[11px] text-foreground outline-none transition placeholder:text-muted-foreground/40 focus:border-primary/30 focus:bg-white/[0.03]"
                  />
                </div>
              </div>

              {/* Temperature & Max Tokens */}
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="mb-1.5 flex items-center justify-between text-[11px] font-semibold text-foreground/90">
                    <span>Temperature</span>
                    <span className="font-mono text-[10px] text-primary">{settings.temperature.toFixed(1)}</span>
                  </label>
                  <input
                    type="range"
                    min={0}
                    max={2}
                    step={0.1}
                    value={settings.temperature}
                    onChange={(e) => updateTemperature(Number(e.target.value))}
                    className="h-1 w-full cursor-pointer appearance-none rounded-full bg-white/10 accent-primary outline-none"
                  />
                  <p className="mt-1 text-[10px] text-muted-foreground/60">越低越稳定，越高越创造性。</p>
                </div>

                <div>
                  <label className="mb-1.5 block text-[11px] font-semibold text-foreground/90">Max Tokens</label>
                  <input
                    type="number"
                    min={256}
                    max={8192}
                    step={256}
                    value={settings.maxTokens}
                    onChange={(e) => updateMaxTokens(Number(e.target.value))}
                    className="w-full rounded-lg border border-white/[0.08] bg-white/[0.02] px-3 py-2 text-[11px] text-foreground outline-none transition focus:border-primary/30 focus:bg-white/[0.03]"
                  />
                </div>
              </div>
            </div>

            {/* 说明 */}
            <div className="rounded-lg border border-white/[0.04] bg-white/[0.02] p-3">
              <h3 className="mb-1 text-[11px] font-semibold text-foreground/90">调用说明</h3>
              <p className="text-[10px] leading-relaxed text-muted-foreground/70">
                配置完成后，3D 工作台中的 AI 诊断面板会携带当前网络结构、训练指标和激活数据，向上述端点发送请求。
                返回的结果将以结构化形式展示在右侧面板中。本地服务请确保 CORS 已正确配置。
              </p>
            </div>
          </div>
        </div>

        <div className="h-8" />
      </main>
    </div>
  );
}
