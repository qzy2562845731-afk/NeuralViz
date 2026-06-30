import { useState, useEffect } from 'react';
import apiService from '../services/api';
import { DarkSelect } from './ui/DarkSelect';

/* ============================================
   DeploymentConfig — 部署配置界面
   - 支持环境选择、资源配置、参数调整
   - 部署状态监控
   - 生成 Docker Compose 配置
   ============================================ */

interface DeploymentSettings {
  environment: 'local' | 'docker' | 'cloud';
  backend: {
    host: string;
    port: number;
    workers: number;
    debug: boolean;
  };
  frontend: {
    host: string;
    port: number;
    build_mode: 'development' | 'production';
  };
  database: {
    type: 'sqlite' | 'postgresql' | 'mysql';
    url: string;
  };
  resources: {
    cpu_limit: number;
    memory_limit_mb: number;
    gpu_enabled: boolean;
    gpu_device: string;
  };
  security: {
    cors_origins: string;
    api_key_enabled: boolean;
    api_key: string;
  };
}

const DEFAULT_SETTINGS: DeploymentSettings = {
  environment: 'local',
  backend: {
    host: '0.0.0.0',
    port: 8000,
    workers: 1,
    debug: true,
  },
  frontend: {
    host: '0.0.0.0',
    port: 5173,
    build_mode: 'development',
  },
  database: {
    type: 'sqlite',
    url: 'sqlite:///./neuralviz.db',
  },
  resources: {
    cpu_limit: 2,
    memory_limit_mb: 2048,
    gpu_enabled: false,
    gpu_device: '0',
  },
  security: {
    cors_origins: '*',
    api_key_enabled: false,
    api_key: '',
  },
};

interface DeploymentConfigProps {
  isOpen: boolean;
  onClose: () => void;
}

export function DeploymentConfig({ isOpen, onClose }: DeploymentConfigProps) {
  const [settings, setSettings] = useState<DeploymentSettings>(DEFAULT_SETTINGS);
  const [activeTab, setActiveTab] = useState<'config' | 'docker' | 'preview'>('config');
  const [backendStatus, setBackendStatus] = useState<'unknown' | 'running' | 'stopped'>('unknown');
  const [frontendStatus, setFrontendStatus] = useState<'unknown' | 'running' | 'stopped'>('unknown');

  // 检查服务状态
  const checkStatus = async () => {
    try {
      const resp = await apiService.getServiceStatus();
      setBackendStatus(resp?.data?.status === 'ok' ? 'running' : 'stopped');
    } catch {
      setBackendStatus('stopped');
    }
    try {
      const resp = await fetch('http://localhost:5173/', { method: 'HEAD' });
      setFrontendStatus(resp.ok ? 'running' : 'stopped');
    } catch {
      setFrontendStatus('stopped');
    }
  };

  useEffect(() => {
    if (isOpen) checkStatus();
  }, [isOpen]);

  const update = (section: keyof DeploymentSettings, key: string, value: any) => {
    setSettings(prev => ({
      ...prev,
      [section]: {
        ...(prev[section] as Record<string, any>),
        [key]: value,
      },
    }));
  };

  const generateDockerCompose = () => {
    const db = settings.database;
    const be = settings.backend;
    const fe = settings.frontend;
    const res = settings.resources;

    const gpuSection = res.gpu_enabled ? `
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]` : '';

    return `version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    container_name: neuralviz-backend
    ports:
      - "${be.port}:${be.port}"
    environment:
      - HOST=${be.host}
      - PORT=${be.port}
      - DEBUG=${be.debug}
      - DATABASE_URL=${db.url}
      - CORS_ORIGINS=${settings.security.cors_origins}
      ${settings.security.api_key_enabled ? `- API_KEY=${settings.security.api_key}` : ''}
    volumes:
      - ./backend/uploads:/app/uploads
      - ./backend/logs:/app/logs
    ${res.gpu_enabled ? 'runtime: nvidia' : ''}${gpuSection}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:${be.port}/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
      args:
        BUILD_MODE: ${fe.build_mode}
        VITE_API_BASE_URL: http://localhost:${be.port}
    container_name: neuralviz-frontend
    ports:
      - "${fe.port}:80"
    depends_on:
      backend:
        condition: service_healthy
    restart: unless-stopped

  ${db.type === 'postgresql' ? `
  postgres:
    image: postgres:15-alpine
    container_name: neuralviz-db
    environment:
      - POSTGRES_USER=neuralviz
      - POSTGRES_PASSWORD=neuralviz
      - POSTGRES_DB=neuralviz
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
    restart: unless-stopped
  ` : ''}`;
  };

  const copyDockerCompose = () => {
    navigator.clipboard.writeText(generateDockerCompose());
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="w-full max-w-3xl max-h-[85vh] overflow-hidden rounded-2xl border border-white/[0.08] bg-[#0f1119] shadow-2xl flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-white/[0.06] px-6 py-4">
          <div>
            <h2 className="text-base font-bold">部署配置</h2>
            <p className="mt-0.5 text-xs text-muted-foreground">配置生产环境部署参数，生成Docker编排文件</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={checkStatus}
              className="flex h-8 items-center gap-1.5 rounded-md border border-white/[0.08] bg-white/[0.02] px-2.5 text-xs text-muted-foreground transition-all hover:bg-white/[0.05] hover:text-foreground"
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M1 4v6h6M23 20v-6h-6" />
                <path d="M20.49 9A9 9 0 005.64 5.64L1 10m22 4l-4.64 4.36A9 9 0 013.51 15" />
              </svg>
              检测状态
            </button>
            <button
              onClick={onClose}
              className="flex h-8 w-8 items-center justify-center rounded-md border border-white/[0.08] bg-white/[0.02] text-muted-foreground transition-all hover:text-foreground"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 6 6 18M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>

        {/* Tabs */}
        <div className="flex border-b border-white/[0.06] px-6">
          {(['config', 'docker', 'preview'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 text-xs font-medium transition-all border-b-2 ${
                activeTab === tab
                  ? 'border-primary text-primary'
                  : 'border-transparent text-muted-foreground hover:text-foreground'
              }`}
            >
              {{ config: '参数配置', docker: 'Docker Compose', preview: '部署预览' }[tab]}
            </button>
          ))}
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-6">
          {activeTab === 'config' && (
            <ConfigTab settings={settings} update={update} />
          )}

          {activeTab === 'docker' && (
            <DockerTab settings={settings} generateDockerCompose={generateDockerCompose} copyDockerCompose={copyDockerCompose} />
          )}

          {activeTab === 'preview' && (
            <PreviewTab backendStatus={backendStatus} frontendStatus={frontendStatus} settings={settings} />
          )}
        </div>

        {/* Footer */}
        <div className="flex items-center justify-between border-t border-white/[0.06] px-6 py-3">
          <button
            onClick={() => setSettings(DEFAULT_SETTINGS)}
            className="text-xs text-muted-foreground transition-all hover:text-red-400"
          >
            恢复默认
          </button>
          <div className="flex gap-2">
            <button
              onClick={onClose}
              className="rounded-md border border-white/[0.08] bg-white/[0.02] px-4 py-1.5 text-xs text-muted-foreground transition-all hover:bg-white/[0.05] hover:text-foreground"
            >
              关闭
            </button>
            <button
              onClick={() => {
                // 保存配置
                try {
                  localStorage.setItem('neuralviz_deployment', JSON.stringify(settings));
                  onClose();
                } catch {}
              }}
              className="rounded-md bg-primary px-4 py-1.5 text-xs font-medium text-white transition-all hover:bg-primary/90"
            >
              保存配置
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ====== 参数配置 Tab ====== */
function ConfigTab({
  settings, update,
}: {
  settings: DeploymentSettings;
  update: (section: keyof DeploymentSettings, key: string, value: any) => void;
}) {
  return (
    <div className="space-y-5">
      {/* 环境选择 */}
      <Section title="部署环境">
        <div className="flex gap-2">
          {([
            { value: 'local', label: '本地开发', desc: 'localhost' },
            { value: 'docker', label: 'Docker', desc: '容器化' },
            { value: 'cloud', label: '云服务器', desc: '生产环境' },
          ] as const).map(env => (
            <button
              key={env.value}
              onClick={() => update('environment', 'value', env.value)}
              className={`flex-1 rounded-lg border p-3 text-left transition-all ${
                settings.environment === env.value
                  ? 'border-primary/40 bg-primary/10'
                  : 'border-white/[0.08] bg-white/[0.02] hover:bg-white/[0.04]'
              }`}
            >
              <div className="text-xs font-medium text-foreground/80">{env.label}</div>
              <div className="text-[10px] text-muted-foreground">{env.desc}</div>
            </button>
          ))}
        </div>
      </Section>

      {/* 后端配置 */}
      <Section title="后端服务">
        <div className="grid grid-cols-2 gap-3">
          <FormField label="主机地址" value={settings.backend.host} onChange={v => update('backend', 'host', v)} />
          <FormField label="端口" value={String(settings.backend.port)} onChange={v => update('backend', 'port', parseInt(v) || 8000)} type="number" />
          <FormField label="工作进程数" value={String(settings.backend.workers)} onChange={v => update('backend', 'workers', parseInt(v) || 1)} type="number" />
          <div className="flex items-center gap-2">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.backend.debug}
                onChange={e => update('backend', 'debug', e.target.checked)}
                className="h-3.5 w-3.5 rounded border-white/[0.15] bg-white/[0.04] accent-primary"
              />
              <span className="text-xs text-foreground/70">Debug模式</span>
            </label>
          </div>
        </div>
      </Section>

      {/* 前端配置 */}
      <Section title="前端服务">
        <div className="grid grid-cols-2 gap-3">
          <FormField label="端口" value={String(settings.frontend.port)} onChange={v => update('frontend', 'port', parseInt(v) || 5173)} type="number" />
          <div>
            <label className="mb-1 block text-[10px] text-muted-foreground">构建模式</label>
            <DarkSelect
              options={[
                { value: 'development', label: '开发模式 (HMR)' },
                { value: 'production', label: '生产模式 (优化构建)' },
              ]}
              value={settings.frontend.build_mode}
              onChange={(v) => update('frontend', 'build_mode', v)}
            />
          </div>
        </div>
      </Section>

      {/* 数据库配置 */}
      <Section title="数据库">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="mb-1 block text-[10px] text-muted-foreground">数据库类型</label>
            <DarkSelect
              options={[
                { value: 'sqlite', label: 'SQLite (文件)' },
                { value: 'postgresql', label: 'PostgreSQL' },
                { value: 'mysql', label: 'MySQL' },
              ]}
              value={settings.database.type}
              onChange={(v) => update('database', 'type', v)}
            />
          </div>
          <FormField label="连接URL" value={settings.database.url} onChange={v => update('database', 'url', v)} />
        </div>
      </Section>

      {/* 资源配置 */}
      <Section title="计算资源">
        <div className="grid grid-cols-2 gap-3">
          <FormField label="CPU限制(核)" value={String(settings.resources.cpu_limit)} onChange={v => update('resources', 'cpu_limit', parseFloat(v) || 1)} type="number" />
          <FormField label="内存限制(MB)" value={String(settings.resources.memory_limit_mb)} onChange={v => update('resources', 'memory_limit_mb', parseInt(v) || 1024)} type="number" />
          <div className="col-span-2 flex items-center gap-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.resources.gpu_enabled}
                onChange={e => update('resources', 'gpu_enabled', e.target.checked)}
                className="h-3.5 w-3.5 rounded border-white/[0.15] bg-white/[0.04] accent-green-500"
              />
              <span className="text-xs text-foreground/70">启用GPU加速</span>
            </label>
            {settings.resources.gpu_enabled && (
              <FormField label="GPU设备" value={settings.resources.gpu_device} onChange={v => update('resources', 'gpu_device', v)} />
            )}
          </div>
        </div>
      </Section>

      {/* 安全配置 */}
      <Section title="安全配置">
        <div className="grid grid-cols-2 gap-3">
          <FormField label="CORS允许源" value={settings.security.cors_origins} onChange={v => update('security', 'cors_origins', v)} />
          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={settings.security.api_key_enabled}
                onChange={e => update('security', 'api_key_enabled', e.target.checked)}
                className="h-3.5 w-3.5 rounded border-white/[0.15] bg-white/[0.04] accent-amber-500"
              />
              <span className="text-xs text-foreground/70">API密钥认证</span>
            </label>
          </div>
          {settings.security.api_key_enabled && (
            <FormField label="API密钥" value={settings.security.api_key} onChange={v => update('security', 'api_key', v)} />
          )}
        </div>
      </Section>
    </div>
  );
}

/* ====== Docker Compose Tab ====== */
function DockerTab({
  generateDockerCompose, copyDockerCompose,
}: {
  settings: DeploymentSettings;  // eslint-disable-line @typescript-eslint/no-unused-vars
  generateDockerCompose: () => string;
  copyDockerCompose: () => void;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    copyDockerCompose();
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <p className="text-xs text-muted-foreground">
          根据当前配置自动生成的 Docker Compose 编排文件
        </p>
        <button
          onClick={handleCopy}
          className="flex items-center gap-1 rounded-md border border-white/[0.08] bg-white/[0.02] px-3 py-1 text-xs text-muted-foreground transition-all hover:bg-white/[0.05] hover:text-foreground"
        >
          {copied ? (
            <>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="20 6 9 17 4 12" />
              </svg>
              已复制
            </>
          ) : (
            <>
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" />
              </svg>
              复制
            </>
          )}
        </button>
      </div>
      <pre className="rounded-lg border border-white/[0.08] bg-[#0a0c14] p-4 text-[11px] font-mono text-green-400/80 overflow-x-auto whitespace-pre">
        {generateDockerCompose()}
      </pre>
    </div>
  );
}

/* ====== 部署预览 Tab ====== */
function PreviewTab({
  backendStatus, frontendStatus, settings,
}: {
  backendStatus: string;
  frontendStatus: string;
  settings: DeploymentSettings;
}) {
  const statusBadge = (status: string) => {
    const map: Record<string, { color: string; label: string; dot: string }> = {
      running: { color: 'bg-green-500/20 text-green-400', label: '运行中', dot: 'bg-green-400' },
      stopped: { color: 'bg-red-500/20 text-red-400', label: '已停止', dot: 'bg-red-400' },
      unknown: { color: 'bg-gray-500/20 text-gray-400', label: '未知', dot: 'bg-gray-400' },
    };
    const s = map[status] || map.unknown;
    return (
      <span className={`inline-flex items-center gap-1.5 rounded px-2 py-0.5 text-xs ${s.color}`}>
        <span className={`h-1.5 w-1.5 rounded-full ${s.dot}`} />
        {s.label}
      </span>
    );
  };

  return (
    <div className="space-y-4">
      {/* 服务状态 */}
      <div className="rounded-lg border border-white/[0.08] bg-white/[0.01] p-4">
        <h3 className="mb-3 text-xs font-medium text-foreground/80">服务状态</h3>
        <div className="grid grid-cols-2 gap-4">
          <div className="flex items-center justify-between rounded-md border border-white/[0.06] px-3 py-2">
            <div>
              <div className="text-xs font-medium text-foreground/80">后端服务</div>
              <div className="text-[10px] text-muted-foreground">http://localhost:{settings.backend.port}</div>
            </div>
            {statusBadge(backendStatus)}
          </div>
          <div className="flex items-center justify-between rounded-md border border-white/[0.06] px-3 py-2">
            <div>
              <div className="text-xs font-medium text-foreground/80">前端服务</div>
              <div className="text-[10px] text-muted-foreground">http://localhost:{settings.frontend.port}</div>
            </div>
            {statusBadge(frontendStatus)}
          </div>
        </div>
      </div>

      {/* 部署摘要 */}
      <div className="rounded-lg border border-white/[0.08] bg-white/[0.01] p-4">
        <h3 className="mb-3 text-xs font-medium text-foreground/80">部署摘要</h3>
        <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
          <div className="flex justify-between">
            <span className="text-muted-foreground">环境:</span>
            <span className="text-foreground/80 font-medium">
              {{ local: '本地开发', docker: 'Docker容器', cloud: '云服务器' }[settings.environment]}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">数据库:</span>
            <span className="text-foreground/80 font-medium">{settings.database.type.toUpperCase()}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">CPU限制:</span>
            <span className="text-foreground/80 font-medium">{settings.resources.cpu_limit} 核</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">内存限制:</span>
            <span className="text-foreground/80 font-medium">{settings.resources.memory_limit_mb} MB</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">GPU:</span>
            <span className="text-foreground/80 font-medium">{settings.resources.gpu_enabled ? '已启用' : '未启用'}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">API认证:</span>
            <span className="text-foreground/80 font-medium">{settings.security.api_key_enabled ? '已启用' : '未启用'}</span>
          </div>
        </div>
      </div>

      {/* 快速命令 */}
      <div className="rounded-lg border border-white/[0.08] bg-white/[0.01] p-4">
        <h3 className="mb-3 text-xs font-medium text-foreground/80">快速启动命令</h3>
        <div className="space-y-2">
          <div>
            <div className="text-[10px] text-muted-foreground mb-1">Docker部署:</div>
            <pre className="rounded bg-[#0a0c14] px-3 py-1.5 text-[11px] font-mono text-green-400/80">
              docker-compose up -d
            </pre>
          </div>
          <div>
            <div className="text-[10px] text-muted-foreground mb-1">本地开发:</div>
            <pre className="rounded bg-[#0a0c14] px-3 py-1.5 text-[11px] font-mono text-green-400/80">
              {'# 终端1: 启动后端\ncd backend && python main.py\n\n# 终端2: 启动前端\nnpm run dev'}
            </pre>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ====== 辅助组件 ====== */
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div>
      <h3 className="mb-2 text-xs font-semibold text-foreground/70">{title}</h3>
      {children}
    </div>
  );
}

function FormField({
  label, value, onChange, type = 'text',
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  type?: string;
}) {
  return (
    <div>
      <label className="mb-1 block text-[10px] text-muted-foreground">{label}</label>
      <input
        type={type}
        value={value}
        onChange={e => onChange(e.target.value)}
        className="w-full rounded-md border border-white/[0.08] bg-white/[0.02] px-2.5 py-1.5 text-xs text-foreground focus:border-primary/40 focus:outline-none font-mono"
      />
    </div>
  );
}