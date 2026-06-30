export type AIProvider = 'openai' | 'anthropic' | 'deepseek' | 'zhipu' | 'qwen' | 'wenxin' | 'custom';

export interface AISettings {
  provider: AIProvider;
  apiKey: string;
  apiEndpoint: string;
  model: string;
  temperature: number;
  maxTokens: number;
  enabled: boolean;
}

export interface AIProviderPreset {
  label: string;
  endpoint: string;
  models: string[];
  docsUrl?: string;
  keyPlaceholder?: string;
}

export const AI_PROVIDER_PRESETS: Record<AIProvider, AIProviderPreset> = {
  openai: {
    label: 'OpenAI',
    endpoint: 'https://api.openai.com/v1/chat/completions',
    models: ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-3.5-turbo'],
    docsUrl: 'https://platform.openai.com/api-keys',
    keyPlaceholder: 'sk-...',
  },
  anthropic: {
    label: 'Anthropic',
    endpoint: 'https://api.anthropic.com/v1/messages',
    models: ['claude-3-5-sonnet-20241022', 'claude-3-opus-20240229', 'claude-3-haiku-20240307'],
    docsUrl: 'https://console.anthropic.com/settings/keys',
    keyPlaceholder: 'sk-ant-...',
  },
  deepseek: {
    label: 'DeepSeek',
    endpoint: 'https://api.deepseek.com/v1/chat/completions',
    models: ['deepseek-chat', 'deepseek-reasoner', 'deepseek-coder'],
    docsUrl: 'https://platform.deepseek.com/api_keys',
    keyPlaceholder: 'sk-...',
  },
  zhipu: {
    label: '智谱 AI',
    endpoint: 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
    models: ['glm-4', 'glm-4-flash', 'glm-4-air', 'glm-3-turbo'],
    docsUrl: 'https://open.bigmodel.cn/usercenter/apikeys',
    keyPlaceholder: 'xxx.xxx',
  },
  qwen: {
    label: '通义千问',
    endpoint: 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
    models: ['qwen-max', 'qwen-plus', 'qwen-turbo', 'qwen-long'],
    docsUrl: 'https://dashscope.console.aliyun.com/apiKey',
    keyPlaceholder: 'sk-...',
  },
  wenxin: {
    label: '文心一言',
    endpoint: 'https://qianfan.baidubce.com/v2/chat/completions',
    models: ['ernie-4.0-8k', 'ernie-4.0-turbo-8k', 'ernie-speed-128k', 'ernie-lite-8k'],
    docsUrl: 'https://console.bce.baidu.com/qianfan/ais/console/applicationConsole/application',
    keyPlaceholder: 'Bearer token',
  },
  custom: {
    label: '自定义 / 本地服务',
    endpoint: 'http://localhost:8000/v1/chat/completions',
    models: ['local-model', 'custom-model'],
    keyPlaceholder: '本地服务可留空',
  },
};

// 快速选择服务商列表（用于自定义板块顶部展示）
export const QUICK_PROVIDERS: AIProvider[] = ['deepseek', 'zhipu', 'qwen', 'wenxin', 'openai', 'anthropic'];

export const DEFAULT_AI_SETTINGS: AISettings = {
  provider: 'custom',
  apiKey: '',
  apiEndpoint: AI_PROVIDER_PRESETS.custom.endpoint,
  model: AI_PROVIDER_PRESETS.custom.models[0],
  temperature: 0.3,
  maxTokens: 2048,
  enabled: false,
};
