/* ============================================
   aiService — AI 大模型分析服务
   - 替代本地规则分析，调用真实 LLM API 进行网络训练诊断
   - 支持 OpenAI / Anthropic / DeepSeek / 智谱 / 通义千问 / 文心 / 自定义
   - 除 Anthropic 外均使用 OpenAI 兼容的 chat/completions 格式
   ============================================ */

import type { AISettings } from '../types/ai';

/* ---------- localStorage 存储键（与 useAISettings 保持一致） ---------- */
const AI_SETTINGS_STORAGE_KEY = 'nn-ai-settings-v1';

/* ---------- 请求超时时间（毫秒） ---------- */
const REQUEST_TIMEOUT_MS = 30000;

/* ---------- Anthropic 服务商标识 ---------- */
const ANTHROPIC_PROVIDER = 'anthropic';

/* ---------- 各分析级别对应的 max_tokens 上限 ---------- */
const LEVEL_MAX_TOKENS: Record<AnalysisLevel, number> = {
  brief: 200,
  standard: 500,
  deep: 1500,
};

/* ============================================
   类型定义
   ============================================ */

/** 建议类型 */
export type AISuggestionType = 'info' | 'warning' | 'success' | 'critical';

/** 分析级别 */
export type AnalysisLevel = 'brief' | 'standard' | 'deep';

/** 单条 AI 建议 */
export interface AISuggestion {
  type: AISuggestionType;
  title: string;
  description: string;
  layerId?: string;
  layerType?: string;
  confidence: number;
}

/** AI 分析入参 */
export interface AIAnalysisParams {
  architecture: {
    name: string;
    layers: Array<{
      id: string;
      name: string;
      type: string;
      params: number;
      nodeCount?: number;
      kernelSize?: number;
      outputShape?: number[];
    }>;
  };
  currentStep: number;
  isPlaying: boolean;
  analysisLevel: AnalysisLevel;
}

/** buildAnalysisPrompt 返回的 prompt 结构 */
interface BuiltPrompt {
  systemPrompt: string;
  userPrompt: string;
  maxTokens: number;
}

/* ============================================
   设置读取与校验
   ============================================ */

/**
 * 从 localStorage 读取 AI 设置
 * @returns 解析后的 AISettings；不存在或解析失败时返回 null
 */
export function loadAISettings(): AISettings | null {
  try {
    const raw = localStorage.getItem(AI_SETTINGS_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as AISettings;
    return parsed;
  } catch {
    // 解析失败（JSON 格式错误或 localStorage 不可用）时返回 null
    return null;
  }
}

/**
 * 检查 AI 是否已完成有效配置
 * 需同时满足：enabled=true、apiKey 非空、apiEndpoint 非空、model 非空
 * @returns 是否已配置
 */
export function isAIConfigured(): boolean {
  const settings = loadAISettings();
  if (!settings) return false;
  return Boolean(
    settings.enabled &&
      settings.apiKey &&
      settings.apiEndpoint &&
      settings.model
  );
}

/* ============================================
   Prompt 构建
   ============================================ */

/**
 * 构建发送给 LLM 的 prompt
 * - system prompt：设定 AI 为神经网络训练分析专家，要求返回 JSON 数组
 * - user prompt：包含网络架构摘要、层类型分布、训练状态
 * - 根据分析级别调整详细程度与 max_tokens
 * @param params 分析入参
 * @returns 包含 systemPrompt / userPrompt / maxTokens 的对象
 */
export function buildAnalysisPrompt(params: AIAnalysisParams): BuiltPrompt {
  const { architecture, currentStep, isPlaying, analysisLevel } = params;
  const maxTokens = LEVEL_MAX_TOKENS[analysisLevel];

  /* ---------- system prompt：根据级别调整详细程度 ---------- */
  const baseSystem = `你是一位资深的神经网络训练分析专家，精通 CNN、RNN、Transformer 等各类网络架构的训练诊断与优化。

请根据用户提供的网络架构与训练状态，给出结构化、可操作的分析建议。

【输出格式要求】
必须返回一个 JSON 数组，数组中每个元素包含以下字段：
- type: 建议类型，取值为 "info" | "warning" | "success" | "critical"
- title: 建议标题（简短，不超过 20 字）
- description: 建议详细描述（结合具体数据说明）
- layerId: 关联的层 ID（可选，若针对特定层）
- layerType: 关联的层类型（可选）
- confidence: 置信度，0 到 1 之间的数字

只返回 JSON 数组本身，不要包含任何解释文字、markdown 代码块标记或前后缀。`;

  const levelInstruction: Record<AnalysisLevel, string> = {
    brief: `\n\n【本次分析级别：精简】
请仅给出 2-3 条最核心的结论，聚焦最重要的结构问题与优化方向，描述简练。`,
    standard: `\n\n【本次分析级别：标准】
请给出 4-6 条分析建议，覆盖结构健康度、参数分布、训练进度等维度。`,
    deep: `\n\n【本次分析级别：深度】
请进行深度分析，给出 6-10 条建议，需覆盖：每层参数占比评估、潜在瓶颈层、调参方案（学习率/batch_size）、训练路线图（初期/中期/后期建议）、过拟合/欠拟合风险与正则化策略。`,
  };

  const systemPrompt = baseSystem + levelInstruction[analysisLevel];

  /* ---------- user prompt：网络架构摘要 + 训练状态 ---------- */
  const layers = architecture.layers;
  const totalParams = layers.reduce((sum, l) => sum + (l.params || 0), 0);

  // 层类型分布统计
  const typeDistribution = layers.reduce<Record<string, number>>((acc, l) => {
    acc[l.type] = (acc[l.type] || 0) + 1;
    return acc;
  }, {});
  const distributionText = Object.entries(typeDistribution)
    .map(([t, c]) => `${t}: ${c}`)
    .join('，');

  // 各层明细（深度级别输出更详细）
  const layerDetails = layers
    .map((l, idx) => {
      const parts = [
        `${idx + 1}. ${l.name} [${l.type}]`,
        `参数 ${(l.params || 0).toLocaleString()}`,
      ];
      if (l.nodeCount !== undefined) parts.push(`节点 ${l.nodeCount}`);
      if (l.kernelSize !== undefined) parts.push(`核 ${l.kernelSize}`);
      if (l.outputShape && l.outputShape.length > 0) {
        parts.push(`输出 ${l.outputShape.join('×')}`);
      }
      return parts.join('，');
    })
    .join('\n');

  const userPrompt = `【网络架构】
名称：${architecture.name}
总层数：${layers.length}
总参数量：${totalParams.toLocaleString()}

【层类型分布】
${distributionText}

【各层明细】
${layerDetails}

【训练状态】
当前训练步数：${currentStep}
训练状态：${isPlaying ? '训练中' : '已暂停 / 未启动'}

请基于以上信息进行分析，返回 JSON 数组格式的建议。`;

  return { systemPrompt, userPrompt, maxTokens };
}

/* ============================================
   响应解析
   ============================================ */

/**
 * 将 LLM 返回的文本解析为结构化建议数组
 * - 优先尝试解析 JSON 数组（兼容 markdown 代码块包裹）
 * - JSON 解析失败时按段落分割为纯文本建议
 * @param text LLM 返回的原始文本
 * @returns 结构化建议数组
 */
export function parseAIResponse(text: string): AISuggestion[] {
  if (!text || !text.trim()) {
    return [];
  }

  const trimmed = text.trim();

  /* ---------- 尝试提取并解析 JSON ---------- */
  const jsonText = extractJsonArray(trimmed);
  if (jsonText !== null) {
    try {
      const parsed = JSON.parse(jsonText);
      const suggestions = normalizeParsedSuggestions(parsed);
      if (suggestions.length > 0) {
        return suggestions;
      }
    } catch {
      // JSON 解析失败，降级到纯文本处理
    }
  }

  /* ---------- 降级：按段落分割为文本建议 ---------- */
  return parsePlainTextSuggestions(trimmed);
}

/**
 * 从文本中提取 JSON 数组字符串
 * 兼容三种情况：纯 JSON、markdown 代码块包裹、JSON 前后有多余文字
 * @returns 提取到的 JSON 字符串；未找到返回 null
 */
function extractJsonArray(text: string): string | null {
  // 1. 去除 markdown 代码块标记 ```json ... ``` 或 ``` ... ```
  const codeBlockMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (codeBlockMatch && codeBlockMatch[1]) {
    return codeBlockMatch[1].trim();
  }

  // 2. 直接匹配最外层的 JSON 数组 [...]（贪婪匹配最外层括号）
  const arrayMatch = text.match(/\[[\s\S]*\]/);
  if (arrayMatch) {
    return arrayMatch[0];
  }

  // 3. 匹配单个 JSON 对象 {...}（部分模型可能返回单对象而非数组）
  const objectMatch = text.match(/\{[\s\S]*\}/);
  if (objectMatch) {
    return objectMatch[0];
  }

  return null;
}

/**
 * 将解析后的 JSON 数据标准化为 AISuggestion 数组
 * 兼容数组、单对象、含 suggestions 字段的对象等结构
 */
function normalizeParsedSuggestions(parsed: unknown): AISuggestion[] {
  let rawList: unknown[] = [];

  if (Array.isArray(parsed)) {
    rawList = parsed;
  } else if (parsed && typeof parsed === 'object') {
    const obj = parsed as Record<string, unknown>;
    // 兼容 { suggestions: [...] } 或 { data: [...] } 等包装
    if (Array.isArray(obj.suggestions)) {
      rawList = obj.suggestions;
    } else if (Array.isArray(obj.data)) {
      rawList = obj.data;
    } else if (Array.isArray(obj.results)) {
      rawList = obj.results;
    } else {
      // 单个对象视为一条建议
      rawList = [obj];
    }
  }

  return rawList
    .map((item) => normalizeSingleSuggestion(item))
    .filter((s): s is AISuggestion => s !== null);
}

/**
 * 将单个解析对象标准化为 AISuggestion
 * 进行字段校验与类型归一化，无效数据返回 null
 */
function normalizeSingleSuggestion(item: unknown): AISuggestion | null {
  if (!item || typeof item !== 'object') return null;
  const obj = item as Record<string, unknown>;

  const title = typeof obj.title === 'string' ? obj.title.trim() : '';
  const description =
    typeof obj.description === 'string'
      ? obj.description.trim()
      : typeof obj.content === 'string'
        ? obj.content.trim()
        : '';

  // title 与 description 至少有一个非空才视为有效
  if (!title && !description) return null;

  const type = normalizeSuggestionType(obj.type);
  const confidence = normalizeConfidence(obj.confidence);

  const suggestion: AISuggestion = {
    type,
    title: title || description.slice(0, 20),
    description,
    confidence,
  };

  if (typeof obj.layerId === 'string' && obj.layerId) {
    suggestion.layerId = obj.layerId;
  }
  if (typeof obj.layerType === 'string' && obj.layerType) {
    suggestion.layerType = obj.layerType;
  }

  return suggestion;
}

/** 归一化建议类型，非法值降级为 info */
function normalizeSuggestionType(raw: unknown): AISuggestionType {
  const valid: AISuggestionType[] = ['info', 'warning', 'success', 'critical'];
  if (typeof raw === 'string' && valid.includes(raw as AISuggestionType)) {
    return raw as AISuggestionType;
  }
  return 'info';
}

/** 归一化置信度，限制在 [0, 1] 区间 */
function normalizeConfidence(raw: unknown): number {
  if (typeof raw !== 'number' || isNaN(raw)) return 0.5;
  return Math.max(0, Math.min(1, raw));
}

/**
 * 将纯文本按段落分割为建议数组
 * 每个非空段落作为一条 info 类型建议
 */
function parsePlainTextSuggestions(text: string): AISuggestion[] {
  const paragraphs = text
    .split(/\n\s*\n/)
    .map((p) => p.trim())
    .filter((p) => p.length > 0);

  if (paragraphs.length === 0) return [];

  return paragraphs.map((para, idx) => {
    // 取第一行或前 20 字作为标题
    const firstLine = para.split('\n')[0].trim();
    const title = firstLine.length > 20 ? firstLine.slice(0, 20) + '…' : firstLine || `建议 ${idx + 1}`;
    return {
      type: 'info' as AISuggestionType,
      title,
      description: para,
      confidence: 0.5,
    };
  });
}

/* ============================================
   核心：调用 LLM API
   ============================================ */

/**
 * 调用 LLM API 进行神经网络训练分析
 * @param params 分析入参（架构、训练状态、分析级别）
 * @returns 结构化建议数组
 * @throws 网络错误 / API 错误 / 解析错误时抛出带描述的 Error
 */
export async function callAIAnalysis(params: AIAnalysisParams): Promise<AISuggestion[]> {
  // 1. 读取并校验配置
  const settings = loadAISettings();
  if (!settings) {
    throw new Error('未找到 AI 配置，请先在设置页面配置 AI 服务商');
  }
  if (!isAIConfigured()) {
    throw new Error('AI 配置不完整，请检查 enabled / apiKey / apiEndpoint / model');
  }

  // 2. 构建 prompt
  const { systemPrompt, userPrompt, maxTokens } = buildAnalysisPrompt(params);

  // 3. 构建请求（区分 Anthropic 与 OpenAI 兼容格式）
  const isAnthropic = settings.provider === ANTHROPIC_PROVIDER;
  const requestBody = isAnthropic
    ? buildAnthropicRequestBody(settings, systemPrompt, userPrompt, maxTokens)
    : buildOpenAIRequestBody(settings, userPrompt, systemPrompt, maxTokens);

  const headers = buildRequestHeaders(settings, isAnthropic);

  // 4. 发起请求（带 30 秒超时）
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  let response: Response;
  try {
    response = await fetch(settings.apiEndpoint, {
      method: 'POST',
      headers,
      body: JSON.stringify(requestBody),
      signal: controller.signal,
    });
  } catch (err: unknown) {
    clearTimeout(timeoutId);
    throw handleFetchError(err);
  }
  clearTimeout(timeoutId);

  // 5. 校验 HTTP 状态码
  if (!response.ok) {
    const errMsg = await extractApiErrorMessage(response, isAnthropic);
    throw new Error(`AI 服务返回错误 (HTTP ${response.status})：${errMsg}`);
  }

  // 6. 解析响应 JSON
  let responseData: unknown;
  try {
    responseData = await response.json();
  } catch {
    throw new Error('AI 服务返回的内容不是有效的 JSON');
  }

  // 7. 提取文本内容
  const contentText = extractContentText(responseData, isAnthropic);
  if (!contentText) {
    throw new Error('AI 服务返回的内容为空，无法提取文本');
  }

  // 8. 解析为结构化建议
  return parseAIResponse(contentText);
}

/* ============================================
   请求体与请求头构建
   ============================================ */

/**
 * 构建 OpenAI 兼容格式的请求体
 * system prompt 作为 messages 数组的首条 system 消息
 */
function buildOpenAIRequestBody(
  settings: AISettings,
  userPrompt: string,
  systemPrompt: string,
  maxTokens: number
): Record<string, unknown> {
  return {
    model: settings.model,
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: userPrompt },
    ],
    temperature: settings.temperature,
    max_tokens: maxTokens,
  };
}

/**
 * 构建 Anthropic 格式的请求体
 * system prompt 作为顶层 system 字段
 */
function buildAnthropicRequestBody(
  settings: AISettings,
  systemPrompt: string,
  userPrompt: string,
  maxTokens: number
): Record<string, unknown> {
  return {
    model: settings.model,
    system: systemPrompt,
    messages: [{ role: 'user', content: userPrompt }],
    temperature: settings.temperature,
    max_tokens: maxTokens,
  };
}

/**
 * 构建请求头
 * - Anthropic：x-api-key + anthropic-version
 * - 其他：Authorization: Bearer {apiKey}
 */
function buildRequestHeaders(
  settings: AISettings,
  isAnthropic: boolean
): Record<string, string> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  if (isAnthropic) {
    headers['x-api-key'] = settings.apiKey;
    headers['anthropic-version'] = '2023-06-01';
  } else {
    headers['Authorization'] = `Bearer ${settings.apiKey}`;
  }

  return headers;
}

/* ============================================
   响应处理
   ============================================ */

/**
 * 从响应数据中提取文本内容
 * - OpenAI 兼容：choices[0].message.content
 * - Anthropic：content[0].text
 */
function extractContentText(data: unknown, isAnthropic: boolean): string {
  if (!data || typeof data !== 'object') return '';
  const obj = data as Record<string, unknown>;

  if (isAnthropic) {
    // Anthropic 格式：{ content: [{ type: 'text', text: '...' }] }
    const content = obj.content;
    if (Array.isArray(content)) {
      const textBlock = content.find(
        (block) =>
          block && typeof block === 'object' && (block as Record<string, unknown>).type === 'text'
      ) as Record<string, unknown> | undefined;
      if (textBlock && typeof textBlock.text === 'string') {
        return textBlock.text;
      }
      // 兜底：取第一个块的 text 字段
      const firstBlock = content[0] as Record<string, unknown> | undefined;
      if (firstBlock && typeof firstBlock.text === 'string') {
        return firstBlock.text;
      }
    }
    return '';
  }

  // OpenAI 兼容格式：{ choices: [{ message: { content: '...' } }] }
  const choices = obj.choices;
  if (Array.isArray(choices) && choices.length > 0) {
    const firstChoice = choices[0] as Record<string, unknown> | undefined;
    const message = firstChoice?.message as Record<string, unknown> | undefined;
    if (message && typeof message.content === 'string') {
      return message.content;
    }
  }

  return '';
}

/**
 * 从错误响应中提取可读的错误信息
 */
async function extractApiErrorMessage(
  response: Response,
  isAnthropic: boolean
): Promise<string> {
  try {
    const errData = await response.json();
    if (!errData || typeof errData !== 'object') return response.statusText;

    const obj = errData as Record<string, unknown>;

    // Anthropic 错误格式：{ error: { type, message } }
    if (isAnthropic && obj.error && typeof obj.error === 'object') {
      const err = obj.error as Record<string, unknown>;
      if (typeof err.message === 'string') return err.message;
    }

    // OpenAI 兼容错误格式：{ error: { message } } 或 { error: { code, message } }
    if (obj.error && typeof obj.error === 'object') {
      const err = obj.error as Record<string, unknown>;
      if (typeof err.message === 'string') return err.message;
    }

    // 通用字段
    if (typeof obj.message === 'string') return obj.message;
    if (typeof obj.detail === 'string') return obj.detail;

    return response.statusText;
  } catch {
    return response.statusText;
  }
}

/**
 * 处理 fetch 阶段的错误（网络错误 / 超时）
 */
function handleFetchError(err: unknown): Error {
  if (err instanceof Error) {
    // 请求超时（AbortController 触发）
    if (err.name === 'AbortError') {
      return new Error('AI 请求超时（30 秒），请检查网络或稍后重试');
    }
    // 网络连接错误（CORS / DNS / 离线等）
    if (err.name === 'TypeError' && err.message.includes('Failed to fetch')) {
      return new Error('无法连接到 AI 服务，请检查 apiEndpoint 是否正确以及网络是否畅通（注意 CORS 限制）');
    }
    return new Error(`AI 请求失败：${err.message}`);
  }
  return new Error('AI 请求发生未知错误');
}

export default {
  loadAISettings,
  isAIConfigured,
  callAIAnalysis,
  parseAIResponse,
  buildAnalysisPrompt,
};
