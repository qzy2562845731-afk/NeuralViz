/// <reference types="vite/client" />
// 使用Vite代理避免CORS问题，开发环境通过 /api 代理到后端
const API_BASE_URL = import.meta.env.DEV ? '/api' : 'http://localhost:8000/api';

export interface ApiResponse<T = any> {
  code: number;
  message: string;
  data: T;
}

export interface ServiceStatusData {
  status: string;
  version: string;
  supported_formats: string[];
}

export interface ModelParseData {
  model_id: string;
  model_name: string;
  format: string;
  total_params: number;
  layer_count: number;
  input_shape: number[];
  output_shape: number[];
  layers: any[];
  [key: string]: any;
}

export interface InferenceResultData {
  activations: Record<string, number[]>;
  predictions: Array<{ class_id: number; probability: number }>;
  input_size: [number, number];
  success: boolean;
  inference_time?: number;
  [key: string]: any;
}

export interface ExperimentData {
  experiment_id: string;
  name: string;
  description: string;
  model_id: string | null;
  model_name: string | null;
  model_architecture: any;
  status: string;
  total_params: number;
  layer_count: number;
  best_accuracy: number;
  final_loss: number;
  total_epochs: number;
  current_step: number;
  hyperparams: Record<string, any>;
  config: Record<string, any>;
  tags: string[];
  created_at: string;
  updated_at: string;
  metrics?: {
    best_accuracy: number;
    final_loss: number;
    total_epochs: number;
    current_step: number;
    latest_step: number;
  };
  [key: string]: any;
}

export interface ExperimentListData {
  items: ExperimentData[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface ExperimentMetricData {
  step: number;
  epoch: number;
  loss: number;
  accuracy: number;
  val_loss: number;
  val_accuracy: number;
  learning_rate: number;
  batch_size: number;
  metric_type: string;
  extra_data: any;
  created_at: string;
}

export interface ExperimentMetricsData {
  experiment_id: string;
  metrics: ExperimentMetricData[];
  count: number;
}

/* 实验详情接口返回的全量数据结构（detail=true） */
export interface ExperimentDetailData extends ExperimentData {
  basic_info: {
    experiment_id: string;
    name: string;
    description: string;
    status: string;
    tags: string[];
    created_at: string;
    updated_at: string;
    remark: string;
  };
  model_config: {
    model_type: string;
    model_name: string;
    total_params: number;
    total_layers: number;
    input_shape: number[];
    output_shape: number[];
  };
  layers: Array<{
    name: string;
    type: string;
    params: number;
    input_shape: number[];
    output_shape: number[];
    node_count: number;
    activation: string;
    kernel_size?: number;
  }>;
  hyperparams: {
    learning_rate: number;
    batch_size: number;
    optimizer: string;
    total_epochs: number;
    random_seed: number;
    loss_function: string;
    dataset_name: string;
    dataset_version: string;
  };
  metrics_summary: {
    best_accuracy: number;
    final_loss: number;
    best_epoch: number | null;
    training_duration: number | null;
  };
  training_history: Array<{
    epoch: number;
    step: number;
    train_loss: number;
    val_loss: number;
    train_acc: number;
    val_acc: number;
    precision: number;
    recall: number;
    f1: number;
    learning_rate: number;
    gradient_norm: number;
    weight_norm: number;
    per_class_precision?: number[];
    per_class_recall?: number[];
    per_class_f1?: number[];
    confusion_matrix: number[][];
    prediction_distribution?: number[];
  }>;
  training_logs: any[];
}

/* 数据集数据结构 */
export interface DatasetData {
  dataset_id: string;
  name: string;
  description: string;
  version: string;
  file_path: string;
  extract_path: string | null;
  sample_count: number;
  class_count: number;
  image_size: string | null;
  feature_shape: string | null;
  dataset_type: string | null;
  class_distribution: Record<string, number>;
  file_hash: string | null;
  status: string;
  error_message: string | null;
  tags: string[];
  created_at: string;
  updated_at: string;
}

export interface DatasetListData {
  items: DatasetData[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

/* 训练状态数据结构 */
export interface TrainingStatusData {
  experiment_id: string;
  status: string; // pending / running / completed / failed / stopped / idle
  current_epoch: number;
  total_epochs: number;
  latest_metrics?: {
    epoch: number;
    loss: number;
    accuracy: number;
    val_loss: number;
    val_accuracy: number;
    learning_rate: number;
    batch_size: number;
    extra_data?: string;
  };
  error?: string | null;
  elapsed_seconds?: number;
  message?: string;
}

/* 训练日志数据结构 */
export interface TrainingLogsData {
  experiment_id: string;
  logs: string[];
  total: number;
}

/* 训练指标数据结构 */
export interface TrainingMetricItem {
  step: number;
  epoch: number;
  loss: number;
  accuracy: number;
  val_loss: number;
  val_accuracy: number;
  learning_rate: number;
  batch_size: number;
  metric_type: string;
  extra_data: string | null;
  created_at: string | null;
}

export interface TrainingMetricsData {
  experiment_id: string;
  metrics: TrainingMetricItem[];
  count: number;
}

/* 模型配置结构（CNN架构参数） */
export interface ModelConfig {
  channels: number[];
  attention: string;
  use_bn: boolean;
  use_dropout: boolean;
  dropout_rate: number;
  use_residual: boolean;
  fc_hidden: number[];
  use_attention: boolean;
  [key: string]: any;
}

/* CNN可视化数据结构 */
export interface VisualizationData {
  experiment_id: string;
  conv_kernels?: Record<string, number[][]>;
  feature_maps?: Record<string, number[][][]>;
  activation_maps?: Record<string, number[][]>;
  grad_cam?: Record<string, number[][]>;
  filter_visualizations?: Record<string, string>;
  [key: string]: any;
}

/* 消融实验单个配置项 */
export interface AblationConfig {
  name: string;
  model_config: Partial<ModelConfig>;
  [key: string]: any;
}

/* 消融实验运行参数 */
export interface AblationRunParams {
  dataset_id: string;
  name_prefix: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  val_split: number;
  configs: AblationConfig[];
  [key: string]: any;
}

/* 消融实验单个结果 */
export interface AblationResultItem {
  config_name: string;
  experiment_id: string;
  status: string;
  best_accuracy?: number;
  final_loss?: number;
  total_epochs?: number;
  training_duration?: number;
  model_config: Partial<ModelConfig>;
  metrics_summary?: Record<string, any>;
  [key: string]: any;
}

/* 消融实验组结果 */
export interface AblationResultData {
  group_name: string;
  dataset_id: string;
  total_configs: number;
  completed_count: number;
  results: AblationResultItem[];
  created_at: string;
  [key: string]: any;
}

async function request<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<ApiResponse<T>> {
  const url = `${API_BASE_URL}${endpoint}`;
  
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 8000);
  
  try {
    const response = await fetch(url, {
      ...options,
      signal: controller.signal,
      headers: {
        ...options.headers,
      },
    });

    clearTimeout(timeoutId);

    let data: any;
    try {
      data = await response.json();
    } catch {
      data = { code: response.status, message: response.statusText, data: null };
    }

    if (!response.ok) {
      const errorMessage = data?.detail || data?.message || `请求失败: ${response.status}`;
      throw new Error(errorMessage);
    }

    if (data && typeof data.code !== 'undefined') {
      return data as ApiResponse<T>;
    }

    return {
      code: response.status,
      message: 'success',
      data: data,
    } as ApiResponse<T>;
  } catch (err: any) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      throw new Error('请求超时，请检查服务是否正常运行');
    }
    if (err instanceof TypeError && err.message.includes('Failed to fetch')) {
      throw new Error('无法连接到服务器，请检查本地服务是否启动');
    }
    throw err;
  }
}

export const apiService = {
  async getServiceStatus(): Promise<ApiResponse<ServiceStatusData>> {
    return request<ServiceStatusData>('/health/status', {
      method: 'GET',
    });
  },

  async parseModel(file: File): Promise<ApiResponse<ModelParseData>> {
    const formData = new FormData();
    formData.append('file', file);

    return request<ModelParseData>('/model/parse', {
      method: 'POST',
      body: formData,
    });
  },

  async inferenceImage(
    modelId: string,
    file: File
  ): Promise<ApiResponse<InferenceResultData>> {
    const formData = new FormData();
    formData.append('model_id', modelId);
    formData.append('file', file);

    return request<InferenceResultData>('/inference/image', {
      method: 'POST',
      body: formData,
    });
  },

  getModelUrl(modelId: string): string {
    return `${API_BASE_URL}/model/${modelId}`;
  },

  // ========== 实验管理 ==========

  async listExperiments(params?: {
    page?: number;
    page_size?: number;
    status?: string;
    search?: string;
  }): Promise<ApiResponse<ExperimentListData>> {
    const query = new URLSearchParams();
    if (params?.page) query.set('page', String(params.page));
    if (params?.page_size) query.set('page_size', String(params.page_size));
    if (params?.status) query.set('status', params.status);
    if (params?.search) query.set('search', params.search);
    const qs = query.toString();
    return request<ExperimentListData>(`/experiment${qs ? `?${qs}` : ''}`, {
      method: 'GET',
    });
  },

  async getExperiment(experimentId: string): Promise<ApiResponse<ExperimentData>> {
    return request<ExperimentData>(`/experiment/${experimentId}`, {
      method: 'GET',
    });
  },

  async getExperimentDetail(experimentId: string): Promise<ApiResponse<ExperimentDetailData>> {
    return request<ExperimentDetailData>(`/experiment/${experimentId}?detail=true`, {
      method: 'GET',
    });
  },

  /**
   * 批量导出实验 CSV（对接后端全量字段接口）
   * - experiment_ids 为空时导出全部未删除实验
   * - 返回 Blob，可直接触发浏览器下载
   */
  async exportExperimentsCSV(experimentIds?: string[]): Promise<Blob> {
    const body = experimentIds && experimentIds.length > 0
      ? { experiment_ids: experimentIds }
      : {};
    const url = `${API_BASE_URL}/experiment/batch/export`;
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      throw new Error(`导出 CSV 失败: ${response.status}`);
    }
    return await response.blob();
  },

  async getExperimentsSummary(ids: string[]): Promise<ApiResponse<ExperimentData[]>> {
    const query = new URLSearchParams();
    query.set('ids', ids.join(','));
    return request<ExperimentData[]>(`/experiment/summary?${query.toString()}`, {
      method: 'GET',
    });
  },

  async createExperiment(data: {
    name: string;
    description?: string;
    model_id?: string;
    model_name?: string;
    model_architecture?: any;
    hyperparams?: Record<string, any>;
    config?: {
      model_config?: Partial<ModelConfig>;
      [key: string]: any;
    };
    tags?: string[];
    total_params?: number;
    layer_count?: number;
    status?: string;
    best_accuracy?: number;
    final_loss?: number;
    total_epochs?: number;
    current_step?: number;
  }): Promise<ApiResponse<ExperimentData>> {
    return request<ExperimentData>('/experiment', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
  },

  async updateExperiment(
    experimentId: string,
    data: Partial<ExperimentData>
  ): Promise<ApiResponse<ExperimentData>> {
    return request<ExperimentData>(`/experiment/${experimentId}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
  },

  async deleteExperiment(experimentId: string): Promise<ApiResponse<null>> {
    return request<null>(`/experiment/${experimentId}`, {
      method: 'DELETE',
    });
  },

  async renameExperiment(
    experimentId: string,
    name: string,
  ): Promise<ApiResponse<ExperimentData>> {
    return request<ExperimentData>(`/experiment/${experimentId}/rename`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
  },

  async batchDeleteExperiments(params: {
    experimentIds?: string[];
    deleteAll?: boolean;
  }): Promise<ApiResponse<{ deleted: number }>> {
    return request<{ deleted: number }>('/experiment/batch/delete', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        experiment_ids: params.experimentIds,
        delete_all: params.deleteAll,
      }),
    });
  },

  async getExperimentMetrics(
    experimentId: string,
    params?: { metric_type?: string; limit?: number }
  ): Promise<ApiResponse<ExperimentMetricsData>> {
    const query = new URLSearchParams();
    if (params?.metric_type) query.set('metric_type', params.metric_type);
    if (params?.limit) query.set('limit', String(params.limit));
    const qs = query.toString();
    return request<ExperimentMetricsData>(
      `/experiment/${experimentId}/metrics${qs ? `?${qs}` : ''}`,
      { method: 'GET' }
    );
  },

  async exportExperimentMetricsCSV(experimentId: string): Promise<Blob> {
    const url = `${API_BASE_URL}/experiment/${experimentId}/export/metrics-csv`;
    const response = await fetch(url, { method: 'GET' });
    if (!response.ok) throw new Error(`导出指标CSV失败: ${response.status}`);
    return await response.blob();
  },

  async exportExperimentJSON(experimentId: string): Promise<Blob> {
    const url = `${API_BASE_URL}/experiment/${experimentId}/export/json`;
    const response = await fetch(url, { method: 'GET' });
    if (!response.ok) throw new Error(`导出JSON失败: ${response.status}`);
    return await response.blob();
  },

  async addExperimentMetrics(
    experimentId: string,
    metrics: any[]
  ): Promise<ApiResponse<{ added: number }>> {
    return request<{ added: number }>(`/experiment/${experimentId}/metrics`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ metrics }),
    });
  },

  // ========== 数据集管理 ==========

  async listDatasets(params?: {
    page?: number;
    page_size?: number;
    search?: string;
    tags?: string;
    status?: string;
  }): Promise<ApiResponse<DatasetListData>> {
    const query = new URLSearchParams();
    if (params?.page) query.set('page', String(params.page));
    if (params?.page_size) query.set('page_size', String(params.page_size));
    if (params?.search) query.set('search', params.search);
    if (params?.tags) query.set('tags', params.tags);
    if (params?.status) query.set('status', params.status);
    const qs = query.toString();
    return request<DatasetListData>(`/dataset${qs ? `?${qs}` : ''}`, {
      method: 'GET',
    });
  },

  async getDataset(datasetId: string): Promise<ApiResponse<DatasetData>> {
    return request<DatasetData>(`/dataset/${datasetId}`, {
      method: 'GET',
    });
  },

  async getDatasetVersions(name: string): Promise<ApiResponse<{ name: string; versions: DatasetData[]; count: number }>> {
    return request<{ name: string; versions: DatasetData[]; count: number }>(
      `/dataset/${encodeURIComponent(name)}/versions`,
      { method: 'GET' }
    );
  },

  async uploadDataset(
    file: File,
    name: string,
    description?: string,
    tags?: string
  ): Promise<ApiResponse<DatasetData>> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name);
    if (description) formData.append('description', description);
    if (tags) formData.append('tags', tags);
    return request<DatasetData>('/dataset/upload', {
      method: 'POST',
      body: formData,
    });
  },

  /** 直接上传非压缩格式数据集（CSV / JSON / NumPy） */
  async uploadDirectDataset(
    file: File,
    name?: string,
    description?: string,
    tags?: string
  ): Promise<ApiResponse<DatasetData>> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('name', name || file.name);
    if (description) formData.append('description', description);
    if (tags) formData.append('tags', tags);
    return request<DatasetData>('/dataset/upload/direct', {
      method: 'POST',
      body: formData,
    });
  },

  async deleteDataset(datasetId: string): Promise<ApiResponse<null>> {
    return request<null>(`/dataset/${datasetId}`, {
      method: 'DELETE',
    });
  },

  async reparseDataset(datasetId: string): Promise<ApiResponse<DatasetData>> {
    return request<DatasetData>(`/dataset/${datasetId}/reparse`, {
      method: 'POST',
    });
  },

  /** 下载内置标准数据集 (MNIST/CIFAR-10) */
  async downloadBuiltinDataset(
    datasetName: 'mnist' | 'cifar10'
  ): Promise<ApiResponse<{ task_id: string; dataset_name: string; status: string; progress: number; message: string; dataset_id?: string }>> {
    return request(`/dataset/builtin/${datasetName}/download`, {
      method: 'POST',
    });
  },

  /** 查询内置数据集下载状态 */
  async getBuiltinDownloadStatus(datasetName: 'mnist' | 'cifar10'): Promise<ApiResponse<any>> {
    return request(`/dataset/builtin/${datasetName}/status`, {
      method: 'GET',
    });
  },

  /** 列出内置标准数据集 */
  async listBuiltinDatasets(): Promise<ApiResponse<any[]>> {
    return request('/dataset/builtin', { method: 'GET' });
  },

  /** 获取数据集预览（样本图、类别分布、数据质量） */
  async getDatasetPreview(
    datasetId: string,
    samples: number = 16
  ): Promise<ApiResponse<{
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
    samples: Array<{ image: string; label: string | number; index: number }>;
  }>> {
    return request(`/dataset/${datasetId}/preview?samples=${samples}`, {
      method: 'GET',
    });
  },

  // ========== 数据导出（Excel/CSV/PDF/SVG/PNG） ==========

  /** 导出所有实验列表为Excel（多sheet） */
  async exportAllExperimentsExcel(): Promise<Blob> {
    const url = `${API_BASE_URL}/export/experiments/excel`;
    const response = await fetch(url, { method: 'GET' });
    if (!response.ok) throw new Error(`导出Excel失败: ${response.status}`);
    return await response.blob();
  },

  /** 导出单个实验为Excel */
  async exportExperimentExcel(experimentId: string): Promise<Blob> {
    const url = `${API_BASE_URL}/export/experiment/${experimentId}/excel`;
    const response = await fetch(url, { method: 'GET' });
    if (!response.ok) throw new Error(`导出实验Excel失败: ${response.status}`);
    return await response.blob();
  },

  /** 导出单个实验指标CSV */
  async exportMetricsCSV(experimentId: string): Promise<Blob> {
    const url = `${API_BASE_URL}/export/experiment/${experimentId}/metrics/csv`;
    const response = await fetch(url, { method: 'GET' });
    if (!response.ok) throw new Error(`导出指标CSV失败: ${response.status}`);
    return await response.blob();
  },

  /** 导出混淆矩阵（CSV/JSON/PNG） */
  async exportConfusionMatrix(
    experimentId: string,
    fmt: 'csv' | 'json' | 'png' = 'csv',
    dpi: number = 300,
    normalize: boolean = false
  ): Promise<Blob> {
    const url = `${API_BASE_URL}/export/experiment/${experimentId}/confusion_matrix/${fmt}?dpi=${dpi}&normalize=${normalize}`;
    const response = await fetch(url, { method: 'GET' });
    if (!response.ok) throw new Error(`导出混淆矩阵失败: ${response.status}`);
    return await response.blob();
  },

  /** 导出训练图表（loss_curve/acc_curve/roc/pr/confusion_matrix） */
  async exportChart(
    experimentId: string,
    chartType: 'loss_curve' | 'acc_curve' | 'roc' | 'pr' | 'confusion_matrix',
    fmt: 'png' | 'svg' | 'pdf' = 'png',
    dpi: number = 300,
    normalize: boolean = false
  ): Promise<Blob> {
    const url = `${API_BASE_URL}/export/experiment/${experimentId}/chart/${chartType}/${fmt}?dpi=${dpi}&normalize=${normalize}`;
    const response = await fetch(url, { method: 'GET' });
    if (!response.ok) throw new Error(`导出图表失败: ${response.status}`);
    return await response.blob();
  },

  // ========== 模型推理扩展（Grad-CAM） ==========

  /** Grad-CAM 显著性可视化 */
  async gradcamVisualization(
    modelId: string,
    file: File,
    options?: { target_class?: number; use_plusplus?: boolean; alpha?: number }
  ): Promise<ApiResponse<{
    original_image: string;
    heatmap: string;
    overlay: string;
    predicted_class: number;
    confidence: number;
    target_class: number;
  }>> {
    const formData = new FormData();
    formData.append('model_id', modelId);
    formData.append('file', file);
    if (options?.target_class !== undefined) formData.append('target_class', String(options.target_class));
    if (options?.use_plusplus !== undefined) formData.append('use_plusplus', String(options.use_plusplus));
    if (options?.alpha !== undefined) formData.append('alpha', String(options.alpha));

    return request('/inference/gradcam', {
      method: 'POST',
      body: formData,
    });
  },

  /** 触发文件下载的工具函数 */
  downloadBlob(blob: Blob, filename: string): void {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  },

  // ============================================================
  // 训练引擎模块
  // ============================================================

  /** 启动训练任务 */
  async startTraining(
    experimentId: string,
    params?: {
      dataset_id?: string;
      hyperparams?: Record<string, any>;
      model_config?: Record<string, any>;
    }
  ): Promise<ApiResponse<TrainingStatusData>> {
    return request<TrainingStatusData>(`/training/start/${experimentId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params || {}),
    });
  },

  /** 停止训练任务（优雅退出） */
  async stopTraining(experimentId: string): Promise<ApiResponse<TrainingStatusData>> {
    return request<TrainingStatusData>(`/training/stop/${experimentId}`, {
      method: 'POST',
    });
  },

  /** 获取训练实时状态 */
  async getTrainingStatus(experimentId: string): Promise<ApiResponse<TrainingStatusData>> {
    return request<TrainingStatusData>(`/training/status/${experimentId}`, {
      method: 'GET',
    });
  },

  /** 增量获取训练日志 */
  async getTrainingLogs(
    experimentId: string,
    since: number = 0
  ): Promise<ApiResponse<TrainingLogsData>> {
    return request<TrainingLogsData>(`/training/logs/${experimentId}?since=${since}`, {
      method: 'GET',
    });
  },

  /** 获取全量训练指标时序数据 */
  async getTrainingMetrics(experimentId: string): Promise<ApiResponse<TrainingMetricsData>> {
    return request<TrainingMetricsData>(`/training/metrics/${experimentId}`, {
      method: 'GET',
    });
  },

  /** 获取CNN可视化数据（卷积核、特征图、Grad-CAM等） */
  async getVisualizations(experimentId: string): Promise<ApiResponse<VisualizationData>> {
    return request<VisualizationData>(`/training/visualizations/${experimentId}`, {
      method: 'GET',
    });
  },

  /** 运行消融实验（批量对比不同模型配置） */
  async runAblationExperiment(params: AblationRunParams): Promise<ApiResponse<{ group_name: string; experiment_ids: string[]; message: string }>> {
    return request<{ group_name: string; experiment_ids: string[]; message: string }>('/training/ablation/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
  },

  /** 获取消融实验结果 */
  async getAblationResults(groupName: string): Promise<ApiResponse<AblationResultData>> {
    return request<AblationResultData>(`/training/ablation/results/${encodeURIComponent(groupName)}`, {
      method: 'GET',
    });
  },

  // ============================================================
  // 自定义实验模板
  // ============================================================

  /** 获取自定义实验模板列表 */
  async listTemplates(templateType?: 'comparison' | 'ablation'): Promise<ApiResponse<any[]>> {
    const query = templateType ? `?template_type=${templateType}` : '';
    return request(`/experiment/templates${query}`, { method: 'GET' });
  },

  /** 保存自定义实验模板 */
  async saveTemplate(data: {
    name: string;
    description?: string;
    template_type: 'comparison' | 'ablation';
    configs: any[];
    comparison_metrics?: string[];
  }): Promise<ApiResponse<any>> {
    return request('/experiment/templates', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
  },

  /** 删除自定义实验模板 */
  async deleteTemplate(templateId: string): Promise<ApiResponse<null>> {
    return request(`/experiment/templates/${templateId}`, { method: 'DELETE' });
  },

  /** 运行自定义对比/消融实验 */
  async runCustomExperiment(params: {
    dataset_id: string;
    experiment_type: 'comparison' | 'ablation';
    name_prefix: string;
    epochs: number;
    batch_size: number;
    learning_rate: number;
    val_split: number;
    configs: any[];
    comparison_metrics?: string[];
  }): Promise<ApiResponse<{ group_name: string; experiment_ids: string[]; message: string }>> {
    return request('/training/ablation/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
  },

  // ============================================================
  // 模型导入
  // ============================================================

  /** 导入外部模型文件 */
  async importModel(params: {
    model_name: string;
    model_format?: string;
    dataset_type?: string;
    feature_shape?: string;
    num_classes?: number;
  }): Promise<ApiResponse<{
    model_name: string;
    model_path: string;
    model_format: string;
    total_params: number;
    layer_count: number;
    layers: any[];
    model_class: string;
    weight_loaded: boolean;
    available_architectures: string[];
  }>> {
    const query = new URLSearchParams();
    query.set('model_name', params.model_name);
    query.set('model_format', params.model_format || 'pytorch');
    query.set('dataset_type', params.dataset_type || 'image_folder');
    query.set('feature_shape', params.feature_shape || '1x28x28');
    query.set('num_classes', String(params.num_classes || 10));
    return request(`/training/import-model?${query.toString()}`, {
      method: 'POST',
    });
  },

  /** 获取元数据：注意力机制类型 */
  async getAttentionTypes(): Promise<ApiResponse<{ attention_types: any[] }>> {
    return request('/training/meta/attention-types', { method: 'GET' });
  },

  /** 获取元数据：损失函数列表 */
  async getLossFunctions(): Promise<ApiResponse<{ loss_functions: any[] }>> {
    return request('/training/meta/loss-functions', { method: 'GET' });
  },

  /** 获取元数据：模型架构列表 */
  async getModelArchitectures(): Promise<ApiResponse<{ architectures: string[] }>> {
    return request('/training/meta/model-architectures', { method: 'GET' });
  },

  /** 获取元数据：激活函数列表 */
  async getActivations(): Promise<ApiResponse<{ activations: any[] }>> {
    return request('/training/meta/activations', { method: 'GET' });
  },
};

export { apiService as trainingApi };
export default apiService;
