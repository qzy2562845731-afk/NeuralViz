# NeuralViz Backend

神经网络可视化后端服务

## 环境要求

- Python 3.10+
- Windows / Linux / macOS

## 安装依赖

```bash
cd backend
pip install -r requirements.txt
```

## 启动服务

### 方式一：直接运行

```bash
python main.py
```

### 方式二：使用 uvicorn

```bash
uvicorn main:app --reload --port 8000 --host 0.0.0.0
```

### 方式三：使用 FastAPI CLI

```bash
fastapi dev main.py --port 8000
```

## 访问地址

| 服务 | 地址 |
|------|------|
| API 服务 | http://localhost:8000 |
| Swagger 文档 | http://localhost:8000/docs |
| ReDoc 文档 | http://localhost:8000/redoc |

## API 接口

### 健康检查

```bash
# 基础健康检查
GET /api/health

# 服务状态与支持格式
GET /api/health/status
```

### 模型管理

```bash
# 解析模型
POST /api/model/parse
Content-Type: multipart/form-data

file: <模型文件>

# 获取模型信息
GET /api/model/{model_id}

# 获取模型列表
GET /api/model/
```

### 图片推理

```bash
# 单张图片推理
POST /api/inference/image
Content-Type: multipart/form-data

model_id: <模型ID>
file: <图片文件>
```

## 支持的模型格式

| 格式 | 扩展名 | 说明 |
|------|--------|------|
| ONNX | .onnx | 通用模型格式 |
| PyTorch | .pt, .pth | PyTorch 模型 |
| Keras | .h5, .keras | Keras 模型 |
| TensorFlow | .pb | Frozen Graph |
| Pickle | .pickle, .pkl | Python 序列化 |

## 目录结构

```
backend/
├── app/
│   ├── api/          # API 路由
│   ├── core/         # 核心配置
│   ├── ml/           # ML 适配器
│   ├── models/       # 数据模型
│   ├── schemas/      # Pydantic 模型
│   └── services/     # 业务逻辑
├── uploads/          # 上传文件目录
│   ├── models/       # 模型文件
│   └── images/       # 图片文件
├── main.py          # 入口文件
└── requirements.txt  # 依赖清单
```

## 前端对接

前端运行在 `http://localhost:5174`，后端运行在 `http://localhost:8000`。

后端已配置 CORS 允许 `http://localhost:5174` 跨域访问。

### 前端调用示例

```javascript
// 检测连接
const response = await fetch('http://localhost:8000/api/health/status');
const data = await response.json();
console.log(data); // {code: 200, message: "服务正常", data: {...}}

// 解析模型
const formData = new FormData();
formData.append('file', modelFile);
const parseResponse = await fetch('http://localhost:8000/api/model/parse', {
  method: 'POST',
  body: formData
});
const parseData = await parseResponse.json();
console.log(parseData.data); // {model_id, name, layers, ...}

// 图片推理
const inferenceForm = new FormData();
inferenceForm.append('model_id', modelId);
inferenceForm.append('file', imageFile);
const inferenceResponse = await fetch('http://localhost:8000/api/inference/image', {
  method: 'POST',
  body: inferenceForm
});
const inferenceData = await inferenceResponse.json();
console.log(inferenceData.data); // {predictions, inference_time, activations}
```
