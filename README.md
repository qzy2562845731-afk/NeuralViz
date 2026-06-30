# NeuralViz - 神经网络训练可视化系统

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License" />
  <img src="https://img.shields.io/badge/react-18.2-blue" alt="React" />
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python" />
  <img src="https://img.shields.io/badge/pytorch-2.6+-red" alt="PyTorch" />
  <img src="https://img.shields.io/badge/fastapi-0.115-green" alt="FastAPI" />
</p>

NeuralViz 是一个全栈神经网络训练可视化平台，支持上传 / 加载 CNN 模型并进行交互式 3D 可视化，提供在线调参训练、实时监控、Grad-CAM 可解释性分析、多组对比实验与消融实验等完整功能。

## 功能特性

### 模型管理
- 支持上传自定义 PyTorch 模型（`.pt` / `.pth`）和 ONNX 模型（`.onnx`）
- 内置 MNIST 示例 CNN 模型，可快速完整体验全流程
- 模型自动解析，提取网络层级结构、参数量、FLOPs 等元信息

### 3D 可视化
- 基于 Three.js 的 CNN 3D 交互式网络结构展示
- 支持逐层展开、旋转/缩放、特征图可视化
- 层级检查器（Layer Inspector）查看每层输入输出维度、参数量
- 激活值直方图、梯度权重分布图

### 在线训练
- 可视化配置注意力机制、损失函数、优化器、学习率等超参数
- 一键启动模型训练，实时监控 Loss / Accuracy 训练曲线
- 训练过程支持暂停 / 恢复 / 终止
- 自动训练模式：预设超参数组合，自动搜索最佳配置

### 推理与分析
- 单张 / 批量推理，展示预测结果与置信度分布
- Grad-CAM 注意力热力图生成，可视化模型关注区域
- 混淆矩阵、预测分布图等多维度评估指标
- 特征图查看器，逐层观察中间表示

### 实验管理
- 多组对比实验，可视化不同超参数组合的效果差异
- 消融实验（Ablation Study），评估各模块贡献度
- 实验导出（PDF / Excel），支持完整实验报告生成
- 实验历史记录与回放

### AI 诊断
- 内置 AI 训练诊断面板，自动分析训练状态
- 检测过拟合 / 欠拟合、梯度消失 / 爆炸等问题
- 基于训练曲线智能推荐超参数调整方向

### 数据集管理
- 支持上传 CSV / JSON / NPY / 图片文件夹等格式数据集
- 数据集预览、统计分析、格式转换
- 内置数据集下载器（MNIST 等）

## 技术栈

| 层级 | 技术 |
|------|------|
| 前端框架 | React 18 + TypeScript |
| 构建工具 | Vite 5 |
| UI 框架 | TailwindCSS 4 + shadcn/ui |
| 3D 渲染 | Three.js |
| 数据可视化 | ECharts |
| 路由 | React Router v7 |
| 后端框架 | FastAPI 0.115 |
| 数据库 | SQLAlchemy 2.0 |
| 机器学习 | PyTorch 2.6 + ONNX Runtime |
| 数据验证 | Pydantic v2 |

## 项目结构

```
NeuralViz/
├── backend/                    # 后端服务
│   ├── main.py                 # FastAPI 入口
│   ├── requirements.txt        # Python 依赖
│   ├── app/
│   │   ├── api/                # API 路由层
│   │   │   ├── dataset.py      # 数据集管理接口
│   │   │   ├── experiment.py   # 实验管理接口
│   │   │   ├── export.py       # 导出接口
│   │   │   ├── gradcam.py      # Grad-CAM 接口
│   │   │   ├── health.py       # 健康检查
│   │   │   ├── inference.py    # 推理接口
│   │   │   ├── model.py        # 模型管理接口
│   │   │   └── training.py     # 训练接口
│   │   ├── core/               # 核心模块
│   │   │   ├── config.py       # 配置管理
│   │   │   ├── database.py     # 数据库初始化
│   │   │   ├── exception.py    # 异常处理
│   │   │   ├── logging_config.py
│   │   │   └── security.py     # 安全工具
│   │   ├── ml/                 # 机器学习模块
│   │   │   ├── pytorch_adapter.py   # PyTorch 模型适配器
│   │   │   ├── onnx_adapter.py      # ONNX 模型适配器
│   │   │   ├── model_builder.py     # 模型构建器
│   │   │   ├── training_utils.py    # 训练工具
│   │   │   ├── gradcam.py           # Grad-CAM 实现
│   │   │   ├── attention.py         # 注意力机制
│   │   │   ├── augmentation.py      # 数据增强
│   │   │   ├── losses.py            # 损失函数
│   │   │   ├── metrics.py           # 评估指标
│   │   │   └── datasets/            # 数据集解析器
│   │   ├── models/             # 数据模型（SQLAlchemy）
│   │   ├── schemas/            # Pydantic 数据模式
│   │   └── services/           # 业务逻辑层
│   └── tests/                  # 后端测试
├── src/                        # 前端源码
│   ├── App.tsx                 # 应用入口
│   ├── main.tsx                # 渲染入口
│   ├── components/             # 组件
│   │   ├── cnn3d/              # CNN 3D 可视化组件
│   │   ├── home/               # 首页组件
│   │   ├── layout/             # 布局组件
│   │   ├── ui/                 # 基础 UI 组件
│   │   ├── workbench/          # 工作台组件
│   │   └── dataset/            # 数据集组件
│   ├── pages/                  # 页面
│   ├── hooks/                  # 自定义 Hooks
│   ├── services/               # API 服务层
│   ├── contexts/               # React Context
│   ├── utils/                  # 工具函数
│   └── types/                  # TypeScript 类型定义
├── public/                     # 静态资源
├── package.json                # 前端依赖
├── tsconfig.json               # TypeScript 配置
├── vite.config.ts              # Vite 配置
├── .gitignore                  # Git 忽略规则
└── LICENSE                     # MIT 许可证
```

## 快速开始

### 环境要求

- **Python**: 3.10+
- **Node.js**: 18+
- **npm**: 9+

### 1. 克隆项目

```bash
git clone git@github.com:qzy2562845731-afk/NeuralViz.git
cd NeuralViz
```

### 2. 后端启动

```bash
cd backend

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 启动后端服务
python main.py
# 或: uvicorn main:app --reload --port 8000 --host 0.0.0.0
```

后端服务默认运行在 `http://localhost:8000`，API 文档自动生成在 `http://localhost:8000/docs`。

### 3. 前端启动

```bash
# 在项目根目录
npm install
npm run dev
```

前端开发服务器默认运行在 `http://localhost:5173`。

### 4. 访问应用

打开浏览器访问 `http://localhost:5173`，即可开始使用 NeuralViz。

## 使用指南

### 快速体验

1. 进入首页，点击「内置 MNIST 示例」加载预置 CNN 模型
2. 在 3D 可视化面板中旋转、缩放查看网络结构
3. 点击「开始训练」，使用默认超参数启动训练
4. 实时观察 Loss 曲线和 Accuracy 曲线变化
5. 训练完成后，在推理面板上传测试图片查看预测结果
6. 切换至 Grad-CAM 面板，查看模型注意力热力图

### 自定义模型

1. 在工作台导入自定义 PyTorch（`.pt`）或 ONNX（`.onnx`）模型
2. 系统自动解析网络结构并生成 3D 可视化
3. 配置训练超参数后启动训练
4. 对比不同实验的训练效果

### 实验对比

1. 训练完成后保存实验
2. 创建多个实验（不同超参数组合）
3. 在实验对比页面查看多组实验的 Loss/Accuracy 对比曲线
4. 导出实验报告为 PDF 或 Excel

## 开发

### 后端测试

```bash
cd backend
python -m pytest tests/ -v
```

### 前端构建

```bash
npm run build
```

构建产物输出到 `dist/` 目录。

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m '添加某个功能'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 创建 Pull Request

## 开源协议

本项目基于 [MIT License](LICENSE) 开源。

## 联系方式

- GitHub Issues: [提交问题](https://github.com/qzy2562845731-afk/NeuralViz/issues)