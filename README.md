# NeuralViz — 神经网络训练可视化平台

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License" />
  <img src="https://img.shields.io/badge/react-18.2-blue" alt="React" />
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python" />
  <img src="https://img.shields.io/badge/pytorch-2.12-red" alt="PyTorch" />
  <img src="https://img.shields.io/badge/fastapi-0.115-green" alt="FastAPI" />
  <img src="https://img.shields.io/badge/tests-288_passed-brightgreen" alt="Tests" />
  <img src="https://img.shields.io/badge/build-passing-brightgreen" alt="Build" />
</p>

<p align="center">
  <b>全栈神经网络训练可视化平台</b> — 上传模型、交互式 3D 可视化、在线调参训练、实时监控、Grad-CAM 可解释性分析、多组对比与消融实验
</p>

---

## 目录

- [简介](#简介)
- [功能特性](#功能特性)
- [技术栈](#技术栈)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
  - [环境要求](#环境要求)
  - [后端启动](#后端启动)
  - [前端启动](#前端启动)
  - [验证安装](#验证安装)
- [使用指南](#使用指南)
  - [场景一：快速体验（5 分钟）](#场景一快速体验5-分钟)
  - [场景二：上传自定义模型](#场景二上传自定义模型)
  - [场景三：在线训练与调参](#场景三在线训练与调参)
  - [场景四：实验对比与分析](#场景四实验对比与分析)
  - [场景五：自动化训练](#场景五自动化训练)
- [开发指南](#开发指南)
- [测试](#测试)
- [贡献指南](#贡献指南)
- [开源协议](#开源协议)

---

## 简介

NeuralViz 是一个面向深度学习研究与教学的全栈可视化平台。它通过 **Three.js 3D 渲染** 将 CNN 网络结构以交互式图形呈现，通过 **FastAPI + PyTorch** 提供在线训练引擎，结合 **ECharts** 实时展示训练曲线，让神经网络的内部运作变得一目了然。

**典型应用场景：**
- 深度学习课程教学演示
- 模型调试与超参数调优
- 论文实验对比与消融分析
- 模型可解释性研究（Grad-CAM）

---

## 功能特性

### 模型管理
- 上传 PyTorch (`.pt`/`.pth`) 和 ONNX (`.onnx`) 模型，自动解析
- 内置 MNIST 示例 CNN 模型，开箱即用
- 提取网络层级结构、参数量、FLOPs 等元信息

### 3D 可视化
- Three.js 驱动的 CNN 3D 交互式网络结构展示
- 逐层展开、旋转/缩放、特征图可视化
- Layer Inspector：查看每层输入输出维度、参数量
- 激活值直方图、梯度权重分布图

### 在线训练引擎
- 可视化配置：注意力机制（SE / CBAM / Self-Attention / ECA / MHSA / GCT）、损失函数（CrossEntropy / Focal / Dice / LabelSmoothing / Triplet / Contrastive / Combined）、优化器、学习率、Batch Size、Epochs
- 支持数据增强：随机翻转、旋转、CutMix、MixUp
- 学习率调度器：CosineAnnealing / ReduceLROnPlateau / Step / Exponential
- 训练过程支持暂停 / 恢复 / 终止
- 混合精度训练（AMP）支持
- 断点续训（Checkpoint）

### 推理与分析
- 单张 / 批量图片推理，展示预测结果与置信度
- Grad-CAM 注意力热力图，可视化模型关注区域
- 混淆矩阵、ROC/PR 曲线、精确率/召回率/F1
- 特征图逐层查看器

### 实验管理
- 多组对比实验，可视化不同超参数组合差异
- 消融实验（Ablation Study），评估各模块贡献度
- 实验导出（PDF / Excel），完整实验报告生成
- 实验历史记录与回放

### AI 诊断面板
- 自动分析训练状态：过拟合/欠拟合检测
- 梯度消失/爆炸预警
- 基于训练曲线智能推荐超参数调整方向

### 数据集管理
- 支持 CSV / JSON / NPY / NPZ / 图片文件夹 等多种格式
- 数据集预览、统计分析、格式自动检测
- 内置 MNIST 示例数据集

### 自动化训练
- 预设超参数搜索空间，自动网格搜索 / 随机搜索
- 自动对比最佳配置
- 批量实验调度

---

## 技术栈

| 层级 | 技术 | 说明 |
|------|------|------|
| 前端框架 | React 18 + TypeScript | 严格模式，类型安全 |
| 构建工具 | Vite 5 | 极速 HMR，按需编译 |
| UI 框架 | TailwindCSS 4 + shadcn/ui | 暗色科技风格 |
| 动画 | Framer Motion 11 | 页面过渡与微交互 |
| 3D 渲染 | Three.js 0.184 | CNN 网络结构 3D 展示 |
| 数据可视化 | ECharts 5 | 训练曲线、混淆矩阵、ROC |
| 路由 | React Router v7 | 懒加载代码分割 |
| 后端框架 | FastAPI 0.115 | 自动生成 API 文档 |
| 数据库 | SQLAlchemy 2.0 + SQLite | ORM + 自动迁移 |
| 机器学习 | PyTorch 2.12 + ONNX Runtime 1.27 | 训练与推理引擎 |
| 数据验证 | Pydantic v2 | 请求/响应模型校验 |
| 测试 | pytest 9.1 | 288 个测试用例 |

---

## 项目结构

```
NeuralViz/
├── backend/                        # 后端服务
│   ├── main.py                     # FastAPI 入口
│   ├── requirements.txt            # Python 生产依赖
│   ├── requirements-dev.txt        # Python 开发依赖
│   ├── .env.example                # 环境变量模板
│   ├── app/
│   │   ├── api/                    # API 路由层 (8 个模块)
│   │   │   ├── health.py           # 健康检查
│   │   │   ├── model.py            # 模型解析与管理
│   │   │   ├── training.py         # 训练引擎
│   │   │   ├── inference.py        # 推理服务
│   │   │   ├── experiment.py       # 实验管理
│   │   │   ├── dataset.py          # 数据集管理
│   │   │   ├── export.py           # 导出服务
│   │   │   └── gradcam.py          # Grad-CAM 可视化
│   │   ├── core/                   # 核心模块
│   │   │   ├── config.py           # 配置管理 (pydantic-settings)
│   │   │   ├── database.py         # 数据库初始化与迁移
│   │   │   ├── exception.py        # 全局异常处理
│   │   │   ├── security.py         # 安全工具 (路径净化)
│   │   │   └── logging_config.py   # 日志配置
│   │   ├── ml/                     # 机器学习模块
│   │   │   ├── model_builder.py    # 可配置 CNN/MLP 构建器
│   │   │   ├── pytorch_adapter.py  # PyTorch 模型适配器
│   │   │   ├── onnx_adapter.py     # ONNX 模型适配器
│   │   │   ├── training_utils.py   # 训练工具 (Checkpoint/EarlyStopping/Scheduler/AMP)
│   │   │   ├── attention.py        # 6 种注意力机制
│   │   │   ├── losses.py           # 7 种损失函数
│   │   │   ├── augmentation.py     # 数据增强 (CutMix/MixUp)
│   │   │   ├── gradcam.py          # Grad-CAM 实现
│   │   │   ├── metrics.py          # 评估指标 (ROC/PR/混淆矩阵)
│   │   │   └── datasets/           # 数据集解析器 (CSV/JSON/NPY/ImageFolder)
│   │   ├── models/                 # SQLAlchemy 数据模型
│   │   ├── schemas/                # Pydantic 数据模式
│   │   └── services/               # 业务逻辑层
│   └── tests/                      # 16 个测试文件, 288 个用例
├── src/                            # 前端源码
│   ├── App.tsx                     # 路由配置 (9 个路由)
│   ├── main.tsx                    # 渲染入口
│   ├── components/
│   │   ├── cnn3d/                  # CNN 3D 可视化组件
│   │   ├── home/                   # 首页 (Hero/Feature/DynamicShowcase)
│   │   ├── layout/                 # 布局 (AppLayout/SidebarNav)
│   │   ├── workbench/              # 工作台 (模型管理/训练/推理)
│   │   ├── dataset/                # 数据集组件
│   │   └── ui/                     # 基础 UI 组件 (shadcn)
│   ├── pages/                      # 8 个页面 (懒加载)
│   ├── hooks/                      # 自定义 Hooks
│   ├── services/                   # API 服务层 (60+ 方法)
│   ├── contexts/                   # React Context (Toast/GlobalTraining)
│   ├── utils/                      # 工具函数
│   └── types/                      # TypeScript 类型定义
├── public/                         # 静态资源
├── package.json                    # 前端依赖
├── vite.config.ts                  # Vite 配置 (含 API 代理)
├── tsconfig.json                   # TypeScript 严格模式
├── .gitignore                      # Git 忽略规则
└── LICENSE                         # MIT 许可证
```

---

## 快速开始

### 环境要求

| 工具 | 最低版本 | 验证命令 |
|------|----------|----------|
| Python | 3.10+ | `python --version` |
| Node.js | 18+ | `node --version` |
| npm | 9+ | `npm --version` |

### 1. 克隆项目

```bash
git clone https://github.com/qzy2562845731-afk/NeuralViz.git
cd NeuralViz
```

### 2. 后端启动

```bash
cd backend

# === 创建虚拟环境（推荐） ===
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# === 安装依赖 ===
pip install -r requirements.txt

# （可选）安装开发依赖（测试/代码质量工具）
pip install -r requirements-dev.txt

# === 配置环境变量（可选） ===
# 复制模板文件并根据需要修改
cp .env.example .env

# === 启动后端服务 ===
python main.py
```

启动成功后输出：
```
服务启动完成 - API文档: http://localhost:8000/docs
示例数据集就绪: SampleMNIST (示例) (2000 样本)
```

### 3. 前端启动

新开一个终端，在项目根目录执行：

```bash
# 安装依赖
npm install

# 启动开发服务器
npm run dev
```

前端默认运行在 `http://localhost:5173`，API 请求自动代理到后端 `http://localhost:8000`。

### 4. 验证安装

```bash
# 后端健康检查
curl http://localhost:8000/api/health
# 预期: {"code":200,"message":"服务运行正常","data":{"status":"online"}}

# 运行后端测试
cd backend
python -m pytest tests/ -v
# 预期: 288 passed
```

---

## 使用指南

### 场景一：快速体验（5 分钟）

1. 打开浏览器访问 `http://localhost:5173`
2. 在首页点击 **「开始使用」** 进入工作台
3. 系统自动加载内置 MNIST 示例 CNN 模型
4. 在 **3D 可视化面板** 中旋转、缩放查看网络结构
5. 点击 **「开始训练」**，使用默认超参数启动训练
6. 实时观察 Loss 和 Accuracy 曲线变化
7. 训练完成后，在推理面板上传手写数字图片查看预测结果
8. 切换至 **Grad-CAM** 面板，查看模型注意力热力图

### 场景二：上传自定义模型

1. 进入工作台 → **模型管理**
2. 点击 **「上传模型」**，选择 `.pt` / `.pth` / `.onnx` 文件
3. 系统自动解析网络结构，生成 3D 可视化
4. 在 Layer Inspector 中查看每层参数和维度
5. 使用推理面板测试模型预测效果

### 场景三：在线训练与调参

1. 在工作台进入 **训练配置**
2. 配置超参数：
   - **注意力机制**：选择 SE / CBAM / Self-Attention / ECA / MHSA / GCT 或 无
   - **损失函数**：CrossEntropy / Focal / Dice / LabelSmoothing / Triplet / Contrastive / Combined
   - **优化器**：SGD / Adam / AdamW，设置学习率
   - **Batch Size**：默认 32
   - **Epochs**：默认 10
   - **学习率调度器**：CosineAnnealing / ReduceLROnPlateau / Step / Exponential
   - **数据增强**：随机翻转/旋转 + CutMix/MixUp
3. 点击 **「开始训练」**
4. 实时监控训练进度、Loss/Accuracy 曲线、梯度范数
5. 训练完成后，系统自动保存实验记录

### 场景四：实验对比与分析

1. 训练多个模型（不同超参数组合）
2. 进入 **实验管理** 页面
3. 勾选多个实验进行对比
4. 查看 Loss/Accuracy 对比曲线、混淆矩阵、ROC/PR 曲线
5. 导出实验报告（PDF 或 Excel）
6. 使用 **AI 诊断面板** 分析训练问题，获取超参数调整建议

### 场景五：自动化训练

1. 进入 **自动训练** 页面
2. 设置超参数搜索空间（学习率范围、Batch Size 选项、注意力机制列表等）
3. 选择搜索策略（网格搜索 / 随机搜索）
4. 点击 **「开始自动训练」**
5. 系统自动执行多组实验，输出最佳配置对比

---

## 开发指南

### 前端开发

```bash
# TypeScript 类型检查
npx tsc --noEmit

# 生产构建
npm run build

# 预览构建产物
npm run preview
```

### 后端开发

```bash
# 启动开发服务器（热重载）
uvicorn main:app --reload --port 8000 --host 0.0.0.0

# 访问交互式 API 文档
open http://localhost:8000/docs

# 代码格式化
ruff check . --fix
black .
```

### 数据库

项目使用 SQLite（零配置），数据库文件自动创建在 `backend/neuralviz.db`。如需更换数据库，修改 `.env` 中的 `DATABASE_URL`。

---

## 测试

```bash
cd backend

# 运行所有测试
python -m pytest tests/ -v

# 运行指定模块测试
python -m pytest tests/test_losses.py -v
python -m pytest tests/test_training.py -v

# 生成覆盖率报告
python -m pytest tests/ --cov=app --cov-report=html
```

当前测试覆盖：**16 个测试文件，288 个测试用例，全部通过。**

---

## 贡献指南

我们欢迎任何形式的贡献！无论是 Bug 修复、功能建议、文档改进还是代码贡献。

### 贡献流程

1. **Fork** 本仓库
2. 创建特性分支：`git checkout -b feature/amazing-feature`
3. 提交更改：`git commit -m 'feat: 添加某个功能'`
4. 推送到分支：`git push origin feature/amazing-feature`
5. 创建 **Pull Request**

### 提交规范

请使用 [Conventional Commits](https://www.conventionalcommits.org/) 格式：
- `feat:` 新功能
- `fix:` Bug 修复
- `docs:` 文档更新
- `refactor:` 代码重构
- `test:` 测试相关
- `chore:` 构建/工具链

### 开发环境设置

```bash
# 安装前后端依赖
cd backend && pip install -r requirements-dev.txt && cd ..
npm install

# 启动开发环境
cd backend && python main.py &  # 后端
npm run dev                      # 前端
```

### 行为准则

- 尊重所有贡献者，保持友善和建设性的沟通
- Issue 和 PR 讨论聚焦技术问题
- 欢迎新手提问，提供友好的入门指导

---

## 开源协议

本项目基于 [MIT License](LICENSE) 开源。你可以自由使用、修改和分发，但需保留原始版权声明。

---

## 致谢

- [Three.js](https://threejs.org/) - 3D 渲染引擎
- [ECharts](https://echarts.apache.org/) - 数据可视化
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [FastAPI](https://fastapi.tiangolo.com/) - 高性能 API 框架
- [shadcn/ui](https://ui.shadcn.com/) - UI 组件库
- [TailwindCSS](https://tailwindcss.com/) - CSS 框架

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://github.com/qzy2562845731-afk">qzy2562845731-afk</a></sub>
</p>