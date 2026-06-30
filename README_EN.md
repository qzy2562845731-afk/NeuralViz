# NeuralViz — Neural Network Training Visualization Platform

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
  <b>Full-stack Neural Network Training Visualization Platform</b> — Upload models, interactive 3D visualization, online hyperparameter tuning, real-time monitoring, Grad-CAM explainability, and multi-experiment comparison
</p>

---

## Features

### Model Management
- Upload PyTorch (`.pt`/`.pth`) and ONNX (`.onnx`) models with automatic parsing
- Built-in MNIST example CNN model for instant experience
- Extract network architecture, parameter count, FLOPs metadata

### 3D Visualization
- Three.js-powered interactive 3D CNN architecture visualization
- Layer-by-layer expansion, rotation/zoom, feature map visualization
- Layer Inspector: view input/output dimensions and parameter counts
- Activation histograms and gradient weight distributions

### Online Training Engine
- Visual configuration: attention mechanisms (SE / CBAM / Self-Attention / ECA / MHSA / GCT), loss functions (CrossEntropy / Focal / Dice / LabelSmoothing / Triplet / Contrastive / Combined), optimizer, learning rate, batch size, epochs
- Data augmentation: random flip, rotation, CutMix, MixUp
- Learning rate schedulers: CosineAnnealing / ReduceLROnPlateau / Step / Exponential
- Training control: pause / resume / stop
- Mixed precision training (AMP) support
- Checkpoint resumption

### Inference & Analysis
- Single/batch image inference with prediction confidence
- Grad-CAM attention heatmaps for model interpretability
- Confusion matrix, ROC/PR curves, precision/recall/F1
- Per-layer feature map viewer

### Experiment Management
- Multi-experiment comparison with different hyperparameter combinations
- Ablation study to evaluate module contributions
- Experiment export (PDF / Excel)
- Historical experiment records and replay

### AI Diagnostics
- Automatic training state analysis: overfitting/underfitting detection
- Gradient vanishing/exploding warnings
- Intelligent hyperparameter adjustment recommendations

### Dataset Management
- Support for CSV / JSON / NPY / NPZ / image folders
- Dataset preview, statistical analysis, format auto-detection
- Built-in MNIST sample dataset

### Auto Training
- Predefined hyperparameter search spaces with grid/random search
- Automatic best configuration comparison
- Batch experiment scheduling

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18 + TypeScript |
| Build Tool | Vite 5 |
| UI | TailwindCSS 4 + shadcn/ui |
| Animation | Framer Motion 11 |
| 3D Rendering | Three.js 0.184 |
| Charts | ECharts 5 |
| Routing | React Router v7 |
| Backend | FastAPI 0.115 |
| Database | SQLAlchemy 2.0 + SQLite |
| ML | PyTorch 2.12 + ONNX Runtime 1.27 |
| Validation | Pydantic v2 |
| Testing | pytest 9.1 (288 tests) |

---

## Quick Start

### Prerequisites

| Tool | Minimum | Check |
|------|---------|-------|
| Python | 3.10+ | `python --version` |
| Node.js | 18+ | `node --version` |
| npm | 9+ | `npm --version` |

### 1. Clone

```bash
git clone https://github.com/qzy2562845731-afk/NeuralViz.git
cd NeuralViz
```

### 2. Backend

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Optional: install dev dependencies
pip install -r requirements-dev.txt

# Optional: configure environment
cp .env.example .env

# Start backend
python main.py
```

Backend runs at `http://localhost:8000`, API docs at `http://localhost:8000/docs`.

### 3. Frontend

```bash
# In project root (new terminal)
npm install
npm run dev
```

Frontend runs at `http://localhost:5173`, API requests proxy to backend.

### 4. Verify

```bash
# Health check
curl http://localhost:8000/api/health
# Expected: {"code":200,"message":"服务运行正常","data":{"status":"online"}}

# Run tests
cd backend
python -m pytest tests/ -v
# Expected: 288 passed
```

---

## Usage

### Quick Tour (5 min)

1. Open `http://localhost:5173`
2. Click **"Start"** to enter the workbench
3. Built-in MNIST CNN model loads automatically
4. Explore the **3D visualization** — rotate, zoom, expand layers
5. Click **"Start Training"** with default hyperparameters
6. Watch real-time Loss/Accuracy curves
7. Upload a handwritten digit image for inference
8. Switch to **Grad-CAM** panel for attention heatmaps

### Custom Model

1. Go to Workbench → Model Management
2. Upload `.pt` / `.pth` / `.onnx` file
3. System auto-parses architecture and generates 3D visualization
4. Inspect layers in Layer Inspector
5. Test predictions in Inference panel

### Online Training

1. Configure hyperparameters:
   - Attention mechanism: SE / CBAM / Self-Attention / ECA / MHSA / GCT
   - Loss function: CrossEntropy / Focal / Dice / LabelSmoothing / Triplet / Contrastive / Combined
   - Optimizer: SGD / Adam / AdamW
   - LR scheduler: CosineAnnealing / ReduceLROnPlateau / Step / Exponential
   - Data augmentation: flip/rotate + CutMix/MixUp
2. Click **"Start Training"**
3. Monitor progress in real-time
4. Experiment auto-saved on completion

---

## Development

```bash
# Frontend type check
npx tsc --noEmit

# Production build
npm run build

# Backend with hot reload
uvicorn main:app --reload --port 8000 --host 0.0.0.0
```

---

## Testing

```bash
cd backend

# All tests
python -m pytest tests/ -v

# Specific module
python -m pytest tests/test_losses.py -v

# With coverage
python -m pytest tests/ --cov=app --cov-report=html
```

**Current: 16 test files, 288 test cases, all passing.**

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repo
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit: `git commit -m 'feat: add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open a Pull Request

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Three.js](https://threejs.org/) — 3D rendering
- [ECharts](https://echarts.apache.org/) — data visualization
- [PyTorch](https://pytorch.org/) — deep learning framework
- [FastAPI](https://fastapi.tiangolo.com/) — high-performance API
- [shadcn/ui](https://ui.shadcn.com/) — UI components
- [TailwindCSS](https://tailwindcss.com/) — CSS framework

---

<p align="center">
  <sub>Built with ❤️ by <a href="https://github.com/qzy2562845731-afk">qzy2562845731-afk</a></sub>
</p>