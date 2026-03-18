# 🔧 Qwen Fine-Tuner

A production-ready toolkit for fine-tuning Qwen3 models with a polished web UI.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![React](https://img.shields.io/badge/react-18-61DAFB.svg)

## Features

### Fine-Tuning Pipeline
- **Qwen3 model support**: 0.6B, 1.7B, 4B, 8B, 14B, 32B
- **LoRA & QLoRA** with fully configurable hyperparameters
- **Multi-GPU support**: NVIDIA (CUDA), Intel Arc (XPU/IPEX), and CPU fallback
- **Chat/instruction** and **raw text** format support
- Train/val splitting, checkpointing, early stopping, loss tracking
- Export merged models or standalone LoRA adapters

### Web UI
- **Training Data Editor** — Import JSONL/CSV/Parquet, visual conversation editor, bulk ops, validation, stats dashboard
- **Training Dashboard** — Launch jobs, configure hyperparameters, real-time loss curves via WebSocket, run comparison, GPU type detection
- **Playground** — Chat with base vs fine-tuned models side-by-side, adjustable generation params, bookmarks

## Architecture

```
┌─────────────────────────────────────────────┐
│                React Frontend               │
│  (TypeScript + shadcn/ui + Recharts)        │
├─────────────────────────────────────────────┤
│           FastAPI Backend (REST + WS)       │
├──────────┬──────────┬───────────────────────┤
│ Data Svc │Train Svc │   Inference Service   │
├──────────┴──────────┴───────────────────────┤
│  Device Detection (CUDA / XPU / CPU)        │
│  transformers / peft / trl                  │
│  bitsandbytes (CUDA) | IPEX (Intel Arc)     │
└─────────────────────────────────────────────┘
```

## Quick Start

### Docker — NVIDIA GPU (default)

```bash
git clone https://github.com/ClawrissaBot/qwen-fine-tuner.git
cd qwen-fine-tuner
docker compose up --build
```

### Docker — Intel Arc GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.intel.yml up --build
```

### Docker — CPU only

```bash
docker compose up --build  # Will auto-detect and fall back to CPU
```

Open http://localhost:3000

### Manual Setup

**Backend (NVIDIA):**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-cuda.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Backend (Intel Arc):**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
# Install PyTorch with XPU support first
pip install torch==2.4.1 --index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/
pip install -r requirements-xpu.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Backend (CPU only):**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## GPU Support

The backend auto-detects the available GPU at startup:

| GPU Type | Device | Quantization (QLoRA) | Requirements |
|----------|--------|---------------------|--------------|
| NVIDIA (CUDA) | `cuda` | bitsandbytes (4/8-bit) | `requirements-cuda.txt` |
| Intel Arc (XPU) | `xpu` | IPEX (4/8-bit) or LoRA fallback | `requirements-xpu.txt` |
| CPU | `cpu` | Not available (LoRA only) | `requirements.txt` |

The GPU status endpoint (`/api/training/gpu`) returns a `device_type` field (`"cuda"`, `"xpu"`, or `"cpu"`) along with GPU details when available.

### Intel Arc Setup Notes

- Requires Intel GPU drivers and `intel-extension-for-pytorch`
- Docker: uses `/dev/dri` passthrough (see `docker-compose.intel.yml`)
- QLoRA on XPU uses `intel-extension-for-transformers`; falls back to standard LoRA if unavailable
- BF16 is used by default on XPU for optimal performance

## Supported Models

| Model | Parameters | VRAM (LoRA) | VRAM (QLoRA) |
|-------|-----------|-------------|--------------|
| Qwen/Qwen3-0.6B | 0.6B | ~4 GB | ~2 GB |
| Qwen/Qwen3-1.7B | 1.7B | ~8 GB | ~4 GB |
| Qwen/Qwen3-4B | 4B | ~12 GB | ~6 GB |
| Qwen/Qwen3-8B | 8B | ~20 GB | ~10 GB |
| Qwen/Qwen3-14B | 14B | ~32 GB | ~18 GB |
| Qwen/Qwen3-32B | 32B | ~72 GB | ~36 GB |

## Configuration

See [`configs/`](configs/) for example training configs. All hyperparameters can also be set via the web UI.

## Example Data

The [`data/examples/`](data/examples/) directory contains sample datasets in chat and raw text format to get started.

## License

MIT — see [LICENSE](LICENSE).
