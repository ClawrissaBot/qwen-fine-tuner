# 🔧 Qwen Fine-Tuner

A production-ready toolkit for fine-tuning Qwen3 models with a polished web UI.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![React](https://img.shields.io/badge/react-18-61DAFB.svg)

<!-- TODO: Add screenshots -->
<!-- ![Dashboard](docs/screenshots/dashboard.png) -->
<!-- ![Data Editor](docs/screenshots/data-editor.png) -->
<!-- ![Playground](docs/screenshots/playground.png) -->

## Features

### Fine-Tuning Pipeline
- **Qwen3 model support**: 0.6B, 1.7B, 4B, 8B, 14B, 32B
- **LoRA & QLoRA** with fully configurable hyperparameters
- **Chat/instruction** and **raw text** format support
- Train/val splitting, checkpointing, early stopping, loss tracking
- Export merged models or standalone LoRA adapters
- Built on `transformers`, `peft`, `trl`, `bitsandbytes`

### Web UI
- **Training Data Editor** — Import JSONL/CSV/Parquet, visual conversation editor, bulk ops, validation, stats dashboard
- **Training Dashboard** — Launch jobs, configure hyperparameters, real-time loss curves via WebSocket, run comparison
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
│  transformers / peft / trl / bitsandbytes   │
└─────────────────────────────────────────────┘
```

## Quick Start

### Docker (recommended)

```bash
git clone https://github.com/Zigdon7/qwen-fine-tuner.git
cd qwen-fine-tuner
docker compose up --build
```

Open http://localhost:3000

### Manual Setup

**Backend:**
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
