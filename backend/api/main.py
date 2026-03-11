"""Qwen Fine-Tuner — FastAPI backend."""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .routes_data import router as data_router
from .routes_training import router as training_router
from .routes_inference import router as inference_router
from ..services.training_manager import training_manager
from ..services.ws_manager import ws_manager

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))

for d in [DATA_DIR, UPLOAD_DIR, OUTPUT_DIR]:
    d.mkdir(parents=True, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    training_manager.shutdown()


app = FastAPI(
    title="Qwen Fine-Tuner",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data_router, prefix="/api/data", tags=["data"])
app.include_router(training_router, prefix="/api/training", tags=["training"])
app.include_router(inference_router, prefix="/api/inference", tags=["inference"])


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/models")
async def list_models():
    """Return supported Qwen3 models."""
    return {
        "models": [
            {"id": "Qwen/Qwen3-0.6B", "name": "Qwen3 0.6B", "params": "0.6B", "vram_lora": "~4 GB", "vram_qlora": "~2 GB"},
            {"id": "Qwen/Qwen3-1.7B", "name": "Qwen3 1.7B", "params": "1.7B", "vram_lora": "~8 GB", "vram_qlora": "~4 GB"},
            {"id": "Qwen/Qwen3-4B", "name": "Qwen3 4B", "params": "4B", "vram_lora": "~12 GB", "vram_qlora": "~6 GB"},
            {"id": "Qwen/Qwen3-8B", "name": "Qwen3 8B", "params": "8B", "vram_lora": "~20 GB", "vram_qlora": "~10 GB"},
            {"id": "Qwen/Qwen3-14B", "name": "Qwen3 14B", "params": "14B", "vram_lora": "~32 GB", "vram_qlora": "~18 GB"},
            {"id": "Qwen/Qwen3-32B", "name": "Qwen3 32B", "params": "32B", "vram_lora": "~72 GB", "vram_qlora": "~36 GB"},
        ]
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
