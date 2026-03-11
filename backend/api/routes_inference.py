"""Inference / playground routes."""

from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..services.inference_service import inference_service

router = APIRouter()


class GenerateRequest(BaseModel):
    model_id: str = "Qwen/Qwen3-0.6B"
    adapter_path: str | None = None
    messages: list[dict] = []
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=0)
    max_tokens: int = Field(512, ge=1, le=8192)
    stream: bool = False


class BookmarkRequest(BaseModel):
    model_id: str
    adapter_path: str | None = None
    messages: list[dict]
    response: str
    generation_params: dict = {}
    note: str = ""


@router.post("/generate")
async def generate(req: GenerateRequest):
    if req.stream:
        return StreamingResponse(
            inference_service.generate_stream(
                model_id=req.model_id,
                adapter_path=req.adapter_path,
                messages=req.messages,
                temperature=req.temperature,
                top_p=req.top_p,
                top_k=req.top_k,
                max_tokens=req.max_tokens,
            ),
            media_type="text/event-stream",
        )
    result = inference_service.generate(
        model_id=req.model_id,
        adapter_path=req.adapter_path,
        messages=req.messages,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        max_tokens=req.max_tokens,
    )
    return result


@router.get("/adapters")
async def list_adapters():
    return {"adapters": inference_service.list_adapters()}


@router.post("/bookmarks")
async def save_bookmark(req: BookmarkRequest):
    return inference_service.save_bookmark(req.model_dump())


@router.get("/bookmarks")
async def list_bookmarks():
    return {"bookmarks": inference_service.list_bookmarks()}


@router.delete("/bookmarks/{bookmark_id}")
async def delete_bookmark(bookmark_id: str):
    inference_service.delete_bookmark(bookmark_id)
    return {"ok": True}
