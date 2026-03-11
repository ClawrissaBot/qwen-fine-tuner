"""Dataset management routes."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from pydantic import BaseModel

from ..services.dataset_service import dataset_service

router = APIRouter()


class DatasetCreate(BaseModel):
    name: str
    description: str = ""
    format: str = "chat"  # chat | raw


class ExampleUpdate(BaseModel):
    messages: list[dict] | None = None
    text: str | None = None
    tags: list[str] = []


class BulkOperation(BaseModel):
    action: str  # delete | duplicate | tag | untag
    example_ids: list[str]
    tag: str | None = None


@router.get("/datasets")
async def list_datasets():
    return {"datasets": dataset_service.list_datasets()}


@router.post("/datasets")
async def create_dataset(req: DatasetCreate):
    ds = dataset_service.create_dataset(req.name, req.description, req.format)
    return ds


@router.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    ds = dataset_service.get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "Dataset not found")
    return ds


@router.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    dataset_service.delete_dataset(dataset_id)
    return {"ok": True}


@router.get("/datasets/{dataset_id}/examples")
async def list_examples(
    dataset_id: str,
    offset: int = 0,
    limit: int = 50,
    search: str = "",
    tag: str = "",
):
    return dataset_service.list_examples(dataset_id, offset, limit, search, tag)


@router.post("/datasets/{dataset_id}/examples")
async def add_example(dataset_id: str, example: ExampleUpdate):
    return dataset_service.add_example(dataset_id, example.model_dump())


@router.put("/datasets/{dataset_id}/examples/{example_id}")
async def update_example(dataset_id: str, example_id: str, example: ExampleUpdate):
    return dataset_service.update_example(dataset_id, example_id, example.model_dump())


@router.delete("/datasets/{dataset_id}/examples/{example_id}")
async def delete_example(dataset_id: str, example_id: str):
    dataset_service.delete_example(dataset_id, example_id)
    return {"ok": True}


@router.post("/datasets/{dataset_id}/examples/reorder")
async def reorder_examples(dataset_id: str, order: list[str]):
    dataset_service.reorder_examples(dataset_id, order)
    return {"ok": True}


@router.post("/datasets/{dataset_id}/bulk")
async def bulk_operation(dataset_id: str, op: BulkOperation):
    return dataset_service.bulk_operation(dataset_id, op.action, op.example_ids, op.tag)


@router.post("/datasets/{dataset_id}/import")
async def import_data(dataset_id: str, file: UploadFile = File(...)):
    content = await file.read()
    filename = file.filename or "data.jsonl"
    return dataset_service.import_file(dataset_id, content, filename)


@router.get("/datasets/{dataset_id}/export")
async def export_data(dataset_id: str, format: str = "jsonl"):
    path = dataset_service.export_file(dataset_id, format)
    media = {"jsonl": "application/jsonl", "csv": "text/csv", "parquet": "application/octet-stream"}
    from fastapi.responses import FileResponse
    return FileResponse(path, media_type=media.get(format, "application/octet-stream"), filename=f"dataset.{format}")


@router.get("/datasets/{dataset_id}/stats")
async def dataset_stats(dataset_id: str):
    return dataset_service.compute_stats(dataset_id)


@router.post("/datasets/{dataset_id}/validate")
async def validate_dataset(dataset_id: str):
    return dataset_service.validate(dataset_id)
