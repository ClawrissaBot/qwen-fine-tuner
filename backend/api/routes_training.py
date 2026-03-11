"""Training job routes."""

from __future__ import annotations

from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException

from ..services.training_manager import training_manager

router = APIRouter()


class TrainingConfig(BaseModel):
    dataset_id: str
    model_id: str = "Qwen/Qwen3-0.6B"
    method: str = "lora"  # lora | qlora
    data_format: str = "chat"  # chat | raw

    # LoRA params
    lora_r: int = Field(16, description="LoRA rank")
    lora_alpha: int = Field(32, description="LoRA alpha")
    lora_dropout: float = Field(0.05, description="LoRA dropout")
    target_modules: list[str] = Field(
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        description="Modules to apply LoRA to",
    )

    # Training params
    epochs: int = Field(3, description="Number of epochs")
    batch_size: int = Field(4, description="Per-device batch size")
    gradient_accumulation_steps: int = Field(4, description="Gradient accumulation steps")
    learning_rate: float = Field(2e-4, description="Learning rate")
    lr_scheduler: str = Field("cosine", description="LR scheduler type")
    warmup_ratio: float = Field(0.05, description="Warmup ratio")
    weight_decay: float = Field(0.01, description="Weight decay")
    max_seq_length: int = Field(2048, description="Max sequence length")
    val_split: float = Field(0.1, description="Validation split ratio")
    save_steps: int = Field(100, description="Save checkpoint every N steps")
    eval_steps: int = Field(50, description="Evaluate every N steps")
    early_stopping_patience: int = Field(5, description="Early stopping patience (0=disabled)")
    fp16: bool = Field(True, description="Use FP16 mixed precision")
    bf16: bool = Field(False, description="Use BF16 mixed precision")
    gradient_checkpointing: bool = Field(True, description="Enable gradient checkpointing")

    # QLoRA
    quant_bits: int = Field(4, description="Quantization bits for QLoRA (4 or 8)")

    # Output
    output_name: str = Field("", description="Run name (auto-generated if empty)")
    merge_adapter: bool = Field(False, description="Merge adapter into base model after training")


class ExportRequest(BaseModel):
    merge: bool = True


@router.post("/start")
async def start_training(config: TrainingConfig):
    job_id = training_manager.start_job(config.model_dump())
    return {"job_id": job_id}


@router.get("/jobs")
async def list_jobs():
    return {"jobs": training_manager.list_jobs()}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    job = training_manager.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.post("/jobs/{job_id}/stop")
async def stop_job(job_id: str):
    training_manager.stop_job(job_id)
    return {"ok": True}


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    training_manager.delete_job(job_id)
    return {"ok": True}


@router.post("/jobs/{job_id}/export")
async def export_model(job_id: str, req: ExportRequest):
    path = training_manager.export_model(job_id, req.merge)
    return {"path": str(path)}


@router.get("/gpu")
async def gpu_status():
    return training_manager.gpu_status()
