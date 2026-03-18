"""Core fine-tuning logic for Qwen3 models — supports CUDA, Intel XPU, and CPU."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer, SFTConfig

from .device import (
    get_device,
    get_device_map,
    get_dtype,
    is_cuda,
    is_xpu,
    is_cpu,
    has_bitsandbytes,
    has_ipex_quantization,
)

logger = logging.getLogger(__name__)


class MetricsCallback(TrainerCallback):
    """Send metrics via callback during training."""

    def __init__(self, callback: Callable[[dict], None], stop_event: threading.Event | None = None):
        self.callback = callback
        self.stop_event = stop_event

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs:
            metrics = {
                "step": state.global_step,
                "epoch": round(state.epoch or 0, 2),
                **{k: round(v, 6) if isinstance(v, float) else v for k, v in logs.items()},
            }
            self.callback(metrics)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.stop_event and self.stop_event.is_set():
            control.should_training_stop = True


class EarlyStoppingCallback(TrainerCallback):
    """Stop training if eval loss doesn't improve."""

    def __init__(self, patience: int = 5):
        self.patience = patience
        self.best_loss = float("inf")
        self.wait = 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            loss = metrics["eval_loss"]
            if loss < self.best_loss:
                self.best_loss = loss
                self.wait = 0
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    control.should_training_stop = True


def load_dataset_from_config(config: dict) -> tuple[Dataset, Dataset | None]:
    """Load and split dataset for training."""
    from ..services.dataset_service import dataset_service

    dataset_id = config["dataset_id"]
    examples = dataset_service._load_examples(dataset_id)
    data_format = config.get("data_format", "chat")

    if data_format == "chat":
        records = []
        for e in examples:
            msgs = e.get("messages", [])
            if msgs:
                records.append({"messages": msgs})
        ds = Dataset.from_list(records)
    else:
        records = []
        for e in examples:
            text = e.get("text", "")
            if text:
                records.append({"text": text})
        ds = Dataset.from_list(records)

    val_split = config.get("val_split", 0.1)
    if val_split > 0 and len(ds) > 10:
        split = ds.train_test_split(test_size=val_split, seed=42)
        return split["train"], split["test"]
    return ds, None


def _build_quantization_config(config: dict):
    """Build quantization config appropriate for the detected device.

    Returns (quantization_config, method_override) where method_override is
    the effective method to use ('qlora' or 'lora' if quantization isn't available).
    """
    method = config.get("method", "lora")
    if method != "qlora":
        return None, method

    bits = config.get("quant_bits", 4)

    # CUDA: use bitsandbytes
    if is_cuda() and has_bitsandbytes():
        compute_dtype = get_dtype(prefer_bf16=config.get("bf16", False))
        if bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            ), "qlora"
        else:
            return BitsAndBytesConfig(load_in_8bit=True), "qlora"

    # XPU: try intel-extension-for-transformers
    if is_xpu() and has_ipex_quantization():
        logger.info("Using Intel Extension for Transformers quantization on XPU")
        # IEFT handles quantization at model load time; we return a marker
        # and handle it in the model loading step
        return {"_ipex_quantization": True, "bits": bits}, "qlora"

    # Fallback: no quantization available, use LoRA instead
    logger.warning(
        "QLoRA requested but no quantization backend available for %s. "
        "Falling back to standard LoRA.",
        get_device(),
    )
    return None, "lora"


def _load_model(model_id: str, quant_config, config: dict):
    """Load model with device-appropriate settings."""
    compute_dtype = get_dtype(prefer_bf16=config.get("bf16", False))
    device_map = get_device_map()

    # XPU with IPEX quantization
    if isinstance(quant_config, dict) and quant_config.get("_ipex_quantization"):
        try:
            from intel_extension_for_transformers.transformers import AutoModelForCausalLM as IEFTModel

            bits = quant_config.get("bits", 4)
            logger.info("Loading model with IPEX %d-bit quantization on XPU", bits)
            model = IEFTModel.from_pretrained(
                model_id,
                load_in_4bit=(bits == 4),
                load_in_8bit=(bits == 8),
                torch_dtype=compute_dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
            return model
        except Exception as e:
            logger.warning("IPEX quantization failed (%s), loading without quantization", e)
            quant_config = None

    # Standard loading (CUDA with bitsandbytes, or no quantization)
    bnb_config = quant_config if isinstance(quant_config, BitsAndBytesConfig) else None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=compute_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # For XPU without quantization, ensure model is on the right device
    if is_xpu() and device_map == "xpu":
        try:
            import intel_extension_for_pytorch as ipex  # noqa: F401
            model = model.to("xpu")
        except Exception:
            pass

    return model


def run_finetuning(
    config: dict,
    output_dir: str,
    stop_event: threading.Event | None = None,
    metrics_callback: Callable[[dict], None] | None = None,
):
    """Execute a fine-tuning run."""
    device = get_device()
    logger.info("Starting fine-tuning on device: %s", device)

    model_id = config["model_id"]
    output_path = Path(output_dir)
    adapter_dir = output_path / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    (output_path / "config.json").write_text(json.dumps(config, indent=2))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config
    quant_config, effective_method = _build_quantization_config(config)

    # Load model
    model = _load_model(model_id, quant_config, config)

    if effective_method == "qlora":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.get("gradient_checkpointing", True)
        )

    # LoRA config
    lora_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    train_ds, val_ds = load_dataset_from_config(config)

    # Formatting function
    data_format = config.get("data_format", "chat")

    def format_chat(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    if data_format == "chat":
        train_ds = train_ds.map(format_chat)
        if val_ds:
            val_ds = val_ds.map(format_chat)

    # Determine precision flags
    use_bf16 = config.get("bf16", False)
    use_fp16 = config.get("fp16", True) and not use_bf16

    # XPU prefers bf16; CPU should use neither
    if is_xpu():
        use_bf16 = True
        use_fp16 = False
    elif is_cpu():
        use_fp16 = False
        use_bf16 = False

    # Training arguments
    training_args = SFTConfig(
        output_dir=str(adapter_dir),
        num_train_epochs=config.get("epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 4),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 2e-4),
        lr_scheduler_type=config.get("lr_scheduler", "cosine"),
        warmup_ratio=config.get("warmup_ratio", 0.05),
        weight_decay=config.get("weight_decay", 0.01),
        max_seq_length=config.get("max_seq_length", 2048),
        fp16=use_fp16,
        bf16=use_bf16,
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        save_steps=config.get("save_steps", 100),
        eval_steps=config.get("eval_steps", 50) if val_ds else None,
        eval_strategy="steps" if val_ds else "no",
        logging_steps=10,
        save_total_limit=3,
        load_best_model_at_end=True if val_ds else False,
        metric_for_best_model="eval_loss" if val_ds else None,
        report_to="none",
        dataset_text_field="text",
        packing=False,
    )

    # Callbacks
    callbacks = []
    if metrics_callback:
        callbacks.append(MetricsCallback(metrics_callback, stop_event))
    elif stop_event:
        callbacks.append(MetricsCallback(lambda m: None, stop_event))

    patience = config.get("early_stopping_patience", 0)
    if patience > 0 and val_ds:
        callbacks.append(EarlyStoppingCallback(patience=patience))

    # Train
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    trainer.train()

    # Save adapter
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    if config.get("merge_adapter", False):
        merge_and_export(model_id, str(adapter_dir), str(output_path / "merged"))


def merge_and_export(base_model: str, adapter_path: str, output_path: str):
    """Merge LoRA adapter back into base model and save."""
    from peft import PeftModel

    compute_dtype = get_dtype(prefer_bf16=False)
    device_map = get_device_map()

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=compute_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
