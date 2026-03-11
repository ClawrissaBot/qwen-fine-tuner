"""Core fine-tuning logic for Qwen3 models."""

from __future__ import annotations

import json
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


def run_finetuning(
    config: dict,
    output_dir: str,
    stop_event: threading.Event | None = None,
    metrics_callback: Callable[[dict], None] | None = None,
):
    """Execute a fine-tuning run."""
    model_id = config["model_id"]
    method = config.get("method", "lora")
    output_path = Path(output_dir)
    adapter_dir = output_path / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    (output_path / "config.json").write_text(json.dumps(config, indent=2))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config for QLoRA
    bnb_config = None
    if method == "qlora":
        bits = config.get("quant_bits", 4)
        if bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if config.get("bf16") else torch.float16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if config.get("bf16") else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if method == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=config.get("gradient_checkpointing", True))

    # LoRA config
    lora_config = LoraConfig(
        r=config.get("lora_r", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
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
        text = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
        return {"text": text}

    if data_format == "chat":
        train_ds = train_ds.map(format_chat)
        if val_ds:
            val_ds = val_ds.map(format_chat)

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
        fp16=config.get("fp16", True) and not config.get("bf16", False),
        bf16=config.get("bf16", False),
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

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    Path(output_path).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
