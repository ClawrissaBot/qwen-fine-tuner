"""Model inference service for the playground."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

OUTPUT_DIR = Path("outputs")
BOOKMARKS_FILE = OUTPUT_DIR / "bookmarks.json"


class InferenceService:
    def __init__(self):
        self._loaded_model: str | None = None
        self._loaded_adapter: str | None = None
        self._model = None
        self._tokenizer = None
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _ensure_model(self, model_id: str, adapter_path: str | None = None):
        """Load model + optional adapter, caching across calls."""
        if self._loaded_model == model_id and self._loaded_adapter == adapter_path and self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        self._tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        if adapter_path and Path(adapter_path).exists():
            self._model = PeftModel.from_pretrained(self._model, adapter_path)
            self._model = self._model.merge_and_unload()

        self._model.eval()
        self._loaded_model = model_id
        self._loaded_adapter = adapter_path

    def generate(
        self,
        model_id: str,
        adapter_path: str | None,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        max_tokens: int = 512,
    ) -> dict:
        self._ensure_model(model_id, adapter_path)
        import torch

        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                top_p=top_p,
                top_k=top_k,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
        response = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        return {
            "response": response,
            "tokens_generated": len(new_tokens),
            "model_id": model_id,
            "adapter_path": adapter_path,
        }

    def generate_stream(
        self,
        model_id: str,
        adapter_path: str | None,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        max_tokens: int = 512,
    ) -> Generator[str, None, None]:
        self._ensure_model(model_id, adapter_path)
        import torch
        from transformers import TextIteratorStreamer
        import threading

        text = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        streamer = TextIteratorStreamer(self._tokenizer, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_tokens,
            "temperature": max(temperature, 0.01),
            "top_p": top_p,
            "top_k": top_k,
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
            "streamer": streamer,
        }

        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()

        for token_text in streamer:
            yield f"data: {json.dumps({'token': token_text})}\n\n"
        yield "data: [DONE]\n\n"

    def list_adapters(self) -> list[dict]:
        adapters = []
        if not OUTPUT_DIR.exists():
            return adapters
        for job_dir in OUTPUT_DIR.iterdir():
            adapter_dir = job_dir / "adapter"
            if adapter_dir.exists() and (adapter_dir / "adapter_config.json").exists():
                adapters.append({
                    "name": job_dir.name,
                    "path": str(adapter_dir),
                })
            merged_dir = job_dir / "merged"
            if merged_dir.exists():
                adapters.append({
                    "name": f"{job_dir.name} (merged)",
                    "path": str(merged_dir),
                    "merged": True,
                })
        return adapters

    def save_bookmark(self, data: dict) -> dict:
        bookmarks = self._load_bookmarks()
        bookmark = {
            "id": str(uuid.uuid4())[:8],
            **data,
            "created_at": datetime.utcnow().isoformat(),
        }
        bookmarks.append(bookmark)
        BOOKMARKS_FILE.write_text(json.dumps(bookmarks, indent=2))
        return bookmark

    def list_bookmarks(self) -> list[dict]:
        return self._load_bookmarks()

    def delete_bookmark(self, bookmark_id: str):
        bookmarks = [b for b in self._load_bookmarks() if b["id"] != bookmark_id]
        BOOKMARKS_FILE.write_text(json.dumps(bookmarks, indent=2))

    def _load_bookmarks(self) -> list[dict]:
        if BOOKMARKS_FILE.exists():
            return json.loads(BOOKMARKS_FILE.read_text())
        return []


inference_service = InferenceService()
