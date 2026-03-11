"""Dataset storage and manipulation service."""

from __future__ import annotations

import io
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

DATA_DIR = Path("data")
DATASETS_DIR = DATA_DIR / "datasets"
EXPORT_DIR = DATA_DIR / "exports"


class DatasetService:
    def __init__(self):
        DATASETS_DIR.mkdir(parents=True, exist_ok=True)
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    def _meta_path(self, dataset_id: str) -> Path:
        return DATASETS_DIR / dataset_id / "meta.json"

    def _examples_path(self, dataset_id: str) -> Path:
        return DATASETS_DIR / dataset_id / "examples.json"

    def _load_meta(self, dataset_id: str) -> dict | None:
        p = self._meta_path(dataset_id)
        if not p.exists():
            return None
        return json.loads(p.read_text())

    def _save_meta(self, dataset_id: str, meta: dict):
        p = self._meta_path(dataset_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(meta, indent=2))

    def _load_examples(self, dataset_id: str) -> list[dict]:
        p = self._examples_path(dataset_id)
        if not p.exists():
            return []
        return json.loads(p.read_text())

    def _save_examples(self, dataset_id: str, examples: list[dict]):
        p = self._examples_path(dataset_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(examples, indent=2))

    def list_datasets(self) -> list[dict]:
        results = []
        if not DATASETS_DIR.exists():
            return results
        for d in sorted(DATASETS_DIR.iterdir()):
            meta = self._load_meta(d.name)
            if meta:
                meta["example_count"] = len(self._load_examples(d.name))
                results.append(meta)
        return results

    def create_dataset(self, name: str, description: str = "", fmt: str = "chat") -> dict:
        dataset_id = str(uuid.uuid4())[:8]
        meta = {
            "id": dataset_id,
            "name": name,
            "description": description,
            "format": fmt,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        self._save_meta(dataset_id, meta)
        self._save_examples(dataset_id, [])
        meta["example_count"] = 0
        return meta

    def get_dataset(self, dataset_id: str) -> dict | None:
        meta = self._load_meta(dataset_id)
        if meta:
            meta["example_count"] = len(self._load_examples(dataset_id))
        return meta

    def delete_dataset(self, dataset_id: str):
        import shutil
        d = DATASETS_DIR / dataset_id
        if d.exists():
            shutil.rmtree(d)

    def list_examples(self, dataset_id: str, offset: int = 0, limit: int = 50, search: str = "", tag: str = "") -> dict:
        examples = self._load_examples(dataset_id)
        if search:
            s = search.lower()
            examples = [e for e in examples if s in json.dumps(e).lower()]
        if tag:
            examples = [e for e in examples if tag in e.get("tags", [])]
        total = len(examples)
        return {"total": total, "examples": examples[offset : offset + limit]}

    def add_example(self, dataset_id: str, data: dict) -> dict:
        examples = self._load_examples(dataset_id)
        example = {
            "id": str(uuid.uuid4())[:8],
            "messages": data.get("messages"),
            "text": data.get("text"),
            "tags": data.get("tags", []),
            "created_at": datetime.utcnow().isoformat(),
        }
        examples.append(example)
        self._save_examples(dataset_id, examples)
        self._touch(dataset_id)
        return example

    def update_example(self, dataset_id: str, example_id: str, data: dict) -> dict:
        examples = self._load_examples(dataset_id)
        for i, e in enumerate(examples):
            if e["id"] == example_id:
                examples[i].update({k: v for k, v in data.items() if v is not None})
                self._save_examples(dataset_id, examples)
                self._touch(dataset_id)
                return examples[i]
        raise ValueError("Example not found")

    def delete_example(self, dataset_id: str, example_id: str):
        examples = self._load_examples(dataset_id)
        examples = [e for e in examples if e["id"] != example_id]
        self._save_examples(dataset_id, examples)
        self._touch(dataset_id)

    def reorder_examples(self, dataset_id: str, order: list[str]):
        examples = self._load_examples(dataset_id)
        by_id = {e["id"]: e for e in examples}
        reordered = [by_id[eid] for eid in order if eid in by_id]
        remaining = [e for e in examples if e["id"] not in set(order)]
        self._save_examples(dataset_id, reordered + remaining)

    def bulk_operation(self, dataset_id: str, action: str, example_ids: list[str], tag: str | None = None) -> dict:
        examples = self._load_examples(dataset_id)
        ids_set = set(example_ids)
        affected = 0
        if action == "delete":
            before = len(examples)
            examples = [e for e in examples if e["id"] not in ids_set]
            affected = before - len(examples)
        elif action == "duplicate":
            dupes = []
            for e in examples:
                if e["id"] in ids_set:
                    dupe = {**e, "id": str(uuid.uuid4())[:8], "created_at": datetime.utcnow().isoformat()}
                    dupes.append(dupe)
                    affected += 1
            examples.extend(dupes)
        elif action == "tag" and tag:
            for e in examples:
                if e["id"] in ids_set:
                    tags = e.get("tags", [])
                    if tag not in tags:
                        tags.append(tag)
                        e["tags"] = tags
                        affected += 1
        elif action == "untag" and tag:
            for e in examples:
                if e["id"] in ids_set:
                    tags = e.get("tags", [])
                    if tag in tags:
                        tags.remove(tag)
                        e["tags"] = tags
                        affected += 1
        self._save_examples(dataset_id, examples)
        self._touch(dataset_id)
        return {"affected": affected}

    def import_file(self, dataset_id: str, content: bytes, filename: str) -> dict:
        examples = self._load_examples(dataset_id)
        new_examples = []
        if filename.endswith(".jsonl"):
            for line in content.decode("utf-8").strip().split("\n"):
                if line.strip():
                    row = json.loads(line)
                    new_examples.append(self._normalize_row(row))
        elif filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
            for _, row in df.iterrows():
                new_examples.append(self._normalize_row(row.to_dict()))
        elif filename.endswith(".parquet"):
            df = pd.read_parquet(io.BytesIO(content))
            for _, row in df.iterrows():
                new_examples.append(self._normalize_row(row.to_dict()))
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        examples.extend(new_examples)
        self._save_examples(dataset_id, examples)
        self._touch(dataset_id)
        return {"imported": len(new_examples)}

    def export_file(self, dataset_id: str, fmt: str = "jsonl") -> Path:
        examples = self._load_examples(dataset_id)
        out = EXPORT_DIR / f"{dataset_id}.{fmt}"
        if fmt == "jsonl":
            lines = [json.dumps(e) for e in examples]
            out.write_text("\n".join(lines))
        elif fmt == "csv":
            df = pd.DataFrame(examples)
            df.to_csv(out, index=False)
        elif fmt == "parquet":
            df = pd.DataFrame(examples)
            df.to_parquet(out, index=False)
        return out

    def compute_stats(self, dataset_id: str) -> dict:
        examples = self._load_examples(dataset_id)
        total = len(examples)
        if total == 0:
            return {"total": 0, "avg_turns": 0, "avg_text_length": 0, "tag_counts": {}, "turn_distribution": {}}

        turn_counts = []
        text_lengths = []
        tag_counts: dict[str, int] = {}

        for e in examples:
            msgs = e.get("messages") or []
            turn_counts.append(len(msgs))
            text = e.get("text") or ""
            total_text = text + " ".join(m.get("content", "") for m in msgs)
            text_lengths.append(len(total_text))
            for t in e.get("tags", []):
                tag_counts[t] = tag_counts.get(t, 0) + 1

        turn_dist: dict[str, int] = {}
        for tc in turn_counts:
            key = str(tc)
            turn_dist[key] = turn_dist.get(key, 0) + 1

        return {
            "total": total,
            "avg_turns": sum(turn_counts) / total if total else 0,
            "avg_text_length": sum(text_lengths) / total if total else 0,
            "min_text_length": min(text_lengths) if text_lengths else 0,
            "max_text_length": max(text_lengths) if text_lengths else 0,
            "tag_counts": tag_counts,
            "turn_distribution": turn_dist,
        }

    def validate(self, dataset_id: str) -> dict:
        examples = self._load_examples(dataset_id)
        meta = self._load_meta(dataset_id)
        fmt = meta.get("format", "chat") if meta else "chat"
        issues = []
        for i, e in enumerate(examples):
            eid = e.get("id", f"index-{i}")
            if fmt == "chat":
                msgs = e.get("messages")
                if not msgs or not isinstance(msgs, list):
                    issues.append({"example_id": eid, "severity": "error", "message": "Missing or invalid messages array"})
                    continue
                for j, m in enumerate(msgs):
                    if "role" not in m:
                        issues.append({"example_id": eid, "severity": "error", "message": f"Message {j}: missing role"})
                    elif m["role"] not in ("system", "user", "assistant"):
                        issues.append({"example_id": eid, "severity": "warning", "message": f"Message {j}: unusual role '{m['role']}'"})
                    if not m.get("content"):
                        issues.append({"example_id": eid, "severity": "warning", "message": f"Message {j}: empty content"})
                # Check for at least user+assistant
                roles = [m.get("role") for m in msgs]
                if "user" not in roles or "assistant" not in roles:
                    issues.append({"example_id": eid, "severity": "warning", "message": "Should have at least one user and one assistant message"})
            elif fmt == "raw":
                if not e.get("text"):
                    issues.append({"example_id": eid, "severity": "error", "message": "Missing text field"})
        return {"total_examples": len(examples), "issues": issues, "valid": len(issues) == 0}

    def _normalize_row(self, row: dict) -> dict:
        example: dict[str, Any] = {
            "id": str(uuid.uuid4())[:8],
            "tags": [],
            "created_at": datetime.utcnow().isoformat(),
        }
        if "messages" in row:
            msgs = row["messages"]
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            example["messages"] = msgs
        elif "text" in row:
            example["text"] = str(row["text"])
        elif "instruction" in row or "input" in row or "output" in row:
            messages = []
            if row.get("instruction"):
                messages.append({"role": "system", "content": str(row["instruction"])})
            if row.get("input"):
                messages.append({"role": "user", "content": str(row["input"])})
            if row.get("output"):
                messages.append({"role": "assistant", "content": str(row["output"])})
            example["messages"] = messages
        elif "prompt" in row and "completion" in row:
            example["messages"] = [
                {"role": "user", "content": str(row["prompt"])},
                {"role": "assistant", "content": str(row["completion"])},
            ]
        else:
            # Best effort: use first string column as text
            for v in row.values():
                if isinstance(v, str) and len(v) > 10:
                    example["text"] = v
                    break
        return example

    def _touch(self, dataset_id: str):
        meta = self._load_meta(dataset_id)
        if meta:
            meta["updated_at"] = datetime.utcnow().isoformat()
            self._save_meta(dataset_id, meta)


dataset_service = DatasetService()
