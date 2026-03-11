"""Training job management — orchestrates fine-tuning runs."""

from __future__ import annotations

import json
import os
import shutil
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

JOBS_FILE = OUTPUT_DIR / "jobs.json"


class TrainingManager:
    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stop_flags: dict[str, threading.Event] = {}
        self._load_jobs()

    def _load_jobs(self):
        if JOBS_FILE.exists():
            self._jobs = {j["id"]: j for j in json.loads(JOBS_FILE.read_text())}

    def _save_jobs(self):
        JOBS_FILE.write_text(json.dumps(list(self._jobs.values()), indent=2, default=str))

    def start_job(self, config: dict) -> str:
        job_id = str(uuid.uuid4())[:8]
        name = config.get("output_name") or f"{config['model_id'].split('/')[-1]}_{job_id}"
        job = {
            "id": job_id,
            "name": name,
            "config": config,
            "status": "queued",
            "created_at": datetime.utcnow().isoformat(),
            "started_at": None,
            "finished_at": None,
            "metrics": [],
            "error": None,
            "output_dir": str(OUTPUT_DIR / name),
        }
        self._jobs[job_id] = job
        self._save_jobs()

        stop_event = threading.Event()
        self._stop_flags[job_id] = stop_event
        t = threading.Thread(target=self._run_training, args=(job_id, stop_event), daemon=True)
        self._threads[job_id] = t
        t.start()
        return job_id

    def _run_training(self, job_id: str, stop_event: threading.Event):
        """Execute training in a background thread."""
        job = self._jobs[job_id]
        config = job["config"]
        job["status"] = "running"
        job["started_at"] = datetime.utcnow().isoformat()
        self._save_jobs()

        try:
            self._broadcast({"type": "job_update", "job_id": job_id, "status": "running"})

            from ..core.trainer import run_finetuning
            run_finetuning(
                config=config,
                output_dir=job["output_dir"],
                stop_event=stop_event,
                metrics_callback=lambda m: self._on_metrics(job_id, m),
            )

            if stop_event.is_set():
                job["status"] = "stopped"
            else:
                job["status"] = "completed"
        except Exception as e:
            job["status"] = "failed"
            job["error"] = traceback.format_exc()
        finally:
            job["finished_at"] = datetime.utcnow().isoformat()
            self._save_jobs()
            self._broadcast({"type": "job_update", "job_id": job_id, "status": job["status"]})

    def _on_metrics(self, job_id: str, metrics: dict):
        job = self._jobs.get(job_id)
        if job:
            job["metrics"].append(metrics)
            self._save_jobs()
            self._broadcast({"type": "metrics", "job_id": job_id, "data": metrics})

    def _broadcast(self, data: dict):
        from .ws_manager import ws_manager
        ws_manager.broadcast_sync(data)

    def list_jobs(self) -> list[dict]:
        return sorted(self._jobs.values(), key=lambda j: j["created_at"], reverse=True)

    def get_job(self, job_id: str) -> dict | None:
        return self._jobs.get(job_id)

    def stop_job(self, job_id: str):
        if job_id in self._stop_flags:
            self._stop_flags[job_id].set()

    def delete_job(self, job_id: str):
        self.stop_job(job_id)
        if job_id in self._jobs:
            job = self._jobs.pop(job_id)
            out = Path(job["output_dir"])
            if out.exists():
                shutil.rmtree(out, ignore_errors=True)
            self._save_jobs()

    def export_model(self, job_id: str, merge: bool = True) -> Path:
        job = self._jobs.get(job_id)
        if not job:
            raise ValueError("Job not found")
        if job["status"] != "completed":
            raise ValueError("Job not completed")

        output_dir = Path(job["output_dir"])
        if merge:
            from ..core.trainer import merge_and_export
            merged_dir = output_dir / "merged"
            merge_and_export(
                base_model=job["config"]["model_id"],
                adapter_path=str(output_dir / "adapter"),
                output_path=str(merged_dir),
            )
            return merged_dir
        return output_dir / "adapter"

    def gpu_status(self) -> dict:
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            return {
                "gpus": [
                    {
                        "id": g.id,
                        "name": g.name,
                        "memory_total": g.memoryTotal,
                        "memory_used": g.memoryUsed,
                        "memory_free": g.memoryFree,
                        "gpu_util": g.load * 100,
                        "temperature": g.temperature,
                    }
                    for g in gpus
                ]
            }
        except Exception:
            return {"gpus": [], "error": "GPU monitoring unavailable"}

    def shutdown(self):
        for event in self._stop_flags.values():
            event.set()


training_manager = TrainingManager()
