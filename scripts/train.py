#!/usr/bin/env python3
"""CLI training script — run fine-tuning without the web UI."""

import argparse
import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.core.trainer import run_finetuning


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3 models from the command line")
    parser.add_argument("--config", required=True, help="Path to training config JSON")
    parser.add_argument("--output-dir", default="outputs/cli_run", help="Output directory")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())

    def on_metrics(m):
        step = m.get("step", "?")
        loss = m.get("loss", "?")
        lr = m.get("learning_rate", "?")
        print(f"  Step {step} | loss={loss} | lr={lr}")

    print(f"Starting training: {config.get('model_id', 'unknown')}")
    print(f"Method: {config.get('method', 'lora')}")
    print(f"Output: {args.output_dir}")
    print("-" * 50)

    run_finetuning(
        config=config,
        output_dir=args.output_dir,
        stop_event=threading.Event(),
        metrics_callback=on_metrics,
    )

    print("-" * 50)
    print("Training complete!")


if __name__ == "__main__":
    main()
