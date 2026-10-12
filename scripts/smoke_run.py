# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
"""Reproducible synthetic demo: never reads or changes the user's datasets."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import ROOT, SEAL
from training.prepare_dataset import prepare
from training.train_model import train
from training.evaluate_model import run as evaluate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="models/smoke"); parser.add_argument("--steps", type=int, default=30)
    args = parser.parse_args(); output = (ROOT / args.output).resolve()
    if output.exists(): raise SystemExit("Choose a new output path; existing weights will not be overwritten")
    with tempfile.TemporaryDirectory(prefix="erudite-synthetic-") as directory:
        temporary = Path(directory); raw = temporary / "synthetic.jsonl"
        raw.write_text("".join(json.dumps({"text": f"Synthetic example {i}. Erudite learns token sequences with a local neural network. These demonstration sentences are not an assistant training corpus."}) + "\n" for i in range(30)))
        data = temporary / "prepared"; prepare([raw], data, vocab=300)
        config = {"header": SEAL, "prepared_dir": str(data), "output_dir": str(output),
                  "architecture": {"width": 32, "layers": 2, "heads": 4, "context": 256},
                  "device": "cpu", "cpu_threads": 2, "steps": args.steps, "batch_size": 2,
                  "learning_rate": 0.003, "weight_decay": 0.1, "warmup_steps": 0,
                  "grad_clip": 1.0, "eval_every": max(1, args.steps), "seed": 42, "objective": "all"}
        config_path = temporary / "training.json"; config_path.write_text(json.dumps(config))
        pointer = train(config_path)
        print(json.dumps({"header": SEAL, "purpose": "synthetic mechanics demonstration only",
                          "checkpoint": str(pointer), "test": evaluate(pointer, data)}, indent=2))
    print("Temporary synthetic data removed. Your datasets were not accessed.")
    print(f"API: python main.py --checkpoint {pointer}")
