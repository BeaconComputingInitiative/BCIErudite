# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
import argparse
import json
from core.config import path, training
from core.inference.checkpoint import load
from training.data import PreparedData
from training.train_model import evaluate

def run(pointer, directory, split="test", device="cpu", batch_size=4):
    model, tokenizer, bundle, _ = load(pointer, device)
    data = PreparedData(directory)
    if data.tokenizer.fingerprint != tokenizer.fingerprint or data.fingerprint != bundle["metadata"]["data_manifest_sha256"]:
        raise ValueError("Evaluation dataset/tokenizer differs from this checkpoint")
    objective = bundle["metadata"]["config"]["objective"]
    return {"split": split, "objective": objective, **evaluate(model, data.windows(split, model.config.context, objective), batch_size, device)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data"); parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--device", default="cpu"); args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.data or path(training()["prepared_dir"]), args.split, args.device), indent=2))
