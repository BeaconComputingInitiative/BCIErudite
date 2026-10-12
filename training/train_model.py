# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
"""Fresh causal-LM training, or explicit continuation of Erudite's own native model."""
import argparse
import json
import math
from pathlib import Path
import random
import time
import torch
from core.config import SEAL, path, read, training
from core.inference.checkpoint import atomic_json, load, save
from core.inference.model import EruditeLM, ModelConfig
from training.data import PreparedData, batch

@torch.inference_mode()
def evaluate(model, windows, batch_size, device):
    previous = model.training; model.eval(); summed = 0.0; tokens = 0
    try:
        for start in range(0, len(windows), batch_size):
            inputs, labels = batch(windows[start:start + batch_size], device)
            _, loss, _ = model(inputs, labels); count = int((labels != -100).sum())
            summed += loss.item() * count; tokens += count
    finally: model.train(previous)
    return {"loss": summed / tokens, "perplexity": math.exp(min(summed / tokens, 50)), "evaluated_tokens": tokens}

def train(config=None, resume=None):
    c = training() if config is None else read(config)
    if not (c["steps"] > 0 and c["batch_size"] > 0 and c["learning_rate"] > 0 and c["eval_every"] > 0
            and c["grad_clip"] > 0 and 0 <= c["warmup_steps"] < c["steps"]): raise ValueError("Invalid training schedule")
    if c["device"].startswith("cuda") and not torch.cuda.is_available(): raise ValueError("CUDA unavailable")
    if c["device"] == "cpu": torch.set_num_threads(c.get("cpu_threads", 2))
    torch.manual_seed(c["seed"]); generator = random.Random(c["seed"])
    data = PreparedData(path(c["prepared_dir"]))
    architecture = ModelConfig(vocab_size=data.tokenizer.vocab_size, **c["architecture"])
    start = 0; resumed = None
    if resume:
        model, tokenizer, resumed, _ = load(resume, c["device"])
        if model.config != architecture or tokenizer.fingerprint != data.tokenizer.fingerprint:
            raise ValueError("Resume architecture/tokenizer differs from prepared data")
        if resumed["metadata"]["data_manifest_sha256"] != data.fingerprint: raise ValueError("Resume dataset differs")
        start = resumed["metadata"]["step"]
        if start >= c["steps"]: raise ValueError("Resume needs a larger total steps value")
    else: model = EruditeLM(architecture).to(c["device"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=c["learning_rate"], weight_decay=c["weight_decay"])
    if resumed and resumed["optimizer"]:
        optimizer.load_state_dict(resumed["optimizer"])
        for group in optimizer.param_groups: group["weight_decay"] = c["weight_decay"]
        if resumed.get("python_rng") is not None: generator.setstate(resumed["python_rng"])
        torch.set_rng_state(resumed["torch_rng"])
        if c["device"].startswith("cuda") and resumed.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(resumed["cuda_rng"])
    train_windows = data.windows("train", architecture.context, c["objective"])
    validation = data.windows("validation", architecture.context, c["objective"])
    output = path(c["output_dir"])
    if output.exists(): raise FileExistsError("Choose a new output_dir; existing runs will not be overwritten")
    output.mkdir(parents=True)
    parameters = sum(p.numel() for p in model.parameters())
    initial = evaluate(model, validation, c["batch_size"], c["device"])
    print(json.dumps({"initialization": "native_resume" if resume else "random_from_scratch",
                      "parameters": parameters, "initial_validation": initial}), flush=True)
    history = []; best = float("inf"); begin = time.monotonic(); seen = 0; model.train()
    for step in range(start + 1, c["steps"] + 1):
        inputs, labels = batch(generator.choices(train_windows, k=c["batch_size"]), c["device"])
        optimizer.zero_grad(set_to_none=True)
        _, loss, _ = model(inputs, labels)
        if not torch.isfinite(loss): raise RuntimeError("Non-finite loss; training stopped")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"], error_if_nonfinite=True)
        if step <= c["warmup_steps"]: factor = step / max(1, c["warmup_steps"])
        else:
            progress = (step - c["warmup_steps"]) / (c["steps"] - c["warmup_steps"])
            factor = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups: group["lr"] = c["learning_rate"] * factor
        optimizer.step(); seen += int((labels != -100).sum())
        if step % c["eval_every"] == 0 or step == c["steps"]:
            metrics = evaluate(model, validation, c["batch_size"], c["device"])
            event = {"step": step, "train_loss": loss.item(), "validation": metrics,
                     "tokens_this_run": seen, "elapsed_seconds": time.monotonic() - begin}
            history.append(event); print(json.dumps(event), flush=True)
            metadata = {"step": step, "parameters": parameters, "validation": metrics,
                        "data_manifest_sha256": data.fingerprint, "system_prompt": data.manifest["system_prompt"],
                        "config": c, "resumed_from": str(resume) if resume else None}
            pointer = save(output, model, data.tokenizer, optimizer, metadata, generator.getstate())
            if metrics["loss"] < best:
                best = metrics["loss"]; atomic_json(output / "best.json", pointer)
            atomic_json(output / "metrics.json", {"header": SEAL, "initial_validation": initial, "history": history,
                                                  "parameters": parameters, "initialization": "resume" if resume else "random"})
    return output / "best.json"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--config"); parser.add_argument("--resume"); args = parser.parse_args()
    print(train(args.config, args.resume))
