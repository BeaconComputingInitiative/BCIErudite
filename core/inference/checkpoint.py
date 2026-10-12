# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
"""Immutable native checkpoints published by an atomic JSON pointer."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import torch
from core.config import SEAL
from core.inference.model import EruditeLM, ModelConfig
from core.inference.tokenizer import BPETokenizer

def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".erudite-", dir=path.parent)
    try:
        with os.fdopen(handle, "w") as out:
            json.dump(payload, out, indent=2); out.write("\n"); out.flush(); os.fsync(out.fileno())
        os.replace(temporary, path)
        path.with_suffix(path.suffix + ".sha256").write_text("# " + SEAL + "\n" + hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.name + "\n")
    finally:
        if Path(temporary).exists(): Path(temporary).unlink()

def save(directory, model, tokenizer, optimizer, metadata, random_state=None):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    filename = f"step-{metadata['step']:08d}.pt"; final = directory / filename
    if final.exists(): raise FileExistsError("Immutable checkpoint already exists")
    bundle = {"format": "erudite-checkpoint-v1", "header": SEAL, "config": vars(model.config),
              "tokenizer": tokenizer.spec, "tokenizer_sha256": tokenizer.fingerprint,
              "weights": model.state_dict(), "metadata": metadata,
              "optimizer": optimizer.state_dict() if optimizer else None,
              "torch_rng": torch.get_rng_state(), "python_rng": random_state,
              "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    handle, temporary = tempfile.mkstemp(prefix="checkpoint-", suffix=".tmp.pt", dir=directory); os.close(handle)
    try:
        with open(temporary, "wb") as file:
            torch.save(bundle, file); file.flush(); os.fsync(file.fileno())
        # Atomic, no-overwrite publication even if two writers target the same step.
        os.link(temporary, final)
    finally:
        if Path(temporary).exists(): Path(temporary).unlink()
    manifest = {"header": SEAL, "format": "erudite-pointer-v1", "file": filename,
                "sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
                "tokenizer_sha256": tokenizer.fingerprint, "step": metadata["step"]}
    atomic_json(final.with_suffix(".json"), manifest)
    atomic_json(directory / "latest.json", manifest)
    return manifest

def load(pointer, device="cpu"):
    pointer = Path(pointer).resolve(); manifest = json.loads(pointer.read_text())
    if manifest.get("format") != "erudite-pointer-v1": raise ValueError("Not an Erudite checkpoint pointer")
    target = (pointer.parent / manifest["file"]).resolve()
    # Read iff pointer stays inside its checkpoint directory and bytes match the declared hash.
    if target.parent != pointer.parent or hashlib.sha256(target.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Checkpoint path/hash mismatch")
    bundle = torch.load(target, map_location="cpu", weights_only=True)
    if bundle.get("format") != "erudite-checkpoint-v1": raise ValueError("Incompatible checkpoint format")
    tokenizer = BPETokenizer.from_spec(bundle["tokenizer"])
    if not (tokenizer.fingerprint == bundle["tokenizer_sha256"] == manifest["tokenizer_sha256"]
            and tokenizer.vocab_size == bundle["config"]["vocab_size"]): raise ValueError("Checkpoint/tokenizer mismatch")
    if device.startswith("cuda") and not torch.cuda.is_available(): raise ValueError("CUDA is unavailable")
    model = EruditeLM(ModelConfig(**bundle["config"]))
    model.load_state_dict(bundle["weights"], strict=True); model.to(device); model.eval()
    return model, tokenizer, bundle, manifest
