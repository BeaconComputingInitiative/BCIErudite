# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
import hashlib
import json
from pathlib import Path
from core.inference.tokenizer import BPETokenizer, SPECIAL

class PreparedData:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if self.manifest.get("format") != "erudite-data-v1": raise ValueError("Invalid data format")
        self.fingerprint = hashlib.sha256((self.directory / "manifest.json").read_bytes()).hexdigest()
        for name, expected in self.manifest["files"].items():
            target = (self.directory / name).resolve()
            if target.parent != self.directory.resolve(): raise ValueError("Unsafe manifest filename")
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected: raise ValueError(f"Data hash mismatch: {name}")
        self.tokenizer = BPETokenizer.load(self.directory / "tokenizer.json")
        if self.tokenizer.fingerprint != self.manifest["tokenizer_sha256"]: raise ValueError("Tokenizer contract mismatch")
        self.rows = {}
        for name in ("train", "validation", "test"):
            records = [json.loads(line) for line in (self.directory / (name + ".jsonl")).read_text().splitlines() if line.strip()]
            if not records: raise ValueError("Empty partition")
            for row in records:
                ids = row["tokens"]; mask = row["assistant_mask"]
                if len(ids) < 2 or len(ids) != len(mask) or any(type(t) is not int or not 0 <= t < self.tokenizer.vocab_size for t in ids):
                    raise ValueError("Invalid token sequence")
                if any(type(t) is not int or t not in (0, 1) for t in mask): raise ValueError("Invalid assistant mask")
            self.rows[name] = records
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            for key in ("source_sha256", "group"):
                if {r[key] for r in self.rows[a]} & {r[key] for r in self.rows[b]}: raise ValueError("Partition leakage")

    def windows(self, partition, context, objective="all"):
        if objective not in {"all", "assistant"}: raise ValueError("Unknown objective")
        windows = []
        for row in self.rows[partition]:
            # Assistant-only refinement deliberately excludes standalone prose records.
            if objective == "assistant" and row["kind"] != "chat": continue
            ids, mask = row["tokens"], row["assistant_mask"]
            for start in range(0, len(ids) - 1, context):
                tokens = ids[start:start + context + 1]
                targets = ids[start + 1:start + len(tokens)]
                if objective == "assistant":
                    targets = [t if enabled else -100 for t, enabled in zip(targets, mask[start + 1:start + len(tokens)])]
                if len(tokens) > 1 and any(t != -100 for t in targets): windows.append((tokens[:-1], targets))
        if not windows: raise ValueError(f"No target windows in {partition}/{objective}")
        return windows

def batch(rows, device):
    import torch
    length = max(len(x) for x, _ in rows)
    inputs = [x + [SPECIAL["pad"]] * (length - len(x)) for x, _ in rows]
    labels = [y + [-100] * (length - len(y)) for _, y in rows]
    return torch.tensor(inputs, dtype=torch.long, device=device), torch.tensor(labels, dtype=torch.long, device=device)
