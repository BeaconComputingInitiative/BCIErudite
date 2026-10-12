# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
"""Validate, deduplicate payloads, group/split documents, then learn BPE on TRAIN only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from core.config import SEAL, runtime
from core.guardrails.validation import record
from core.inference.tokenizer import BPETokenizer, SPECIAL

def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def prepare(sources, destination, vocab=1024, seed=42, tokenizer_chars=1_000_000):
    destination = Path(destination)
    if destination.exists(): raise FileExistsError("Choose a new prepared directory; prior data is preserved")
    records = {}; source_info = []; duplicates = 0; total = 0
    file_hashes = set()
    for source in sorted(map(Path, sources)):
        raw = source.read_bytes(); sha = hashlib.sha256(raw).hexdigest()
        source_info.append({"file": source.name, "sha256": sha, "bytes": len(raw)})
        # Identical files need not be decoded repeatedly; their provenance is still recorded.
        if sha in file_hashes: continue
        file_hashes.add(sha)
        for number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
            if not line.strip(): continue
            try: payload = record(json.loads(line))
            except (TypeError, ValueError) as error: raise ValueError(f"{source.name}:{number}: {error}") from error
            total += 1; key = digest(payload)
            if key in records: duplicates += 1; continue
            # Exact repeated user prompts stay together even when their answers differ.
            group = digest([m["content"] for m in payload["messages"] if m["role"] == "user"]) if "messages" in payload else key
            records[key] = {"payload": payload, "source_sha256": key, "group": group}
    if len(records) < 10: raise ValueError("Need at least ten distinct payloads for an engineering split")
    groups = {}
    for row in records.values(): groups.setdefault(row["group"], []).append(row)
    if len(groups) < 3: raise ValueError("Need at least three independent groups for train/validation/test")
    order = sorted(groups); random.Random(seed).shuffle(order)
    # Whole groups partitioned before tokenization/windowing. Exact overlap is impossible.
    partitions = {"train": [], "validation": [], "test": []}
    target = max(1, len(records) // 10)
    for index, group in enumerate(order):
        remaining = len(order) - index
        if len(partitions["validation"]) < target and remaining > 2: name = "validation"
        elif len(partitions["test"]) < target and remaining > 1: name = "test"
        else: name = "train"
        partitions[name].extend(groups[group])
    # Extremely small group sets still need each partition represented.
    if not all(partitions.values()):
        partitions = {"validation": groups[order[0]], "test": groups[order[1]],
                      "train": [r for group in order[2:] for r in groups[group]]}
    def text(row):
        p = row["payload"]
        return p["text"] if "text" in p else "\n".join(m["content"] for m in p["messages"])
    tokenizer = BPETokenizer.train((text(r) for r in partitions["train"]), vocab, tokenizer_chars)
    destination.mkdir(parents=True)
    tokenizer.save(destination / "tokenizer.json")
    hashes = {}; counts = {}; system = runtime()["system_prompt"]
    for name, rows in partitions.items():
        path = destination / (name + ".jsonl"); token_count = 0
        with path.open("x") as out:
            for row in rows:
                p = row["payload"]
                if "text" in p:
                    tokens = [SPECIAL["bos"]] + tokenizer.encode(p["text"]) + [SPECIAL["eos"]]
                    assistant_mask = [0] * len(tokens); kind = "text"
                else:
                    items = p["messages"]
                    if items[0]["role"] != "system": items = [{"role": "system", "content": system}] + items
                    tokens, assistant_mask = tokenizer.conversation(items); kind = "chat"
                token_count += len(tokens)
                out.write(json.dumps({"header": SEAL, "tokens": tokens, "assistant_mask": assistant_mask,
                                      "kind": kind, "source_sha256": row["source_sha256"], "group": row["group"]}) + "\n")
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        counts[name] = {"documents": len(rows), "tokens": token_count, "groups": len({r["group"] for r in rows})}
    hashes["tokenizer.json"] = hashlib.sha256((destination / "tokenizer.json").read_bytes()).hexdigest()
    manifest = {"header": SEAL, "format": "erudite-data-v1", "seed": seed, "tokenizer_sha256": tokenizer.fingerprint,
                "vocab_size": tokenizer.vocab_size, "system_prompt": system, "sources": source_info,
                "unique_payloads": len(records), "duplicate_rows_excluding_identical_files": duplicates,
                "decoded_rows": total, "tokenizer_training_chars_limit": tokenizer_chars, "partitions": counts, "files": hashes}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("sources", nargs="+"); parser.add_argument("--output", required=True)
    parser.add_argument("--vocab", type=int, default=1024); parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer-chars", type=int, default=1_000_000); args = parser.parse_args()
    print(json.dumps(prepare(args.sources, args.output, args.vocab, args.seed, args.tokenizer_chars), indent=2))
