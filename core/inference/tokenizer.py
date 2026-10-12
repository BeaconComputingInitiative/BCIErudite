# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
"""Native byte BPE with reserved chat symbols and a self-describing vocabulary."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

PATTERN = r"\s+|\w+|[^\w\s]+"
SPECIAL = {"bos": 256, "eos": 257, "pad": 258, "system": 259, "user": 260, "assistant": 261}
BASE = 262

class BPETokenizer:
    def __init__(self, merges=()):
        self.merges = [tuple(p) for p in merges]
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.ranks = {}
        for rank, pair in enumerate(self.merges):
            if len(pair) != 2 or pair in self.ranks or any(t not in self.vocab for t in pair):
                raise ValueError("Invalid or out-of-order BPE merge")
            index = BASE + rank
            self.ranks[pair] = index
            self.vocab[index] = self.vocab[pair[0]] + self.vocab[pair[1]]

    @property
    def vocab_size(self): return BASE + len(self.merges)
    @property
    def spec(self):
        return {"format": "erudite-bpe-v1", "pattern": PATTERN, "special": SPECIAL,
                "merges": [list(p) for p in self.merges], "vocab_size": self.vocab_size}
    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.spec, sort_keys=True).encode()).hexdigest()

    @staticmethod
    def merge(ids, pair, index):
        out = []; i = 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                out.append(index); i += 2
            else: out.append(ids[i]); i += 1
        return out

    @classmethod
    def train(cls, texts, target_vocab=1024, max_chars=1_000_000):
        # Learn iff target can hold bytes+specials and the corpus budget is positive.
        if target_vocab < BASE or max_chars <= 0: raise ValueError("Invalid tokenizer budget")
        counts = Counter(); remaining = max_chars
        for text in texts:
            if not isinstance(text, str): raise ValueError("Tokenizer corpus must be text")
            part = text[:remaining]; remaining -= len(part)
            for piece in re.findall(PATTERN, part): counts[tuple(piece.encode("utf-8"))] += 1
            if not remaining: break
        if not counts: raise ValueError("Tokenizer corpus is empty")
        merges = []
        for index in range(BASE, target_vocab):
            pairs = Counter()
            for ids, weight in counts.items():
                for pair in zip(ids, ids[1:]): pairs[pair] += weight
            if not pairs: break
            # Stable tie-breaking makes this native tokenizer reproducible.
            best = min(pairs, key=lambda pair: (-pairs[pair], pair))
            merges.append(best)
            next_counts = Counter()
            for ids, weight in counts.items():
                next_counts[tuple(cls.merge(ids, best, index))] += weight
            counts = next_counts
        return cls(merges)

    def encode(self, text):
        if not isinstance(text, str): raise ValueError("Expected text")
        output = []
        for piece in re.findall(PATTERN, text):
            ids = list(piece.encode("utf-8"))
            while len(ids) > 1:
                candidates = [(self.ranks[p], p) for p in zip(ids, ids[1:]) if p in self.ranks]
                if not candidates: break
                index, pair = min(candidates)
                ids = self.merge(ids, pair, index)
            output.extend(ids)
        return output

    def decode(self, ids):
        chunks = []
        for index in ids:
            if index in self.vocab: chunks.append(self.vocab[index])
            elif index not in SPECIAL.values(): raise ValueError(f"Unknown token ID {index}")
        return b"".join(chunks).decode("utf-8", errors="replace")

    def conversation(self, messages, generation=False):
        ids = [SPECIAL["bos"]]; mask = [0]
        for message in messages:
            role = message["role"]
            if role not in {"system", "user", "assistant"}: raise ValueError("Unknown role")
            body = self.encode(message["content"])
            tokens = [SPECIAL[role]] + body + [SPECIAL["eos"]]
            ids.extend(tokens)
            mask.extend([0] + [int(role == "assistant")] * (len(body) + 1))
        if generation:
            ids.append(SPECIAL["assistant"]); mask.append(0)
        return ids, mask

    def save(self, path):
        Path(path).write_text(json.dumps({"header": "Beacon Computing Initiative | Developer Seal: Chance Joaquin Boone (CJB)",
                                        **self.spec, "sha256": self.fingerprint}, indent=2) + "\n")
    @classmethod
    def from_spec(cls, spec):
        if spec.get("format") != "erudite-bpe-v1" or spec.get("pattern") != PATTERN or spec.get("special") != SPECIAL:
            raise ValueError("Incompatible tokenizer contract")
        tokenizer = cls(spec["merges"])
        if tokenizer.vocab_size != spec["vocab_size"]: raise ValueError("Vocabulary size mismatch")
        if "sha256" in spec and tokenizer.fingerprint != spec["sha256"]: raise ValueError("Tokenizer hash mismatch")
        return tokenizer
    @classmethod
    def load(cls, path): return cls.from_spec(json.loads(Path(path).read_text()))
