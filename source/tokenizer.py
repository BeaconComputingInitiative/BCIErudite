"""
=====================================================================
Developer: Chance Joaquin Boone
Project: BCIai Platform (Computer Beacon Intelligence)
Module: Production Custom BPE Tokenizer with Pre-tokenization
Developer Seal: BCI-CUSTOM-TOKENIZER-V3
=====================================================================
"""

import json
import os
import regex as re
from typing import List, Dict, Tuple

# GPT-2 style pre-tokenization regex pattern
GPT2_SPLIT_PATTERN = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

class CustomBPETokenizer:
    """Byte-Pair Encoding (BPE) Tokenizer with Regex Pre-tokenization."""
    def __init__(self):
        self.vocab_size = 256
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.pat = GPT2_SPLIT_PATTERN

    def get_stats(self, ids: List[int], counts: Dict[Tuple[int, int], int] = None) -> Dict[Tuple[int, int], int]:
        """Counts frequency of consecutive token pairs."""
        if counts is None:
            counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self, ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        """Merges all occurrences of a specific token pair into a new token index."""
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text: str, target_vocab_size: int = 4096, verbose: bool = True):
        """Trains BPE vocabulary on pre-tokenized chunks."""
        self.vocab_size = target_vocab_size
        num_merges = target_vocab_size - 256
        
        # Pre-tokenize corpus into chunks
        chunks = re.findall(self.pat, text)
        id_chunks = [list(chunk.encode("utf-8")) for chunk in chunks if chunk]

        if verbose:
            print(f"Training BPE tokenizer to vocab size {target_vocab_size} ({num_merges} merges) across {len(id_chunks)} chunks...")

        for i in range(num_merges):
            stats = {}
            for chunk_ids in id_chunks:
                self.get_stats(chunk_ids, stats)
            
            if not stats:
                break
            
            best_pair = max(stats, key=stats.get)
            new_token_id = 256 + i
            
            id_chunks = [self.merge(chunk_ids, best_pair, new_token_id) for chunk_ids in id_chunks]
            self.merges[best_pair] = new_token_id
            self.vocab[new_token_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose and (i + 1) % 1000 == 0:
                print(f"Completed merge {i + 1}/{num_merges}")

        if verbose:
            print(f"✅ Tokenizer training complete. Final vocab size: {len(self.vocab)}")

    def _encode_chunk(self, chunk_bytes: bytes) -> List[int]:
        """Encodes a single pre-tokenized chunk using learned merges."""
        ids = list(chunk_bytes)
        while len(ids) >= 2:
            stats = self.get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            idx = self.merges[pair]
            ids = self.merge(ids, pair, idx)
        return ids

    def encode(self, text: str) -> List[int]:
        """Encodes full text into token integers."""
        chunks = re.findall(self.pat, text)
        ids = []
        for chunk in chunks:
            chunk_bytes = chunk.encode("utf-8")
            ids.extend(self._encode_chunk(chunk_bytes))
        return ids

    def decode(self, ids: List[int]) -> str:
        """Decodes token integer sequence back into readable UTF-8 string."""
        tokens = b"".join(self.vocab[idx] for idx in ids if idx in self.vocab)
        return tokens.decode("utf-8", errors="replace")

    def save(self, filepath: str):
        """Saves tokenizer vocabulary and merges to disk."""
        data = {
            "vocab_size": self.vocab_size,
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
            "vocab": {str(k): list(v) for k, v in self.vocab.items()}
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)
        print(f"✅ Tokenizer saved to {filepath}")

    def load(self, filepath: str):
        """Loads tokenizer vocabulary and merges from disk."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.vocab_size = data["vocab_size"]
        self.merges = {tuple(map(int, k.split(","))): v for k, v in data["merges"].items()}
        self.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        print(f"✅ Tokenizer loaded from {filepath} (Vocab size: {len(self.vocab)})")
