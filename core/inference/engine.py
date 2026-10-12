# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
import codecs
import hashlib
import threading
import torch
from core.config import path
from core.inference.checkpoint import load
from core.inference.tokenizer import SPECIAL

class Engine:
    def __init__(self, config):
        self.config = config; self.model = None; self.tokenizer = None; self.revision = None
        self.cancel = threading.Event()
    def ensure_loaded(self):
        if self.config["device"] == "cpu": torch.set_num_threads(self.config.get("cpu_threads", 2))
        pointer = path(self.config["checkpoint"])
        revision = hashlib.sha256(pointer.read_bytes()).hexdigest()
        # Reload iff checkpoint pointer bytes change. Generation gate serializes this mutation.
        if self.revision != revision:
            model, tokenizer, bundle, manifest = load(pointer, self.config["device"])
            self.model = model; self.tokenizer = tokenizer; self.metadata = bundle["metadata"]
            self.revision = revision; self.manifest = manifest
        return self.manifest
    def stream(self, messages):
        self.cancel.clear(); self.ensure_loaded()
        # The trained prompt is checkpoint-owned, not a client-supplied system override.
        messages = [{"role": "system", "content": self.metadata["system_prompt"]}] + list(messages)
        cap = self.config["max_output_tokens"]; budget = self.model.config.context - cap; dropped = 0
        ids, _ = self.tokenizer.conversation(messages, generation=True)
        while len(ids) > budget and len(messages) > 2:
            del messages[1:3]; dropped += 1
            ids, _ = self.tokenizer.conversation(messages, generation=True)
        if len(ids) > budget: raise ValueError("System prompt and current message do not fit checkpoint context")
        yield {"event": "start", "model": "Erudite", "step": self.manifest["step"], "dropped_turns": dropped}
        tensor = torch.tensor([ids], dtype=torch.long, device=self.config["device"])
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace"); count = 0
        for token in self.model.generate(tensor, cap, self.config["temperature"], self.config["top_p"], SPECIAL["eos"]):
            if self.cancel.is_set():
                yield {"event": "cancelled"}; return
            count += 1
            text = decoder.decode(self.tokenizer.vocab[token])
            if text: yield {"event": "token", "text": text}
        tail = decoder.decode(b"", final=True)
        if tail: yield {"event": "token", "text": tail}
        yield {"event": "done", "generated_tokens": count}
