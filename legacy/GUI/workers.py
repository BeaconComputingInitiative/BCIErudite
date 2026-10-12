"""
=====================================================================
Developer: Chance Joaquin Boone
Project: BCIai Platform (Computer Beacon Intelligence)
Module: Multi-Threaded Execution Workers
Developer Seal: BCI-CUSTOM-LLM-CORE-V7-TORCH
=====================================================================
"""

import time
import torch
from PyQt5.QtCore import QThread, pyqtSignal
from source.tokenizer import CustomCharTokenizer
from source.model import CharacterLLM

class TrainingThread(QThread):
    progress_signal = pyqtSignal(int, float, str)
    finished_signal = pyqtSignal(object, object)

    def __init__(self, corpus: str, epochs: int = 500):
        super().__init__()
        self.corpus = corpus
        self.epochs = epochs

    def run(self):
        tokenizer = CustomCharTokenizer(self.corpus)
        tokens = tokenizer.encode(self.corpus)
        data = torch.tensor(tokens, dtype=torch.long)

        block_size = 16
        batch_size = 8

        if len(data) <= block_size + 1:
            self.progress_signal.emit(0, 0.0, "Error: Corpus string too short for sequence blocks.")
            return

        model = CharacterLLM(vocab_size=tokenizer.vocab_size, block_size=block_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        for epoch in range(1, self.epochs + 1):
            ix = torch.randint(len(data) - block_size - 1, (batch_size,))
            x = torch.stack([data[i : i + block_size] for i in ix])
            y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Progress updates dispatch if and only if epoch modulo condition evaluates true
            if epoch % max(1, self.epochs // 10) == 0 or epoch == self.epochs:
                pct = int((epoch / self.epochs) * 100)
                msg = f"Step {epoch:03d}/{self.epochs} | Loss: {loss.item():.4f}"
                self.progress_signal.emit(pct, loss.item(), msg)
                time.sleep(0.01)

        self.finished_signal.emit(model, tokenizer)

class InferenceWorker(QThread):
    token_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, model: CharacterLLM, tokenizer: CustomCharTokenizer, prompt: str, max_tokens: int = 80):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.max_tokens = max_tokens

    def run(self):
        self.model.eval()
        prompt_tokens = self.tokenizer.encode(self.prompt)
        if not prompt_tokens:
            prompt_tokens = [0]

        idx = torch.tensor([prompt_tokens], dtype=torch.long)
        
        # Token decoding emits if and only if model returns active token generation steps
        for tok_id in self.model.generate(idx, max_new_tokens=self.max_tokens):
            char_str = self.tokenizer.decode([tok_id])
            self.token_signal.emit(char_str)
            time.sleep(0.02)

        self.finished_signal.emit()
