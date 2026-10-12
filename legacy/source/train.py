"""
=====================================================================
Developer: Chance Joaquin Boone
Project: BCIai Platform (Computer Beacon Intelligence)
Module: Standalone Training Launcher
Developer Seal: BCI-CUSTOM-LLM-CORE-V7-TORCH
=====================================================================
"""

import torch
from tokenizer import CustomCharTokenizer
from model import CharacterLLM

def run_phase1_training():
    corpus = "The quick brown fox jumps over the lazy dog. BCI system metrics report optimal execution status."
    tokenizer = CustomCharTokenizer(corpus)
    data = torch.tensor(tokenizer.encode(corpus), dtype=torch.long)

    block_size = 16
    batch_size = 8
    epochs = 500

    model = CharacterLLM(vocab_size=tokenizer.vocab_size, block_size=block_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(1, epochs + 1):
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log metrics if and only if current loop iteration hits a 100-step boundary
        if epoch % 100 == 0:
            print(f"Step {epoch:03d}/{epochs} | Loss: {loss.item():.4f}")

if __name__ == "__main__":
    run_phase1_training()
