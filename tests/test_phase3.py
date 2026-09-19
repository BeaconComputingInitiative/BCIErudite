import torch
from source.tokenizer import CustomBPETokenizer
from source.model import ModernLLM

text = "The quick brown fox jumps over the lazy dog. Computer Beacon Intelligence BCI platform."

# Train BPE Tokenizer
tokenizer = CustomBPETokenizer()
tokenizer.train(text, target_vocab_size=280)
tokens = tokenizer.encode(text)

data = torch.tensor(tokens, dtype=torch.long)
block_size = 16

# Instantiating RoPE + RMSNorm + SwiGLU Architecture
model = ModernLLM(vocab_size=tokenizer.vocab_size, n_embed=64, block_size=block_size, n_layer=2, n_head=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

# Quick training loop sanity check
for step in range(1, 201):
    ix = torch.randint(len(data) - block_size - 1, (4,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    logits, loss = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step:03d}/200 | Loss: {loss.item():.4f}")

# Generation Check
prompt = "The quick "
prompt_tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

generated = list(prompt_tokens[0].numpy())
for tok_id in model.generate(prompt_tokens, max_new_tokens=20, temperature=0.8, top_k=5):
    generated.append(tok_id)

print("\nGenerated Output:")
print(tokenizer.decode(generated))
