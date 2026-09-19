import torch
from source.tokenizer import CustomBPETokenizer
from source.model import ModernLLM

text = "The quick brown fox jumps over the lazy dog. Computer Beacon Intelligence BCI platform. " * 10

tokenizer = CustomBPETokenizer()
tokenizer.train(text, target_vocab_size=280)
tokens = tokenizer.encode(text)

data = torch.tensor(tokens, dtype=torch.long)
block_size = 32

model = ModernLLM(vocab_size=tokenizer.vocab_size, n_embed=64, block_size=block_size, n_layer=2, n_head=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)

print("Training model...")
for step in range(1, 301):
    ix = torch.randint(0, len(data) - block_size - 1, (8,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])

    logits, loss, _ = model(x, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if step % 50 == 0:
        print(f"Step {step}/300 | Loss: {loss.item():.4f}")

# Save & Load Checkpoint
model.save_checkpoint("bci_model.pt")
loaded_model = ModernLLM.load_checkpoint("bci_model.pt")

prompt = "The quick "
prompt_tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

# 1. Greedy Generation Test
generated_greedy = list(prompt_tokens[0].numpy())
for tok_id in loaded_model.generate_kv(prompt_tokens, max_new_tokens=40, greedy=True):
    generated_greedy.append(tok_id)

print("\n--- Greedy Decoding ---")
print(tokenizer.decode(generated_greedy))

# 2. Nucleus (Top-P) Sampling Test
generated_topp = list(prompt_tokens[0].numpy())
for tok_id in loaded_model.generate_kv(prompt_tokens, max_new_tokens=40, temperature=0.7, top_p=0.9, greedy=False):
    generated_topp.append(tok_id)

print("\n--- Nucleus (Top-P = 0.9, Temp = 0.7) ---")
print(tokenizer.decode(generated_topp))
