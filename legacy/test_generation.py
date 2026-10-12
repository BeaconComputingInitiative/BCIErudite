import torch
from source.tokenizer import CustomBPETokenizer
from source.model import ModernLLM

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = CustomBPETokenizer()
tokenizer.load("bci_tokenizer.json")
model = ModernLLM.load_checkpoint("bci_model.pt", device=device)
model.eval()

prompt = "The BCIai platform"
prompt_tokens = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)

print(f"Prompt: '{prompt}'\n")
generated = list(prompt_tokens[0].cpu().numpy())
for tok_id in model.generate_kv(prompt_tokens, max_new_tokens=40, temperature=0.7, top_p=0.9, greedy=False):
    generated.append(tok_id)

print("--- Generated Output ---")
print(tokenizer.decode(generated))
