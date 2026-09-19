from source.tokenizer import CustomBPETokenizer

sample_text = "The quick brown fox jumps over the lazy dog. Computer Beacon Intelligence BCI platform."

tokenizer = CustomBPETokenizer()
tokenizer.train(sample_text, target_vocab_size=280, verbose=True)

encoded = tokenizer.encode(sample_text)
decoded = tokenizer.decode(encoded)

print(f"\nOriginal Length (chars): {len(sample_text)}")
print(f"Encoded Length (tokens): {len(encoded)}")
print(f"Decoded Text Matches: {sample_text == decoded}")
