# Developers seal: Chance Joaquin Boone

import os
import json
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class BCIDataset(Dataset):
    """High-throughput dataset loader for BCI JSONL and markdown text corpuses."""
    def __init__(self, data_path: str, max_length: int = 128):
        self.data_path = data_path
        self.max_length = max_length
        self.samples = []
        
        if data_path.endswith('.jsonl'):
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))
        elif data_path.endswith('.md'):
            with open(data_path, 'r', encoding='utf-8') as f:
                content = f.read()
                chunks = [content[i:i+max_length] for i in range(0, len(content), max_length)]
                self.samples = [{"text": chunk} for chunk in chunks if len(chunk) > 10]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample.get("text", str(sample))
        encoded = [ord(c) % 2048 for c in text[:self.max_length]]
        
        while len(encoded) < self.max_length:
            encoded.append(0)
            
        tensor_data = torch.tensor(encoded, dtype=torch.long)
        return {
            "input_ids": tensor_data[:-1],
            "target_ids": tensor_data[1:]
        }

class SovereignTransformer(nn.Module):
    """High-Performance Scaled Causal Transformer for BCI Node Architecture."""
    def __init__(self, vocab_size: int = 2048, d_model: int = 512, nhead: int = 8, num_layers: int = 6, max_len: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        h = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
        
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)
        h = self.transformer_encoder(h, mask=mask, is_causal=True)
        h = self.norm(h)
        return self.fc_out(h)

class BCITrainingPipeline:
    """
    Beacon Computing Initiative (BCI) High-Performance Training Pipeline.
    """
    def __init__(self, model: nn.Module, dataset: Dataset, batch_size: int = 8):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self._validate_execution_preconditions()

    def _validate_execution_preconditions(self) -> None:
        """
        Enforces execution and sequencing if and only if 
        device readiness and model structural integrity are confirmed.
        """
        device_ready = self.device is not None
        model_valid = isinstance(self.model, nn.Module)
        
        if not (device_ready and model_valid):
            raise RuntimeError(
                "BCI Pipeline Sequencing Error: Execution can proceed "
                "if and only if compute device is ready and the model module is properly initialized."
            )

    def execute_training_loop(self, epochs: int = 2) -> None:
        use_pin = True if self.device.type == 'cuda' else False
        dataloader = DataLoader(
            self.dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=2, 
            pin_memory=use_pin
        )
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=3e-4, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))
        
        # Checkpoint management
        os.makedirs("checkpoints", exist_ok=True, mode=0o755)
        checkpoint_path = "checkpoints/bci_sovereign_model.pt"
        
        if os.path.exists(checkpoint_path):
            print(f"Loading existing sovereign checkpoint from {checkpoint_path}...")
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        
        self.model.to(self.device)
        self.model.train()

        print(f"Starting High-Performance Sovereign Training | Device: {self.device} | Epochs: {epochs}")

        for epoch in range(epochs):
            for step, batch in enumerate(dataloader):
                inputs = batch.get("input_ids", torch.tensor([])).to(self.device, non_blocking=True)
                targets = batch.get("target_ids", torch.tensor([])).to(self.device, non_blocking=True)
                
                if inputs.numel() == 0 or targets.numel() == 0:
                    continue
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                
                loss = nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step()
                
                if step > 0 and step % 100 == 0:
                    print(f"Epoch [{epoch+1}/{epochs}] | Step [{step}/{len(dataloader)}] | Loss: {loss.item():.4f}")
                    # Save incremental checkpoint
                    torch.save(self.model.state_dict(), checkpoint_path)

        torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Training complete. Sovereign model weights secured at {checkpoint_path}")

if __name__ == "__main__":
    print("Beacon Computing Initiative (BCI) Sovereign Engine Initializing...")
    
    dataset_file = "dataset/dataset_combined_30k.jsonl"
    if os.path.exists(dataset_file):
        print(f"Ingesting corpus: {dataset_file}...")
        dataset = BCIDataset(dataset_file, max_length=128)
        
        model = SovereignTransformer(vocab_size=2048, d_model=512, nhead=8, num_layers=6, max_len=128)
        
        pipeline = BCITrainingPipeline(model=model, dataset=dataset, batch_size=8)
        pipeline.execute_training_loop(epochs=2)
    else:
        print(f"Dataset {dataset_file} not found.")
