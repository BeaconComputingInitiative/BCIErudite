"""
=====================================================================
Developer: Chance Joaquin Boone
Project: BCIai Platform (Computer Beacon Intelligence)
Module: Sequence Window Dataset Pipeline
Developer Seal: BCI-CUSTOM-LLM-CORE-V7-TORCH
=====================================================================
"""

import torch
from torch.utils.data import Dataset

class TextSequenceDataset(Dataset):
    """Dataset handler mapping raw integer tokens into sliding input-target windows."""
    def __init__(self, tokens: torch.Tensor, block_size: int):
        self.tokens = tokens
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.tokens) - self.block_size)

    def __getitem__(self, idx: int):
        # Window slicing proceeds if and only if the requested index lies within boundary constraints
        x = self.tokens[idx : idx + self.block_size]
        y = self.tokens[idx + 1 : idx + self.block_size + 1]
        return x, y
