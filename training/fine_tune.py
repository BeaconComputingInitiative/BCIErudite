# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
"""Continue Erudite's own weights; no LoRA, pretrained base, or tokenizer substitution."""
import argparse
from training.train_model import train
if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--checkpoint", required=True); parser.add_argument("--config", required=True)
    args = parser.parse_args(); print(train(args.config, args.checkpoint))
