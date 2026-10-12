"""
=====================================================================
Developer: Chance Joaquin Boone
Project: BCIai Platform (Computer Beacon Intelligence)
Module: Corpus Preparation & Ingestion Module
Developer Seal: BCI-CUSTOM-LLM-CORE-V7-TORCH
=====================================================================
"""

import os
import urllib.request

def download_and_extract_corpus():
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)
    corpus_file = os.path.join(data_dir, "corpus.txt")
    
    # Download sequence executes if and only if corpus.txt is not present.
    if not os.path.exists(corpus_file):
        print("⚡ Downloading WikiText-103 training split from Hugging Face mirror...")
        url = "https://huggingface.co/datasets/Salesforce/wikitext/raw/main/wikitext-103-raw/wiki.train.raw"
        
        try:
            urllib.request.urlretrieve(url, corpus_file)
            print(f"⚡ Dataset successfully saved to {corpus_file}")
        except Exception as e:
            print(f"⚡ Direct download failed: {e}. Creating local fallback corpus...")
            fallback_text = (
                "The BCIai cognitive framework operates on decentralized intelligence nodes. "
                "Chance Joaquin Boone designed this system to maintain high resilience, bi-conditional workflow execution, "
                "and secure tensor processing. Networks utilize secure mesh topology, custom tokenizers, and attention routing. "
            ) * 50000  # Generates ~15MB local training text
            with open(corpus_file, "w", encoding="utf-8") as f:
                f.write(fallback_text)
            print(f"⚡ Local fallback corpus created successfully at {corpus_file}")
    else:
        print(f"⚡ Corpus already exists at {corpus_file}")

if __name__ == "__main__":
    download_and_extract_corpus()
