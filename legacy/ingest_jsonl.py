"""
=====================================================================
Developer: Chance Joaquin Boone
Project: BCIai Platform (Computer Beacon Intelligence)
Module: JSONL Dataset Ingestion & Corpus Compiler
Developer Seal: BCI-CUSTOM-LLM-CORE-V7-TORCH
=====================================================================
"""

import os
import json
import glob

def compile_jsonl_to_corpus():
    dataset_dir = "./dataset"
    corpus_path = "data/corpus.txt"
    os.makedirs("data", exist_ok=True)
    
    jsonl_files = glob.glob(os.path.join(dataset_dir, "*.jsonl"))
    
    # Execution occurs if and only if JSONL files exist in the dataset directory
    if not jsonl_files:
        print("⚡ No .jsonl files found in ./dataset/. Please verify your files were moved correctly.")
        return

    print(f"⚡ Found {len(jsonl_files)} JSONL dataset(s). Compiling into corpus...")
    
    total_records = 0
    with open(corpus_path, "a", encoding="utf-8") as corpus_file:
        for filepath in jsonl_files:
            print(f"⚡ Processing: {os.path.basename(filepath)}")
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        # Extract text depending on standard JSONL schema keys
                        text = ""
                        if isinstance(data, dict):
                            if "text" in data:
                                text = data["text"]
                            elif "content" in data:
                                text = data["content"]
                            elif "messages" in data:
                                text = "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in data["messages"]])
                            else:
                                text = json.dumps(data)
                        
                        if text:
                            corpus_file.write(text + "\n\n")
                            total_records += 1
                    except json.JSONDecodeError:
                        continue

    print(f"⚡ Successfully compiled {total_records:,} records from JSONL files into {corpus_path}!")

if __name__ == "__main__":
    compile_jsonl_to_corpus()
