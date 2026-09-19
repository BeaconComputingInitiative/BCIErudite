"""
=====================================================================
Developer: Chance Joaquin Boone
Project: BCIai Platform (Computer Beacon Intelligence)
Module: Conversational Universal Document & Dataset Interpreter
Developer Seal: BCI-CUSTOM-LLM-CORE-V7-TORCH
=====================================================================
"""

import os
import json
import pandas as pd
import torch
from source.tokenizer import CustomBPETokenizer
from source.model import ModernLLM

class UniversalDocumentInterpreter:
    """Parses files and formats them into natural conversational context for the LLM."""
    def __init__(self, tokenizer_path: str = "bci_tokenizer.json", checkpoint_path: str = "bci_model.pt"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = CustomBPETokenizer()
        self.tokenizer.load(tokenizer_path)
        self.model = ModernLLM.load_checkpoint(checkpoint_path, device=self.device)
        self.model.eval()

    def parse_document(self, filepath: str, max_chars: int = 1500) -> str:
        if not os.path.exists(filepath):
            return ""

        ext = os.path.splitext(filepath)[1].lower()
        extracted_text = ""

        if ext == '.csv':
            df = pd.read_csv(filepath)
            extracted_text = f"The user uploaded a CSV dataset named {os.path.basename(filepath)} with columns: {', '.join(df.columns.tolist())}. Here is a sample of the data:\n{df.head(5).to_string(index=False)}"

        elif ext in ['.xls', '.xlsx']:
            df = pd.read_excel(filepath)
            extracted_text = f"The user uploaded an Excel spreadsheet named {os.path.basename(filepath)} with columns: {', '.join(df.columns.tolist())}. Here is a sample of the data:\n{df.head(5).to_string(index=False)}"

        elif ext == '.pdf':
            try:
                import pypdf
                reader = pypdf.PdfReader(filepath)
                pages_text = [page.extract_text() for page in reader.pages if page.extract_text()]
                extracted_text = f"The user uploaded a PDF document named {os.path.basename(filepath)} containing:\n" + "\n".join(pages_text)
            except ImportError:
                extracted_text = f"The user uploaded PDF {os.path.basename(filepath)}."

        elif ext == '.docx':
            try:
                import docx
                doc = docx.Document(filepath)
                paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
                extracted_text = f"The user uploaded a Word document named {os.path.basename(filepath)} containing:\n" + "\n".join(paragraphs)
            except ImportError:
                extracted_text = f"The user uploaded Word doc {os.path.basename(filepath)}."

        elif ext == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            extracted_text = f"The user uploaded a JSON file named {os.path.basename(filepath)} with content:\n{json.dumps(data, indent=2)}"

        elif ext in ['.txt', '.md', '.py', '.log', '.sh']:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                extracted_text = f"The user provided a text file named {os.path.basename(filepath)} containing:\n{f.read()}"

        if len(extracted_text) > max_chars:
            extracted_text = extracted_text[:max_chars] + "\n[Content truncated for length...]"

        return extracted_text
