# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
"""Regenerate source/documentation hashes; never read dataset or weight contents."""
import hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {'.git', '.venv', '__pycache__', 'dataset', 'datasets', 'models', 'checkpoints', 'runtime'}
if __name__ == '__main__':
    files = sorted(p for p in ROOT.rglob('*') if p.is_file() and not any(part in EXCLUDED for part in p.relative_to(ROOT).parts)
                   and p.name != 'SHA256SUMS' and (not p.name.startswith('.env') or p.name == '.env.example')
                   and p.suffix not in {'.pyc', '.pt', '.bin', '.gguf'})
    body = '# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)\n'
    body += ''.join(hashlib.sha256(p.read_bytes()).hexdigest() + '  ' + p.relative_to(ROOT).as_posix() + '\n' for p in files)
    (ROOT / 'SHA256SUMS').write_text(body)
    print(f'Hashed {len(files)} source/documentation files; data and weights excluded')
