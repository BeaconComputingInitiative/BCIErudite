# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
import hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    failed=[]
    for line in (ROOT/'SHA256SUMS').read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        expected,name=line.split('  ',1);candidate=(ROOT/name).resolve()
        if not candidate.is_relative_to(ROOT) or not candidate.is_file() or hashlib.sha256(candidate.read_bytes()).hexdigest()!=expected:failed.append(name)
    print('Integrity OK' if not failed else 'Changed or missing: '+', '.join(failed))
    sys.exit(bool(failed))
