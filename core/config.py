# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SEAL = "Beacon Computing Initiative | Developer Seal: Chance Joaquin Boone (CJB)"

def read(path):
    # JSON is valid YAML 1.2; config files use YAML filenames and JSON syntax.
    return json.loads(Path(path).read_text())
def path(value): return (ROOT / value).resolve()
def runtime(): return read(ROOT / "config/bci_config.yaml")
def training(): return read(ROOT / "config/training_config.yaml")
