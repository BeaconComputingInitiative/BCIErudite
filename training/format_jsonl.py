# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
import argparse
import json
from pathlib import Path
from core.config import SEAL
from core.guardrails.validation import record
if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("input"); parser.add_argument("output"); args = parser.parse_args()
    rows = [record(json.loads(line)) for line in Path(args.input).read_text().splitlines() if line.strip()]
    with Path(args.output).open("x") as out:
        for row in rows: out.write(json.dumps({"header": SEAL, **row}, ensure_ascii=False) + "\n")
