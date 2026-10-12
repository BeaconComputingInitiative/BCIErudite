#!/usr/bin/env bash
# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# Launch iff the project interpreter exists.
[[ -x .venv/bin/python ]] || { echo "Create .venv and install requirements first" >&2; exit 1; }
exec .venv/bin/python -m streamlit run ui/app.py --server.address 127.0.0.1 --server.port 8501
