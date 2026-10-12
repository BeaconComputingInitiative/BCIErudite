#!/usr/bin/env bash
# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
[[ -x .venv/bin/python ]] || { echo "Create .venv first" >&2; exit 1; }
exec .venv/bin/python -m training.train_model "$@"
