#!/usr/bin/env bash
# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# Push iff this is its own clean git checkout with an attached branch and configured origin.
[[ "$(git rev-parse --show-toplevel)" == "$PWD" ]] || exit 1
[[ -z "$(git status --porcelain)" ]] || { echo "Review and commit the intended changes first" >&2; exit 1; }
branch=$(git symbolic-ref --short HEAD)
git remote get-url origin >/dev/null
git push origin "$branch"
