#!/usr/bin/env bash
# Beacon Computing Initiative | Erudite | Developer Seal: Chance Joaquin Boone (CJB)
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
# Install iff systemd exists, venv exists, the path is safely representable, and no old unit would be overwritten.
command -v systemctl >/dev/null && [[ -x .venv/bin/python ]] || exit 1
[[ "$PWD" != *[[:space:]]* && "$PWD" != *%* && "$PWD" != *\"* && "$PWD" != *\\* ]] || { echo "Use a path without spaces, percent signs, quotes, or backslashes" >&2; exit 1; }
mkdir -p "$HOME/.config/systemd/user"
unit="$HOME/.config/systemd/user/erudite-api.service"
[[ ! -e "$unit" ]] || { echo "Existing service preserved; review it first" >&2; exit 1; }
cat > "$unit" <<UNIT
# Beacon Computing Initiative | Developer Seal: Chance Joaquin Boone (CJB)
[Unit]
Description=Erudite local language model API
After=network.target
[Service]
WorkingDirectory=$PWD
ExecStart=$PWD/.venv/bin/python $PWD/main.py
Restart=on-failure
RestartSec=5
NoNewPrivileges=true
UMask=0077
[Install]
WantedBy=default.target
UNIT
sha256sum "$unit" > "$unit.sha256"
systemctl --user daemon-reload
systemctl --user enable --now erudite-api.service
