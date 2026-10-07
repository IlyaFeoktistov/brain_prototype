#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=python3
[ -x .venv/bin/python ] && PYTHON=.venv/bin/python

exec "$PYTHON" -m brain.chat
