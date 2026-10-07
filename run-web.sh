#!/usr/bin/env bash
# Собрать интерфейс (если ещё не собран) и запустить мозг: http://localhost:8000
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=python3
[ -x .venv/bin/python ] && PYTHON=.venv/bin/python

if [ ! -d web/dist ]; then
  (cd web && npm install && npm run build)
fi
exec "$PYTHON" -m brain.server
