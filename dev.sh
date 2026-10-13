#!/usr/bin/env bash
# Starts backend and frontend together; Ctrl-C stops both.
set -euo pipefail

cd "$(dirname "$0")"

BACKEND_PORT="${AGENTBOX_PORT:-8080}"

if [[ ! -x backend/.venv/bin/python ]]; then
    echo "Нет backend/.venv. Выполните:" >&2
    echo "  python3 -m venv backend/.venv" >&2
    echo "  backend/.venv/bin/pip install -e 'backend[dev]'" >&2
    exit 1
fi

if [[ ! -d frontend/node_modules ]]; then
    echo "Нет frontend/node_modules. Выполните: npm --prefix frontend install" >&2
    exit 1
fi

pids=()
cleanup() {
    trap - INT TERM EXIT
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# --reload-dir keeps the watcher off node_modules and the SQLite WAL files,
# which would otherwise restart the backend on every write.
backend/.venv/bin/python -m uvicorn agentbox.api.app:app \
    --app-dir backend --host 127.0.0.1 --port "$BACKEND_PORT" \
    --reload --reload-dir backend/agentbox &
pids+=($!)

npm --prefix frontend run dev &
pids+=($!)

echo "backend  http://127.0.0.1:${BACKEND_PORT}/api/health"
echo "frontend http://localhost:5173"

# macOS ships bash 3.2, which has no `wait -n`; poll instead so that one
# process dying takes the other down with it.
while :; do
    for pid in "${pids[@]}"; do
        kill -0 "$pid" 2>/dev/null || exit 1
    done
    sleep 1
done
