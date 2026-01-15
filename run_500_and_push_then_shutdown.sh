#!/usr/bin/env bash
set -euo pipefail

cd /lambda/nfs/ServerSide/coup

MSG='500 logs'
LOG_DIR="logs"
mkdir -p "$LOG_DIR"
RUN_LOG="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

echo "==> Starting run; logging to $RUN_LOG"
# Run and capture output to a log file (still shows on screen)
set +e
uv run python run_covert_coup.py --games 500 --device cuda --model Qwen/Qwen2.5-72B-Instruct 2>&1 | tee "$RUN_LOG"
RC=${PIPESTATUS[0]}
set -e

echo "==> Run finished with exit code: $RC"

echo "==> Git add/commit/push"
git add -A

# Commit only if there are changes
if ! git diff --cached --quiet; then
  git commit -m "$MSG"
  git push
  echo "==> Pushed changes."
else
  echo "==> No changes to commit."
fi

echo "==> Shutting down server now"
# This powers off the instance. (May require sudo privileges.)
sudo shutdown -h now

