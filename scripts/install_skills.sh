#!/usr/bin/env bash
# Refresh the vendored skills from agiprolabs/claude-trading-skills (MIT).
# Usage: scripts/install_skills.sh [extra-skill ...]
set -euo pipefail
SKILLS=(market-microstructure-traditional slippage-modeling risk-management
        walk-forward-validation trade-journal ohlcv-processing position-sizing "$@")
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git clone --depth 1 https://github.com/agiprolabs/claude-trading-skills.git "$TMP/repo"
mkdir -p "$ROOT/.claude/skills"
for s in "${SKILLS[@]}"; do
  rm -rf "$ROOT/.claude/skills/$s"
  cp -r "$TMP/repo/skills/$s" "$ROOT/.claude/skills/$s"
  echo "installed $s"
done
cp "$TMP/repo/LICENSE.md" "$ROOT/.claude/skills/THIRD_PARTY_LICENSE.md"
