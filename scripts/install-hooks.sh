#!/usr/bin/env bash
# Install git hooks from scripts/hooks/ into .git/hooks/
# Run once after cloning: make setup  (or bash scripts/install-hooks.sh)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOKS_SRC="$REPO_ROOT/scripts/hooks"
HOOKS_DST="$REPO_ROOT/.git/hooks"

if [ ! -d "$HOOKS_DST" ]; then
    echo "Error: .git/hooks not found — are you in a git repository?"
    exit 1
fi

for hook in "$HOOKS_SRC"/*; do
    name="$(basename "$hook")"
    dst="$HOOKS_DST/$name"
    cp "$hook" "$dst"
    chmod +x "$dst"
    echo "Installed: .git/hooks/$name"
done

echo ""
echo "Done. Hooks will run automatically on git push."
echo "To skip in an emergency: git push --no-verify"
