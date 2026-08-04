#!/usr/bin/env bash
# Tag one release and confirm it actually produced something.
#
# Separate from `release.sh` because the tag is the last step, and separate from a bare
# `git push --tags` because that is how ten releases were published and none were built:
# **GitHub fires no workflow at all when more than three tags arrive in one push**. The
# tags existed, the runs did not, and nobody looked. So this pushes exactly one tag and
# then goes and checks that a Release run started and a release object exists.
#
#   scripts/publish.sh 0.21.0
#
# Run it from `main`, on the snapshot commit for that version — that is where this repo
# keeps release tags (`git tag --merged main` shows the rest).
set -euo pipefail

VERSION="${1:-}"
[[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "uso: scripts/publish.sh X.Y.Z" >&2; exit 1; }
cd "$(dirname "$0")/.."

REPO="${FOX_REPO:-ferrumox/fox}"
RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; NC=$'\033[0m'
step() { echo -e "${YELLOW}▶ $*${NC}"; }
ok()   { echo -e "${GREEN}✓ $*${NC}"; }
die()  { echo -e "${RED}✗ $*${NC}" >&2; exit 1; }

step "estamos en main, sobre la instantánea de $VERSION"
[[ "$(git branch --show-current)" == "main" ]] || die "las etiquetas de release viven en main"
SUBJECT=$(git log -1 --format=%s)
[[ "$SUBJECT" == "release: v$VERSION" ]] || die "HEAD es '$SUBJECT', no 'release: v$VERSION'"
ok "$SUBJECT"

# The 0.14-0.18 failure class: a tag whose tree reports a different version.
step "el árbol declara $VERSION"
grep -q "^version = \"$VERSION\"" Cargo.toml || die "Cargo.toml no dice $VERSION"
ok "Cargo.toml"

step "main está empujada"
git fetch origin main -q
[[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/main)" ]] || die "empuja main antes de etiquetar"
ok "sincronizada"

step "empujando SOLO v$VERSION"
git tag "v$VERSION" 2>/dev/null || true
git push origin "v$VERSION"
ok "etiqueta empujada"

# Pushing the tag is not publishing. Verify, because last time nobody did.
step "comprobando que el workflow arrancó"
RUN=""
for _ in $(seq 1 20); do
  sleep 6
  RUN=$(curl -s "https://api.github.com/repos/$REPO/actions/runs?per_page=20" 2>/dev/null \
        | python3 -c "
import json,sys
try: d=json.load(sys.stdin)
except Exception: raise SystemExit
for r in d.get('workflow_runs', []):
    if r['name']=='Release' and r.get('head_branch')=='v$VERSION':
        print(r['status'], r.get('conclusion') or '', r['html_url']); break
" 2>/dev/null) || true
  [[ -n "$RUN" ]] && break
done
if [[ -z "$RUN" ]]; then
  echo -e "${RED}✗ no se ve ninguna ejecución de Release para v$VERSION tras 2 minutos.${NC}"
  echo "  Si empujaste varias etiquetas a la vez, GitHub no dispara ninguna (>3)."
  echo "  Relánzala a mano: workflow_dispatch en release.yml con tag=v$VERSION"
  exit 1
fi
ok "ejecución: $RUN"
echo
echo "Sigue su resultado y confirma que la release existe:"
echo "  https://github.com/$REPO/releases/tag/v$VERSION"
