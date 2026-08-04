#!/usr/bin/env bash
# Prepare a release, refusing to continue when anything does not line up.
#
# Written after an afternoon in which the release process failed six different ways:
# tags created before the work was finished and moved three times, tags put on the wrong
# branch, a merge that produced a two-parent commit containing none of the branch it
# claimed to merge, a code commit landing on `main`, ten tags pushed at once (GitHub
# silently fires no workflow past three, so nothing was built), and a pre-push check
# waved through with an environment variable.
#
# And one older failure that none of those caused: versions 0.14 through 0.18 were
# published with `Cargo.toml` still saying `0.11.0`. A binary built from those tags
# reported the wrong version and nobody noticed for six releases, because nothing sat
# between "I wrote a CHANGELOG entry" and "this is a release".
#
# So this script does not trust anyone to remember. Every check below exists because
# skipping it already cost something.
#
#   scripts/release.sh 0.21.0
#
# It stops at the release commit. Tagging and pushing are `scripts/publish.sh`, kept
# separate on purpose: the tag is the last step, never the first.
set -euo pipefail

VERSION="${1:-}"
if [[ ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "uso: scripts/release.sh X.Y.Z" >&2
  exit 1
fi
cd "$(dirname "$0")/.."

RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[1;33m'; NC=$'\033[0m'
step() { echo -e "${YELLOW}▶ $*${NC}"; }
ok()   { echo -e "${GREEN}✓ $*${NC}"; }
die()  { echo -e "${RED}✗ $*${NC}" >&2; exit 1; }

# 1. Clean tree. `vendor/llama.cpp` is allowed to be dirty — it is a submodule that
#    tracks a pinned commit and is routinely left modified by builds.
step "árbol limpio"
# git marca un submódulo modificado como ' m' o ' M' según qué haya cambiado dentro.
DIRTY=$(git status --porcelain | grep -vE '^ [mM] vendor/llama\.cpp$' || true)
[[ -z "$DIRTY" ]] || { echo "$DIRTY"; die "hay cambios sin comitear"; }
ok "limpio"

# 2. Not on main. `main` holds only release snapshots; a release is *prepared* on the
#    working branch. A code commit landed on main exactly once, by doing this wrong.
step "rama de trabajo"
BRANCH=$(git branch --show-current)
[[ "$BRANCH" != "main" ]] || die "estás en main: la release se prepara en la rama de trabajo"
ok "$BRANCH"

# 3. The version must not already exist as a tag, anywhere.
step "la etiqueta v$VERSION no existe todavía"
! git rev-parse -q --verify "refs/tags/v$VERSION" >/dev/null || die "v$VERSION ya existe en local"
if git ls-remote --tags origin "v$VERSION" 2>/dev/null | grep -q .; then
  die "v$VERSION ya existe en el remoto"
fi
ok "libre"

# 4. The CHANGELOG must already describe this version, and describe *something*.
#    An entry that is only a heading is how a release ships with no notes.
step "entrada de CHANGELOG para $VERSION"
grep -q "^## \[$VERSION\]" CHANGELOG.md || die "CHANGELOG.md no tiene '## [$VERSION]'"
BODY=$(awk -v v="^## \\\\[$VERSION\\\\]" '$0 ~ v {f=1; next} /^## \[/ {f=0} f' CHANGELOG.md | grep -c '[^[:space:]]' || true)
(( BODY >= 3 )) || die "la entrada de $VERSION tiene $BODY líneas con contenido — escríbela antes"
ok "$BODY líneas"

# 5. The gates. `make ci` includes the real-llama.cpp check; `make e2e` is the one that
#    sees cross-request prefix-cache lifecycle bugs, which unit and golden tests do not.
step "make ci"
make ci >/dev/null || die "make ci falló"
ok "ci"

E2E_MODEL="${E2E_MODEL:-$HOME/.cache/ferrumox/models/llama-3.2-1b-instruct-q8_0.gguf}"
if [[ -f "$E2E_MODEL" ]]; then
  step "make e2e"
  make e2e E2E_MODEL="$E2E_MODEL" >/tmp/fox-e2e.log 2>&1 || { tail -20 /tmp/fox-e2e.log; die "make e2e falló"; }
  grep -q "0 failed" /tmp/fox-e2e.log || { tail -20 /tmp/fox-e2e.log; die "make e2e no reportó 0 fallos"; }
  ok "e2e ($(grep -o 'RESULT: [0-9]* passed' /tmp/fox-e2e.log | tail -1))"
else
  die "no encuentro el modelo de e2e en $E2E_MODEL — pásalo con E2E_MODEL=..."
fi

# 6. The bump itself, in every place the version is written. This is the check that
#    would have caught 0.14-0.18 shipping as 0.11.0.
step "subiendo la versión a $VERSION"
VERSION="$VERSION" python3 - <<'PY'
import os, pathlib, re
v = os.environ['VERSION']
edits = [
    ('Cargo.toml', r'^version = "[^"]+"', f'version = "{v}"', True),
    ('Cargo.lock', r'(name = "ferrumox"\nversion = )"[^"]+"', rf'\g<1>"{v}"', False),
    ('README.md', r'version-\d+\.\d+\.\d+-green', f'version-{v}-green', False),
    ('docs/introduction.md', r'The current release is \*\*v\d+\.\d+\.\d+\*\*\.', f'The current release is **v{v}**.', False),
]
for path, pat, rep, multiline in edits:
    p = pathlib.Path(path)
    if not p.exists():
        continue
    s = p.read_text()
    n, k = re.subn(pat, rep, s, count=1, flags=re.M if multiline else 0)
    if k:
        p.write_text(n)
        print(f"   {path}")
PY

# 7. And verify it took, in every file, before committing. Editing and hoping is what
#    the whole script exists to replace.
step "la versión coincide en todas partes"
MISMATCH=""
grep -q "^version = \"$VERSION\"" Cargo.toml || MISMATCH+=" Cargo.toml"
grep -A1 'name = "ferrumox"' Cargo.lock | grep -q "version = \"$VERSION\"" || MISMATCH+=" Cargo.lock"
grep -q "version-$VERSION-green" README.md 2>/dev/null || MISMATCH+=" README.md"
[[ -z "$MISMATCH" ]] || die "la versión no cuadra en:$MISMATCH"
ok "Cargo.toml, Cargo.lock, README"

git add -A
git commit -q -m "release: $VERSION"
ok "commit $(git log --oneline -1)"
echo
echo "Siguiente paso, cuando esté fusionado a develop y fotografiado en main:"
echo "  scripts/publish.sh $VERSION"
