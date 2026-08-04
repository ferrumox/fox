#!/usr/bin/env bash
# install.sh — Install fox (ferrumox) from GitHub Releases.
#
#   curl -fsSL https://github.com/ferrumox/fox/releases/latest/download/install.sh | sh
#   ./install.sh [--version v0.20.3] [--prefix ~/.local] [--vulkan|--cpu]
#
# Three things this refuses to do, each because the previous version did:
#
#   - Offer platforms that are not published. It mapped macOS and aarch64 to release
#     filenames no workflow builds, so those users got a bare 404 from curl instead of
#     being told no build exists for them.
#   - Ignore the checksum. Every tarball ships a `.sha256` beside it and nothing read it.
#   - Silently install the CPU build on a machine with a GPU. The release carries a
#     Vulkan variant — the one that matters on the AMD/Intel iGPUs fox targets — and the
#     installer did not know it existed, so that build was effectively unreachable.
set -euo pipefail

REPO="${FOX_REPO:-ferrumox/fox}"
VERSION="${FOX_VERSION:-latest}"
PREFIX="${FOX_PREFIX:-/usr/local}"
BIN_DIR="$PREFIX/bin"
VARIANT="${FOX_VARIANT:-auto}"   # auto | vulkan | cpu

while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="$2"; shift 2 ;;
    --prefix)  PREFIX="$2"; BIN_DIR="$PREFIX/bin"; shift 2 ;;
    --vulkan)  VARIANT="vulkan"; shift ;;
    --cpu)     VARIANT="cpu"; shift ;;
    -h|--help) sed -n '2,5p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

die() { echo "error: $*" >&2; exit 1; }

# ── Platform ────────────────────────────────────────────────────────────────────
# Only what the release workflow actually builds. Adding a row here without adding a
# matrix entry there is how a user ends up staring at a 404.
OS="$(uname -s)"; ARCH="$(uname -m)"
if [[ "$OS" != "Linux" || "$ARCH" != "x86_64" ]]; then
  cat >&2 <<EOF
error: no published build for $OS/$ARCH.

fox publishes binaries for Linux x86_64 only. Elsewhere, build it — one command once
Rust and a C toolchain are present:

  git clone --recurse-submodules https://github.com/$REPO
  cd fox && cargo build --release --bin fox

(The submodule matters: llama.cpp is vendored, not a system dependency.)
EOF
  exit 1
fi
TARGET="x86_64-unknown-linux-gnu"

# ── Variant ─────────────────────────────────────────────────────────────────────
# The Vulkan build runs on AMD/Intel iGPUs and any Vulkan-capable GPU, and falls back to
# CPU by itself when there is no device — so where a render node exists it is the better
# default, not a gamble.
if [[ "$VARIANT" == "auto" ]]; then
  if compgen -G "/dev/dri/renderD*" >/dev/null 2>&1; then
    VARIANT="vulkan"
    echo "GPU detected (/dev/dri) — installing the Vulkan build. Override with --cpu."
  else
    VARIANT="cpu"
    echo "No GPU device found — installing the CPU build. Override with --vulkan."
  fi
fi
SUFFIX=""; [[ "$VARIANT" == "vulkan" ]] && SUFFIX="-vulkan"

# ── Version ─────────────────────────────────────────────────────────────────────
if [[ "$VERSION" == "latest" ]]; then
  VERSION="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" 2>/dev/null \
    | grep '"tag_name"' | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')" || true
  [[ -n "$VERSION" ]] || die "could not resolve the latest release. Pass --version vX.Y.Z."
fi
VERSION_NUM="${VERSION#v}"
TARBALL="fox-${VERSION_NUM}-${TARGET}${SUFFIX}.tar.gz"
BASE="https://github.com/$REPO/releases/download/$VERSION"

echo "Installing fox $VERSION ($VARIANT) into $BIN_DIR"

# ── Writability, before downloading anything ────────────────────────────────────
# Failing here beats failing after the download with a raw permission error.
# Walk up to the first ancestor that exists — `--prefix ~/.local/foo` is legitimate
# even when neither `foo` nor `bin` are there yet, and testing only the immediate
# parent reported "not writable" for a path it was perfectly able to create.
probe="$BIN_DIR"
while [[ ! -e "$probe" && "$probe" != "/" && "$probe" != "." ]]; do
  probe="$(dirname "$probe")"
done
if [[ ! -w "$probe" ]]; then
  die "$BIN_DIR is not writable. Re-run with sudo, or install somewhere you own:
    ./install.sh --prefix \$HOME/.local"
fi

TMP_DIR="$(mktemp -d)"; trap 'rm -rf "$TMP_DIR"' EXIT

if ! curl -fsSL "$BASE/$TARBALL" -o "$TMP_DIR/$TARBALL"; then
  die "release $VERSION has no asset named $TARBALL.
    See what it did publish: https://github.com/$REPO/releases/tag/$VERSION"
fi

# ── Checksum ────────────────────────────────────────────────────────────────────
if curl -fsSL "$BASE/${TARBALL}.sha256" -o "$TMP_DIR/${TARBALL}.sha256" 2>/dev/null &&
   command -v sha256sum >/dev/null 2>&1; then
  EXPECTED="$(awk '{print $1}' "$TMP_DIR/${TARBALL}.sha256")"
  ACTUAL="$(sha256sum "$TMP_DIR/$TARBALL" | awk '{print $1}')"
  [[ "$EXPECTED" == "$ACTUAL" ]] || die "checksum mismatch — expected $EXPECTED, got $ACTUAL"
  echo "Checksum verified."
else
  echo "warning: could not verify the checksum for this asset." >&2
fi

tar -xzf "$TMP_DIR/$TARBALL" -C "$TMP_DIR"
mkdir -p "$BIN_DIR"
install -m 755 "$TMP_DIR/fox" "$BIN_DIR/fox"
[[ -f "$TMP_DIR/fox-bench" ]] && install -m 755 "$TMP_DIR/fox-bench" "$BIN_DIR/fox-bench"

# Backend .so files go beside the binary: fox is linked with RPATH=$ORIGIN, so that is
# the only place it looks for them.
shopt -s nullglob
for lib in "$TMP_DIR"/*.so*; do install -m 755 "$lib" "$BIN_DIR/"; done
shopt -u nullglob

# ── What now ────────────────────────────────────────────────────────────────────
echo
echo "Installed: $("$BIN_DIR/fox" --version 2>/dev/null || echo "$BIN_DIR/fox")"
case ":$PATH:" in
  *":$BIN_DIR:"*) ;;
  *) echo
     echo "note: $BIN_DIR is not on your PATH. Add it:"
     echo "  export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac
cat <<EOF

Next:
  fox pull llama3.2        # a small model to start with (2.0 GB)
  fox serve                # OpenAI- and Ollama-compatible, on :8080
  fox models               # the built-in catalogue
EOF
