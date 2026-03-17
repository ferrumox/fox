#!/usr/bin/env bash
# install.sh — Install fox (ferrumox) from GitHub Releases.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/ferrumox/fox/main/install.sh | bash
#   or
#   ./install.sh [--version v0.8.0] [--prefix /usr/local]

set -euo pipefail

REPO="ferrumox/fox"
VERSION="${FOX_VERSION:-latest}"
PREFIX="${FOX_PREFIX:-/usr/local}"
BIN_DIR="$PREFIX/bin"

# Parse arguments.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --version) VERSION="$2"; shift 2 ;;
    --prefix)  PREFIX="$2"; BIN_DIR="$PREFIX/bin"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 1 ;;
  esac
done

# Detect OS and architecture.
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
  Linux)
    case "$ARCH" in
      x86_64)  TARGET="x86_64-unknown-linux-gnu" ;;
      aarch64) TARGET="aarch64-unknown-linux-gnu" ;;
      *)       echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
    esac
    ;;
  Darwin)
    case "$ARCH" in
      x86_64) TARGET="x86_64-apple-darwin" ;;
      arm64)  TARGET="aarch64-apple-darwin" ;;
      *)      echo "Unsupported architecture: $ARCH" >&2; exit 1 ;;
    esac
    ;;
  *)
    echo "Unsupported OS: $OS" >&2
    echo "Please compile from source: cargo build --release --bin fox" >&2
    exit 1
    ;;
esac

# Resolve latest version if needed.
if [[ "$VERSION" == "latest" ]]; then
  VERSION="$(curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" \
    | grep '"tag_name"' | sed 's/.*"tag_name": *"\([^"]*\)".*/\1/')"
fi

VERSION_NUM="${VERSION#v}"
TARBALL="fox-${VERSION_NUM}-${TARGET}.tar.gz"
URL="https://github.com/$REPO/releases/download/$VERSION/$TARBALL"

echo "Installing fox $VERSION for $TARGET …"
echo "Downloading $URL"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

curl -fsSL "$URL" -o "$TMP_DIR/$TARBALL"
tar -xzf "$TMP_DIR/$TARBALL" -C "$TMP_DIR"

mkdir -p "$BIN_DIR"
install -m 755 "$TMP_DIR/fox" "$BIN_DIR/fox"

echo "Installed fox to $BIN_DIR/fox"
echo "Run: fox --help"
