#!/usr/bin/env bash
# build-release-local.sh — Build a CUDA-enabled release tarball locally and
# optionally upload it to a GitHub Release.
#
# Usage:
#   ./scripts/build-release-local.sh [--upload] [--tag v1.0.0]
#
# Prerequisites:
#   - nvcc in PATH (or CUDACXX set)
#   - gh CLI authenticated (only needed with --upload)
#
# Output:
#   dist/fox-<version>-x86_64-unknown-linux-gnu-cuda.tar.gz
#   dist/fox-<version>-x86_64-unknown-linux-gnu-cuda.tar.gz.sha256

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPLOAD=false
TAG=""

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --upload) UPLOAD=true ;;
        --tag)    TAG="$2"; shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
    shift
done

# ── resolve version ───────────────────────────────────────────────────────────
if [[ -z "$TAG" ]]; then
    TAG="$(git -C "$REPO_ROOT" describe --tags --abbrev=0 2>/dev/null || echo "v0.0.0-local")"
fi
VERSION="${TAG#v}"

# ── verify CUDA ───────────────────────────────────────────────────────────────
NVCC="${CUDACXX:-$(command -v nvcc 2>/dev/null || true)}"
if [[ -z "$NVCC" ]]; then
    echo "ERROR: nvcc not found. Install CUDA toolkit or set CUDACXX." >&2
    exit 1
fi
echo "→ CUDA compiler: $NVCC ($(nvcc --version | grep release | awk '{print $5}' | tr -d ','))"

# ── build ─────────────────────────────────────────────────────────────────────
echo "→ Building fox $VERSION with CUDA..."
cd "$REPO_ROOT"
CUDACXX="$NVCC" cargo build --release --bin fox --bin fox-bench

# ── collect artifacts ─────────────────────────────────────────────────────────
RELEASE_DIR="$REPO_ROOT/target/release"
DIST_DIR="$REPO_ROOT/dist"
BUNDLE_NAME="fox-${VERSION}-x86_64-unknown-linux-gnu-cuda"
BUNDLE_DIR="$DIST_DIR/$BUNDLE_NAME"

rm -rf "$BUNDLE_DIR"
mkdir -p "$BUNDLE_DIR"

# Binaries
cp "$RELEASE_DIR/fox"       "$BUNDLE_DIR/"
cp "$RELEASE_DIR/fox-bench" "$BUNDLE_DIR/"

# All backend shared libraries (CUDA + CPU + base)
for so in \
    libggml-base.so \
    libggml-base-native.so \
    libggml.so \
    libggml-cpu.so \
    libggml-cpu-native.so \
    libggml-cuda.so \
    libggml-cuda-native.so; do
    [[ -f "$RELEASE_DIR/$so" ]] && cp "$RELEASE_DIR/$so" "$BUNDLE_DIR/" || true
done

echo "→ Bundle contents:"
ls -lh "$BUNDLE_DIR"

# ── package ───────────────────────────────────────────────────────────────────
TARBALL="$DIST_DIR/${BUNDLE_NAME}.tar.gz"
tar -czf "$TARBALL" -C "$DIST_DIR" "$BUNDLE_NAME"
sha256sum "$TARBALL" > "${TARBALL}.sha256"

echo ""
echo "✓ Tarball:  $TARBALL"
echo "✓ Checksum: ${TARBALL}.sha256"
echo "  $(cat "${TARBALL}.sha256")"

# ── upload ────────────────────────────────────────────────────────────────────
if [[ "$UPLOAD" == true ]]; then
    if ! command -v gh &>/dev/null; then
        echo "ERROR: gh CLI not found. Install it or run without --upload." >&2
        exit 1
    fi

    echo ""
    echo "→ Uploading to GitHub Release $TAG..."
    gh release upload "$TAG" \
        "$TARBALL" \
        "${TARBALL}.sha256" \
        --clobber \
        --repo "$(gh repo view --json nameWithOwner -q .nameWithOwner)"

    echo "✓ Uploaded to release $TAG"
fi
