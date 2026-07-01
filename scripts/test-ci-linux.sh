#!/usr/bin/env bash
# Test the Linux CI steps locally in Docker before pushing a tag.
# Simulates the release workflow for x86_64-unknown-linux-gnu (+ ROCm variant).
#
# Usage:
#   ./scripts/test-ci-linux.sh          # test native build only
#   ./scripts/test-ci-linux.sh --rocm   # also test ROCm dependency install

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

step() { echo -e "\n${YELLOW}▶ $*${NC}"; }
ok()   { echo -e "${GREEN}✓ $*${NC}"; }
fail() { echo -e "${RED}✗ $*${NC}"; exit 1; }

TEST_ROCM=false
for arg in "$@"; do
  [[ "$arg" == "--rocm" ]] && TEST_ROCM=true
done

RUNNER_IMAGE="ubuntu:22.04"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Step 1: build deps ────────────────────────────────────────────────────────
step "Testing: Install build dependencies (Linux native)"
docker run --rm "$RUNNER_IMAGE" bash -c "
  set -e
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -q
  apt-get install -y cmake clang libclang-dev ninja-build
  cmake --version
  clang --version | head -1
" && ok "Build deps install OK" || fail "Build deps install FAILED"

# ── Step 2: ROCm deps (optional) ─────────────────────────────────────────────
if $TEST_ROCM; then
  step "Testing: Install ROCm/HIP SDK"
  docker run --rm "$RUNNER_IMAGE" bash -c "
    set -e
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -q
    apt-get install -y wget gnupg
    wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key \
      | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
    echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] \
      https://repo.radeon.com/rocm/apt/6.2 jammy main' \
      > /etc/apt/sources.list.d/rocm.list
    apt-get update -q
    apt-get install -y \
      rocminfo=1.0.0.60200-66~22.04 \
      rocm-cmake=0.13.0.60200-66~22.04 \
      rocm-device-libs=1.0.0.60200-66~22.04
    apt-get install -y --dry-run rocm-hip-sdk 2>&1 | grep '^Conf rocm-hip-sdk'
  " && ok "ROCm deps install OK" || fail "ROCm deps install FAILED"
fi

# ── Step 3: cargo build (stub mode, no llama.cpp compile) ────────────────────
step "Testing: cargo build --release (stub mode)"
export FOX_SKIP_LLAMA=1
cargo build --release --bin fox 2>&1 | tail -5
ok "cargo build OK"

echo -e "\n${GREEN}All CI checks passed locally.${NC}\n"
