#!/usr/bin/env bash
# Record the README demo GIF. The session is real: fox serves a real GGUF and answers.
#
#   docker build -f Dockerfile.demo -t fox:demo .
#   MODEL=~/.cache/ferrumox/models/some-model.gguf scripts/record_demo.sh
#
# Needs a fox bundle (binary + ggml libraries). `make vulkan` produces one; any build
# works, since the recording runs on CPU inside the container regardless.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${MODEL:-$HOME/.cache/ferrumox/models/llama-3.2-1b-instruct-q8_0.gguf}"
BUNDLE="${BUNDLE:-$ROOT/fox-vulkan}"
[ -x "$BUNDLE/fox" ] || { echo "no fox bundle at $BUNDLE — set BUNDLE= or run 'make vulkan'"; exit 1; }
[ -f "$MODEL" ] || { echo "no model at $MODEL — set MODEL="; exit 1; }

docker run --rm \
  -v "$BUNDLE:/demo" \
  `# Mounted under the name the tape types. Recording with a different model means` \
  `# editing scripts/demo.tape too, or the GIF will label it wrongly.` \
  -v "$MODEL:/demo/llama-3.2-1b.gguf:ro" \
  -v "$ROOT/scripts/demo.tape:/tape/demo.tape:ro" \
  -v "$ROOT/assets:/assets" \
  -w /tape \
  fox:demo /tape/demo.tape

echo "wrote $ROOT/assets/demo.gif"
