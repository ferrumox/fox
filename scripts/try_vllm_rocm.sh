#!/usr/bin/env bash
# Can vLLM run on this machine at all?
#
# Written as a gated feasibility check rather than a benchmark, because the answer may
# well be "no" and that is a publishable result: an engine that cannot run on the
# hardware is a finding about the hardware, not a gap in the comparison.
#
# The machine is an AMD Radeon 890M — gfx1150, RDNA 3.5, an integrated GPU. vLLM's ROCm
# builds target gfx90a (MI200), gfx942 (MI300) and the discrete RDNA3 parts
# (gfx1100-1102). gfx1150 is not among them, so the kernels it needs may simply not be
# compiled in. HSA_OVERRIDE_GFX_VERSION makes the runtime present the iGPU as a discrete
# RDNA3 card; it works for some workloads and faults in the middle of others, which is
# why each gate below is checked separately instead of jumping straight to serving.
#
#   scripts/try_vllm_rocm.sh
#
# Gate 1 tells you whether ROCm sees the device at all, gate 2 whether vLLM initialises,
# gate 3 whether it serves. Record whichever gate fails — "fails at gate 2 with
# <message>" is a far more useful line in a write-up than "did not work".
set -uo pipefail
IMAGE="${IMAGE:-rocm/vllm:latest}"
OVERRIDE="${OVERRIDE:-11.0.0}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"   # small on purpose: this is a gate, not a benchmark

run() {
  docker run --rm \
    --device=/dev/dri --device=/dev/kfd --group-add video \
    --ipc=host --shm-size=8g \
    -e HSA_OVERRIDE_GFX_VERSION="$1" \
    --entrypoint bash "$IMAGE" -c "$2"
}

echo "=== gate 1: does ROCm/torch see the GPU? ==="
for ov in "" "$OVERRIDE"; do
  label="${ov:-no override}"
  echo "--- HSA_OVERRIDE_GFX_VERSION=${label} ---"
  run "$ov" 'python -c "
import torch
print(\"torch\", torch.__version__)
print(\"available:\", torch.cuda.is_available())
if torch.cuda.is_available():
    print(\"device:\", torch.cuda.get_device_name(0))
    print(\"capability:\", torch.cuda.get_device_capability(0))
"' 2>&1 | tail -6
done

echo
echo "=== gate 2: does vLLM initialise an engine? ==="
run "$OVERRIDE" "python -c \"
from vllm import LLM
llm = LLM(model='$MODEL', max_model_len=512, gpu_memory_utilization=0.6, enforce_eager=True)
print('ENGINE OK')
print(llm.generate(['Hello'])[0].outputs[0].text[:80])
\"" 2>&1 | tail -20

echo
echo "If gate 1 reports available: False under both settings, vLLM cannot run here and"
echo "that is the result to record. If gate 1 passes and gate 2 faults, quote the fault:"
echo "a missing-kernel error names the gfx target and is worth reproducing in the notes."
