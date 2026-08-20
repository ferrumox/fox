#!/usr/bin/env bash
# Probe how far rustc's GPU offload backend gets on this host.
#
# Re-run this whenever the nightly toolchain moves: the point of the script is
# that the answer changes over time, and "it does not work yet" is only useful
# if it is cheap to re-check. Findings as of the last run live in
# docs/design/rust-gpu-offload.md.
#
# Usage: scripts/probe_rust_offload.sh [gfx1100|sm_120]
set -uo pipefail

ARCH="${1:-gfx1100}"
case "$ARCH" in
  gfx*) TARGET=amdgcn-amd-amdhsa ;;
  sm_*) TARGET=nvptx64-nvidia-cuda ;;
  *) echo "unknown arch $ARCH (expected gfx* or sm_*)" >&2; exit 2 ;;
esac

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
TOOLCHAIN_LIB="$(rustc +nightly --print sysroot)/lib/rustlib/x86_64-unknown-linux-gnu/lib"

step() { printf '\n== %s\n' "$1"; }
ok()   { printf '   PASS  %s\n' "$1"; }
bad()  { printf '   FAIL  %s\n' "$1"; }

step "toolchain"
rustc +nightly --version || { bad "no nightly toolchain"; exit 1; }
if rustc +nightly -Z help 2>/dev/null | grep -q 'offload=val'; then
  ok "-Zoffload present"
else
  bad "-Zoffload absent — nightly predates the offload frontend"; exit 1
fi
if [ -f "$TOOLCHAIN_LIB/libRustOffload-23.so" ]; then
  ok "offload component installed"
else
  bad "offload component missing — run: rustup component add offload --toolchain nightly"; exit 1
fi

step "scaffold a one-kernel crate in $WORK"
mkdir -p "$WORK/src"
cat > "$WORK/Cargo.toml" <<'EOF'
[package]
name = "offload_probe"
version = "0.1.0"
edition = "2024"

[profile.release]
lto = "fat"
EOF
# `no_std` on the GPU targets only: those targets have no std, but the host-side
# stub that #[offload_kernel] generates does not build without it.
cat > "$WORK/src/lib.rs" <<'EOF'
#![cfg_attr(any(target_arch = "amdgpu", target_arch = "nvptx64"), no_std)]
#![feature(gpu_offload, abi_gpu_kernel)]
#![cfg_attr(target_arch = "amdgpu", feature(stdarch_amdgpu))]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx))]

// The thread-index API is NOT portable: different names, different safety, and
// AMD has no workgroup-size query at all. Any portable kernel needs this shim.
#[cfg(target_arch = "amdgpu")]
#[inline(always)]
fn gid_x() -> u32 {
    use core::arch::amdgpu::{workgroup_id_x, workitem_id_x};
    workitem_id_x() + workgroup_id_x() * 256
}

#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn gid_x() -> u32 {
    use core::arch::nvptx::{_block_dim_x, _block_idx_x, _thread_idx_x};
    unsafe { _thread_idx_x() + _block_idx_x() * _block_dim_x() }
}

#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
#[inline(always)]
fn gid_x() -> u32 {
    0
}

#[core::offload::offload_kernel]
fn fill(x: &mut [f32; 256]) {
    let i = gid_x() as usize;
    if i < x.len() {
        x[i] = i as f32 * 2.0;
    }
}

pub fn run(x: &mut [f32; 256]) {
    core::offload::offload! {
        kernel = fill,
        workgroup_dim = [1, 1, 1],
        thread_dim = [256, 1, 1],
        args = (x,),
    }
}
EOF
cat > "$WORK/src/main.rs" <<'EOF'
fn main() {
    let mut x = [0.0f32; 256];
    offload_probe::run(&mut x);
    // 0, 84, 510 if the kernel ran; all zeros if it silently did nothing.
    println!("x[0]={} x[42]={} x[255]={}", x[0], x[42], x[255]);
}
EOF

cd "$WORK" || exit 1

step "pass 1/3 — host metadata"
if RUSTFLAGS="-Zunstable-options -Zoffload=HostMetadata=$WORK/meta.bin" \
     cargo +nightly build --release --lib >/dev/null 2>&1 && [ -s meta.bin ]; then
  ok "manifest written ($(wc -c < meta.bin) bytes)"
else
  bad "host metadata pass"; exit 1
fi

step "pass 2/3 — device codegen for $ARCH"
if RUSTFLAGS="-Zunstable-options -Zoffload=Device=$WORK/meta.bin -Ctarget-cpu=$ARCH" \
     cargo +nightly build --release --lib -Zbuild-std=core --target "$TARGET" >/dev/null 2>&1; then
  DEV="$(ls -t target/"$TARGET"/release/build/offload_probe/*/out/device.bin 2>/dev/null | head -1)"
  [ -n "$DEV" ] && ok "device image: $(basename "$DEV") ($(wc -c < "$DEV") bytes)" || { bad "no device.bin"; exit 1; }
else
  bad "device pass"; exit 1
fi

step "pass 3/3 — host link"
if RUSTFLAGS="-Zunstable-options -Zoffload=Host=$(realpath "$DEV") -L native=$TOOLCHAIN_LIB -Clink-arg=-lomptarget -Clink-arg=-Wl,-rpath,$TOOLCHAIN_LIB" \
     cargo +nightly build --release >/dev/null 2>&1; then
  ok "linked"
else
  bad "host link"; exit 1
fi

step "is the device image actually embedded?"
# The runtime calls get emitted either way; without the image, __tgt_register_lib
# registers nothing and the kernel launch fails on pointer mapping. Distinguishing
# the two is the whole point of this check.
if readelf -SW target/release/offload_probe 2>/dev/null | grep -qi 'llvm.offloading\|omp_offloading'; then
  ok "offload section present"
else
  bad "NO offload section — the image was not embedded (clang-linker-wrapper is not shipped by the offload component)"
fi

step "run"
HSA_OVERRIDE_GFX_VERSION=11.0.0 LIBOMPTARGET_INFO=1 timeout 90 ./target/release/offload_probe 2>&1 | head -12
