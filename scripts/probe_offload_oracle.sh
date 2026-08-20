#!/usr/bin/env bash
# Differential test: run a `fox-offload` kernel on the GPU and on the CPU oracle,
# from the same source, and compare every thread's partial.
#
# This is the test the offload paper does not do. It compares kernel *performance*
# against RAJA and CUDA; nothing there establishes that a Rust kernel computes what
# the code it replaces computed. Here the same `BlockArgmax::thread` is compiled
# twice — x86_64 and the device — and all N partials are compared, not just the
# winner.
#
# Usage: scripts/probe_offload_oracle.sh [gfx1100|sm_120]
#
# Needs three things the toolchain does not ship. Each one is checked below and
# says what to do if it is missing:
#
#   FOX_OFFLOAD_WRAPPER   an LLVM 23 clang-linker-wrapper (rustup ships none)
#   ld.lld on PATH        the wrapper links the device image with it
#   FOX_HSA_LIB           a directory holding a modern libhsa-runtime64
#
# That last one is the trap. Ubuntu's libhsa-runtime64 is 1.11 (April 2024) and does
# not recognise gfx1150: it enumerates zero GPUs, libomptarget builds no translation
# table, and the launch dies with "does not have a matching target pointer" — which
# reads like a pointer bug and means "no GPU". See docs/design/rustc-offload-fix-results.md.
set -uo pipefail

ARCH="${1:-gfx1100}"
case "$ARCH" in
  gfx*) TARGET=amdgcn-amd-amdhsa ;;
  sm_*) TARGET=nvptx64-nvidia-cuda ;;
  *) echo "unknown arch $ARCH (expected gfx* or sm_*)" >&2; exit 2 ;;
esac

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRATE="$REPO/crates/fox-offload"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
TOOLCHAIN_LIB="$(rustc +nightly --print sysroot)/lib/rustlib/x86_64-unknown-linux-gnu/lib"

step() { printf '\n== %s\n' "$1"; }
ok()   { printf '   PASS  %s\n' "$1"; }
bad()  { printf '   FAIL  %s\n' "$1"; }
skip() { printf '   SKIP  %s\n' "$1"; }

step "prerequisites"
rustc +nightly --version >/dev/null 2>&1 || { bad "no nightly toolchain"; exit 1; }
[ -d "$CRATE" ] || { bad "crates/fox-offload not found at $CRATE"; exit 1; }
ok "nightly and crate present"

if [ -n "${FOX_OFFLOAD_WRAPPER:-}" ] && [ -x "$FOX_OFFLOAD_WRAPPER" ]; then
  ok "clang-linker-wrapper: $FOX_OFFLOAD_WRAPPER"
else
  bad "set FOX_OFFLOAD_WRAPPER to an LLVM 23 clang-linker-wrapper"
  printf '         rustup ships libraries but no tools. Either build rustc from source\n'
  printf '         with --enable-llvm-offload --enable-clang, or extract apt.llvm.org'"'"'s\n'
  printf '         llvm-toolchain-noble-23 with dpkg-deb -x (no root needed).\n'
  exit 1
fi

command -v ld.lld >/dev/null 2>&1 && ok "ld.lld on PATH" || {
  bad "ld.lld not on PATH — the wrapper needs it to link the device image"
  printf '         a symlink to the toolchain'"'"'s rust-lld works:\n'
  printf '         ln -s $(rustc +nightly --print sysroot)/lib/rustlib/x86_64-unknown-linux-gnu/bin/rust-lld somewhere/ld.lld\n'
  exit 1
}

# The device-side OpenMP runtime. Ships with the rustup `offload` component.
OMPDEVICE="$(ls -d "$TOOLCHAIN_LIB/$TARGET" 2>/dev/null | head -1)"
if [ -n "$OMPDEVICE" ] && [ -f "$OMPDEVICE/libompdevice.a" ]; then
  ok "libompdevice.a for $TARGET"
elif [ -f "$TOOLCHAIN_LIB/libompdevice.a" ]; then
  OMPDEVICE="$TOOLCHAIN_LIB"; ok "libompdevice.a (flat layout)"
else
  bad "no libompdevice.a — run: rustup component add offload --toolchain nightly"; exit 1
fi

RUNLIBS="$TOOLCHAIN_LIB"
if [ -n "${FOX_HSA_LIB:-}" ] && [ -e "$FOX_HSA_LIB/libhsa-runtime64.so" ]; then
  RUNLIBS="$FOX_HSA_LIB:$RUNLIBS"; ok "HSA runtime: $FOX_HSA_LIB"
else
  skip "FOX_HSA_LIB unset — using the system libhsa-runtime64"
  printf '         if the run below reports zero devices, that is why. Ubuntu ships 1.11\n'
  printf '         (2024), which does not know gfx1150. Extract a modern one without root:\n'
  printf '         curl -sL -o hsa.deb https://repo.radeon.com/rocm/apt/latest/pool/main/h/hsa-rocr/hsa-rocr_1.18.0.70204-93~24.04_amd64.deb\n'
  printf '         curl -sL -o rpr.deb https://repo.radeon.com/rocm/apt/latest/pool/main/r/rocprofiler-register/rocprofiler-register_0.6.0.70204-93~24.04_amd64.deb\n'
  printf '         dpkg-deb -x hsa.deb r/ ; dpkg-deb -x rpr.deb r/ ; export FOX_HSA_LIB=$PWD/r/opt/rocm-7.2.4/lib\n'
fi

step "scaffold the harness in $WORK"
mkdir -p "$WORK/src"
cat > "$WORK/Cargo.toml" <<EOF
[package]
name = "oracle"
version = "0.1.0"
edition = "2021"

[dependencies]
fox-offload = { path = "$CRATE" }

[profile.release]
lto = "fat"
EOF

# The kernel. Note what is NOT here: no reimplementation. The device kernel calls
# straight into the crate's own BlockArgmax::thread, which is also what launch_cpu
# runs. That identity is the whole point of the test.
cat > "$WORK/src/lib.rs" <<'EOF'
#![cfg_attr(any(target_arch = "amdgpu", target_arch = "nvptx64"), no_std)]
#![feature(gpu_offload, abi_gpu_kernel)]
#![cfg_attr(target_arch = "amdgpu", feature(stdarch_amdgpu))]
#![cfg_attr(target_arch = "nvptx64", feature(stdarch_nvptx))]

use fox_offload::kernels::argmax::{ArgmaxArgs, BlockArgmax, Candidate};
use fox_offload::{GridDim, Kernel, ThreadId};

pub const NGROUPS: u32 = 4;
pub const NTHREADS: u32 = 64;
pub const N: usize = 256;

#[cfg(target_arch = "amdgpu")]
#[inline(always)]
fn ids() -> (u32, u32) {
    use core::arch::amdgpu::{workgroup_id_x, workitem_id_x};
    (workitem_id_x(), workgroup_id_x())
}
#[cfg(target_arch = "nvptx64")]
#[inline(always)]
fn ids() -> (u32, u32) {
    use core::arch::nvptx::{_block_idx_x, _thread_idx_x};
    // SAFETY: the cfg guarantees nvptx64; these read the launch configuration.
    unsafe { (_thread_idx_x(), _block_idx_x()) }
}
#[cfg(not(any(target_arch = "amdgpu", target_arch = "nvptx64")))]
#[inline(always)]
fn ids() -> (u32, u32) {
    (0, 0)
}

#[core::offload::offload_kernel]
fn argmax_partials(logits: &mut [f32; N], partials: &mut [Candidate; N]) {
    let (t, w) = ids();
    let tid = ThreadId::new([t, 0, 0], [w, 0, 0]);
    let grid = GridDim::linear(NGROUPS, NTHREADS);
    let mut args = ArgmaxArgs {
        logits: &logits[..],
        partials: &mut partials[..],
    };
    BlockArgmax::thread(&mut args, tid, &grid);
}

pub fn run_gpu(logits: &mut [f32; N], partials: &mut [Candidate; N]) {
    core::offload::offload! {
        kernel = argmax_partials,
        workgroup_dim = [NGROUPS, 1, 1],
        thread_dim = [NTHREADS, 1, 1],
        args = (logits, partials),
    }
}
EOF

cat > "$WORK/src/main.rs" <<'EOF'
use fox_offload::kernels::argmax::{ArgmaxArgs, BlockArgmax, Candidate};
use fox_offload::{launch_cpu, GridDim};
use oracle::{N, NGROUPS, NTHREADS};

/// Deliberately hostile: negatives, a tie, a NaN, and one real maximum.
fn logits() -> [f32; N] {
    let mut x = [0.0f32; N];
    for (i, v) in x.iter_mut().enumerate() {
        *v = ((i * 37 % 101) as f32) - 50.0;
    }
    x[17] = 12.5;
    x[200] = 12.5; // tie: the greater index must win
    x[99] = f32::NAN; // a NaN must never win
    x[123] = 77.0; // the real maximum
    x
}

fn winner(p: &[Candidate]) -> Candidate {
    p.iter().copied().fold(Candidate::NONE, |a, b| {
        if b.value > a.value || (b.value == a.value && b.index > a.index) {
            b
        } else {
            a
        }
    })
}

fn main() {
    let mut gpu_in = logits();
    let mut gpu_out = [Candidate::NONE; N];
    oracle::run_gpu(&mut gpu_in, &mut gpu_out);

    let cpu_in = logits();
    let mut cpu_out = [Candidate::NONE; N];
    let grid = GridDim::linear(NGROUPS, NTHREADS);
    let mut args = ArgmaxArgs {
        logits: &cpu_in[..],
        partials: &mut cpu_out[..],
    };
    let threads = launch_cpu::<BlockArgmax>(&mut args, &grid);

    let mut diffs = 0usize;
    let mut first = None;
    for k in 0..N {
        let (g, c) = (gpu_out[k], cpu_out[k]);
        let same =
            g.index == c.index && (g.value == c.value || (g.value.is_nan() && c.value.is_nan()));
        if !same {
            diffs += 1;
            if first.is_none() {
                first = Some((k, g, c));
            }
        }
    }

    println!("   threads launched (CPU oracle): {threads}");
    println!("   partials compared:             {N}");
    println!("   divergences:                   {diffs}");
    if let Some((k, g, c)) = first {
        println!("   first at k={k}: gpu={g:?} cpu={c:?}");
    }
    let (gw, cw) = (winner(&gpu_out), winner(&cpu_out));
    println!("   argmax GPU: {gw:?}");
    println!("   argmax CPU: {cw:?}");

    if diffs == 0 && gw == cw {
        println!("   VERDICT: identical");
    } else {
        println!("   VERDICT: DIVERGED");
        std::process::exit(1);
    }
}
EOF

cd "$WORK" || exit 1

step "pass 1/3 — host metadata"
if RUSTFLAGS="-Zunstable-options -Zoffload=HostMetadata=$WORK/meta.bin" \
     cargo +nightly build --release --lib >/dev/null 2>&1 && [ -s meta.bin ]; then
  ok "manifest written"
else
  bad "host metadata pass"; exit 1
fi

# -Zbuild-std=core only. If this ever needs `alloc`, something in fox-offload
# reached for it outside a host-only cfg — that is the regression this catches.
step "pass 2/3 — device codegen for $ARCH, core only"
if RUSTFLAGS="-Zunstable-options -Zoffload=Device=$WORK/meta.bin -Ctarget-cpu=$ARCH" \
     cargo +nightly build --release --lib -Zbuild-std=core --target "$TARGET" >/dev/null 2>&1; then
  DEV="$(ls -t target/"$TARGET"/release/build/oracle/*/out/device.bin 2>/dev/null | head -1)"
  [ -n "$DEV" ] && ok "device image ($(wc -c < "$DEV") bytes)" || { bad "no device.bin"; exit 1; }
else
  bad "device pass — fox-offload does not compile for $TARGET"
  printf '         re-run without >/dev/null to see it. The usual causes are an\n'
  printf '         unconditional `extern crate alloc`, a host-only module that is not\n'
  printf '         cfg-gated, or arch intrinsics used without their feature gate.\n'
  exit 1
fi

step "pass 3/3 — host link"
if RUSTFLAGS="-Zunstable-options -Zoffload=Host=$(realpath "$DEV") -L native=$TOOLCHAIN_LIB -Clink-arg=-lomptarget -Clink-arg=-Wl,-rpath,$TOOLCHAIN_LIB" \
     cargo +nightly build --release >/dev/null 2>&1; then
  ok "linked"
else
  bad "host link"; exit 1
fi

step "embed the device image"
HOSTO="$(ls -t target/release/build/oracle/*/out/host.o 2>/dev/null | head -1)"
[ -n "$HOSTO" ] || { bad "no host.o"; exit 1; }
# Driving the wrapper through `rustc -Clinker=` does not work: rustc adds
# --gc-sections / -nodefaultlibs / -fuse-ld=lld and the image is dropped.
if "$FOX_OFFLOAD_WRAPPER" --should-extract="$ARCH" \
     --device-linker="$TARGET=-L$OMPDEVICE" --device-linker="$TARGET=-lompdevice" \
     --host-triple=x86_64-unknown-linux-gnu --linker-path=/usr/bin/cc \
     -L/usr/lib/gcc/x86_64-linux-gnu/13 -L/usr/lib/x86_64-linux-gnu -L"$TOOLCHAIN_LIB" \
     "$(realpath "$HOSTO")" -lomptarget -lomp -Wl,-rpath,"$TOOLCHAIN_LIB" \
     -o "$WORK/oracle" >/dev/null 2>&1 && grep -qa "$ARCH" "$WORK/oracle"; then
  ok "image present in the binary"
else
  bad "wrapper link failed, or the image was dropped"; exit 1
fi

step "run — GPU against the CPU oracle"
# MANDATORY is not optional here. Without it a broken offload silently falls back to
# the host and prints the correct answer, which is the most efficient way to lose a day.
HSA_OVERRIDE_GFX_VERSION="${HSA_OVERRIDE_GFX_VERSION:-11.0.0}" \
OMP_TARGET_OFFLOAD=MANDATORY LD_LIBRARY_PATH="$RUNLIBS" \
  timeout 180 "$WORK/oracle"
RC=$?

if [ $RC -eq 0 ]; then
  printf '\n   PASS  the same kernel gives the same answer on both\n'
else
  printf '\n   FAIL  see above (exit %d)\n' "$RC"
  printf '         "does not have a matching target pointer" means zero GPUs were\n'
  printf '         registered, not a pointer bug. Check FOX_HSA_LIB.\n'
fi
exit $RC
