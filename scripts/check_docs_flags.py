#!/usr/bin/env python3
"""Fail when the documentation names a flag or env var fox does not have.

Documentation drifts silently and is only found by someone acting on it. In one session
five separate documents were found asserting that prompt reuse was disabled for hybrid
models — true when written, false for a release — and a stale code comment claiming a
branch was unreachable cost an hour of misdirected debugging. Prose cannot be compiled,
but the *names* in it can be checked, and names are what a reader copies.

Two directions, both cheap:

  docs → binary   every `--flag` in the CLI docs must appear in some `fox … --help`
  docs → source   every `FOX_*` env var mentioned must appear in the source

Scope is deliberately narrow: only files that document fox's own interface. README and
design docs quote `llama-server`, Ollama and cargo flags constantly, and checking those
would produce noise nobody reads, which is worse than not checking.

Usage: check_docs_flags.py [path/to/fox]
"""
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FOX = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "target/debug/fox")

# Files that document fox's own surface. Anything else quotes other tools' flags.
DOC_GLOBS = ["docs/cli/*.md", "docs/configuration.md"]

# Flags that are real but not discoverable from `--help`: global cargo/make context in
# examples, or flags of the subcommand the page is *about* being shown as prose.
ALLOW = {
    "--help", "--version", "--recurse-submodules", "--release", "--features",
    "--all-targets", "--no-run", "--nocapture", "--test-threads",
}


def cli_flags() -> set[str]:
    """Every long flag reachable from `fox --help` and each subcommand's help."""
    out = set()

    def scrape(args):
        try:
            r = subprocess.run([FOX, *args, "--help"], capture_output=True, text=True,
                               timeout=30, env={"PATH": "/usr/bin:/bin", "FOX_SKIP_LLAMA": "1"})
        except (OSError, subprocess.TimeoutExpired):
            return ""
        return r.stdout + r.stderr

    root = scrape([])
    out.update(re.findall(r"(--[a-z0-9][a-z0-9-]+)", root))
    subs = re.findall(r"^\s{2}([a-z][a-z0-9-]+)\s{2,}\S", root, re.M)
    for sub in subs:
        if sub == "help":
            continue
        out.update(re.findall(r"(--[a-z0-9][a-z0-9-]+)", scrape([sub])))
    return out


def source_env_vars() -> set[str]:
    found = set()
    for rs in (ROOT / "src").rglob("*.rs"):
        found.update(re.findall(r'"(FOX_[A-Z0-9_]+)"', rs.read_text(errors="ignore")))
    for extra in ("build.rs", "Makefile"):
        p = ROOT / extra
        if p.exists():
            found.update(re.findall(r"\b(FOX_[A-Z0-9_]+)\b", p.read_text(errors="ignore")))
    return found


def main() -> int:
    if not pathlib.Path(FOX).exists():
        print(f"skip: {FOX} not built — run `cargo build --bin fox` first")
        return 0

    known_flags = cli_flags()
    if len(known_flags) < 10:
        print(f"error: only {len(known_flags)} flags scraped from {FOX} — the help output "
              "probably did not parse, and a check that finds nothing is worse than none")
        return 1
    known_env = source_env_vars()

    problems = []
    for glob in DOC_GLOBS:
        for doc in sorted(ROOT.glob(glob)):
            text = doc.read_text(errors="ignore")
            rel = doc.relative_to(ROOT)
            for flag in sorted(set(re.findall(r"`(--[a-z0-9][a-z0-9-]+)", text))):
                if flag not in known_flags and flag not in ALLOW:
                    problems.append(f"{rel}: `{flag}` is documented but no `fox … --help` lists it")
            for var in sorted(set(re.findall(r"`(FOX_[A-Z0-9_]+)`", text))):
                if var not in known_env:
                    problems.append(f"{rel}: `{var}` is documented but appears nowhere in the source")

    if problems:
        print("Documentation names things fox does not have:\n")
        for p in problems:
            print("  " + p)
        print("\nEither the flag was renamed and the docs were not, or the docs invented it.")
        return 1

    print(f"docs ok: {len(known_flags)} flags and {len(known_env)} env vars cross-checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
