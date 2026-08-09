#!/usr/bin/env python3
"""Build a GitHub release body: this version's CHANGELOG section, plus which file to download.

Releases used to publish with no notes at all — `softprops/action-gh-release` was
only ever handed files, so the page was a bare list of six assets. Two of them are
tarballs differing by a `-vulkan` suffix, and nothing said which was which; the one
without a suffix reads as the default, so people took the CPU build and reported the
Vulkan backend missing from the release. It was in the other tarball.

    scripts/release_notes.py 0.21.0 > body.md

Kept out of release.yml on purpose: a heredoc nested inside a YAML block scalar is
indentation-sensitive in two directions at once, and this is testable.
"""

import pathlib
import re
import sys

# Everything after the CHANGELOG section. Asset names are written as suffixes rather
# than spelled out with a version, so this never drifts out of date.
DOWNLOAD_GUIDE = """
---

## Which file do I want?

| Asset | For |
|---|---|
| `…-x86_64-unknown-linux-gnu-vulkan.tar.gz` | **GPU** — any Vulkan device, AMD, Intel or NVIDIA. Needs glibc 2.39+ and a Vulkan driver (Mesa RADV/ANV, or your vendor's) on the host. |
| `…-x86_64-unknown-linux-gnu.tar.gz` | **CPU only.** No GPU backend inside. Built against an older glibc for broad compatibility. |

The suffix is the only difference between the two names, so the CPU build is an easy
one to take by accident. If you want the GPU build, take the `-vulkan` file.

Or let the installer decide — it looks for `/dev/dri` and picks the Vulkan build when
a GPU is present:

```sh
curl -fsSL https://github.com/ferrumox/fox/releases/latest/download/install.sh | sh
```

Pass `--vulkan` or `--cpu` to override the detection.

Unpacking by hand: keep the `.so` files in the same directory as the `fox` binary. It
is linked `RPATH=$ORIGIN` and loads its ggml backends from beside itself. Check a
download with `sha256sum -c <asset>.sha256`.

Once it is running, `fox probe` reports the backend actually in use, e.g.
`Vulkan0 — AMD Radeon 890M`. If that says CPU while the Vulkan build is installed,
the host is missing a Vulkan driver rather than the tarball missing a library.
"""


def section(changelog: str, version: str) -> str:
    """The body of `## [version]`, up to the next version heading."""
    lines = changelog.splitlines()
    start = next(
        (
            i
            for i, line in enumerate(lines)
            if re.match(rf"^## \[{re.escape(version)}\]", line)
        ),
        None,
    )
    if start is None:
        raise SystemExit(f"CHANGELOG has no '## [{version}]' section")

    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].startswith("## [")),
        len(lines),
    )
    body = lines[start + 1 : end]

    # Sections are separated by a `---` rule that belongs to neither side.
    while body and not body[-1].strip():
        body.pop()
    if body and body[-1].strip() == "---":
        body.pop()
    while body and not body[-1].strip():
        body.pop()

    return "\n".join(body).strip()


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: release_notes.py X.Y.Z")
    version = sys.argv[1]
    changelog = pathlib.Path(__file__).resolve().parent.parent / "CHANGELOG.md"
    print(section(changelog.read_text(), version))
    print(DOWNLOAD_GUIDE)


if __name__ == "__main__":
    main()
