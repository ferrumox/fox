#!/usr/bin/env python3
"""Fail when a chat prompt is rendered by hand and then tokenized as raw text.

There are two tokenizers and they are not interchangeable:

    tokenize()             add_special = true,  parse_special = FALSE   raw user text
    build_prompt_tokens()  add_special = false, parse_special = TRUE    chat prompts

Rendering the template and then calling `tokenize()` on the result looks like what
`build_prompt_tokens` does and is not: the template's `<start_of_turn>` markers go in as
literal text instead of the control tokens they are, and a second BOS lands on top of the
one the template already emits.

`fox run` did this from the day it was written. The model saw a conversation with no turn
structure and answered often enough by writing a literal `<start_of_turn>model`, which
the output filter correctly held back as a control pattern — so the reply arrived empty
and the REPL blamed a full context window. Every session was degraded; the empty replies
were only where it became impossible to miss. The same two lines had been copied into
four `bench*` commands.

Nothing in the test suite caught it: `make e2e` passed 22 of 22 because every one of
those tests goes over HTTP, and the HTTP handlers use the right call.

So this checks the shape instead. Within one function body, `apply_chat_template`
followed by `tokenize` is the mistake, whatever the intervening lines say.

Usage: check_prompt_tokenization.py
"""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

# The trait's own default implementation is this shape on purpose: a backend with no
# Jinja template has no special tokens to parse, so generic tokenization is all it can
# do. It is documented as such, and `LlamaCppModel` overrides it.
ALLOW = {("engine/model/mod.rs", "build_prompt_tokens")}

FN_START = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([a-z_0-9]+)", re.M)
COMMENT = re.compile(r"//[^\n]*")


def strip_comments(text: str) -> str:
    """Blank out `//` comments, preserving length so line numbers still line up.

    Needed because the comments warning about this very mistake name both functions,
    and the first version of this check reported the fix as the bug.
    """
    return COMMENT.sub(lambda m: " " * len(m.group(0)), text)


def function_at(text: str, pos: int) -> str:
    """Name of the function containing `pos` — the nearest `fn` declared above it."""
    last = "<top level>"
    for m in FN_START.finditer(text, 0, pos):
        last = m.group(1)
    return last


def main() -> int:
    problems = []
    scanned = 0
    for rs in sorted(SRC.rglob("*.rs")):
        text = strip_comments(rs.read_text(errors="ignore"))
        if "apply_chat_template" not in text:
            continue
        scanned += 1
        rel = str(rs.relative_to(SRC))
        for m in re.finditer(r"apply_chat_template", text):
            fn = function_at(text, m.start())
            if (rel, fn) in ALLOW:
                continue
            # Look ahead within the same function for a raw tokenize of the result.
            nxt = FN_START.search(text, m.end())
            end = nxt.start() if nxt else len(text)
            window = text[m.end():end]
            hit = re.search(r"\.tokenize\(", window)
            if hit:
                line = text.count("\n", 0, m.start()) + 1
                problems.append(
                    f"  src/{rel}:{line} — `{fn}` renders the chat template and then "
                    f"calls `.tokenize()` on it; use `build_prompt_tokens` instead"
                )

    if problems:
        print("Chat prompts tokenized as raw text:\n")
        print("\n".join(problems))
        print(
            "\n`tokenize()` is the raw-text tokenizer (parse_special = false), so the "
            "template's\ncontrol markers become literal text and the model loses its "
            "turn structure.\nSee CHANGELOG 0.20.5."
        )
        return 1

    print(f"prompt tokenization ok: {scanned} file(s) using chat templates checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
