#!/usr/bin/env python3
"""Count reader-facing prose in the five neutral-atom-control chapters."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOWER, UPPER = 2800, 3200


def reader_words(path: Path) -> int:
    text = path.read_text()
    text = re.sub(r"\A---\n.*?\n---\n", "", text, flags=re.DOTALL)
    kept: list[str] = []
    in_display = False
    for line in text.splitlines():
        if "$$" in line:
            pieces = line.split("$$")
            for index, piece in enumerate(pieces):
                if not in_display:
                    kept.append(piece)
                if index < len(pieces) - 1:
                    in_display = not in_display
            continue
        if in_display or re.match(r"^\s*\|.*\|\s*$", line):
            continue
        if re.match(r"^\s*\{%.*%\}\s*$", line):
            continue
        kept.append(line)

    prose = "\n".join(kept)
    prose = re.sub(r"<[^>]+>", " ", prose)
    prose = re.sub(r"\[([^]]+)]\([^)]+\)", r"\1", prose)
    prose = re.sub(r"(?<!\$)\$(?!\$).*?(?<!\$)\$(?!\$)", " symbol ", prose)
    return len(re.findall(r"[A-Za-zÀ-ž0-9]+(?:[’'-][A-Za-zÀ-ž0-9]+)*", prose))


def main() -> None:
    failed = False
    for path in sorted((ROOT / "_projects").glob("neutral-atom-control-part-*.md")):
        count = reader_words(path)
        print(f"{path.name}: {count} reader-facing words")
        failed |= not LOWER <= count <= UPPER
    if failed:
        raise SystemExit(f"chapter prose must remain between {LOWER} and {UPPER} words")


if __name__ == "__main__":
    main()
