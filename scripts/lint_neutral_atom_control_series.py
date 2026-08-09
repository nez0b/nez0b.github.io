#!/usr/bin/env python3
"""Source lint for the neutral-atom project series."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAGES = sorted((ROOT / "_projects").glob("neutral-atom-control*.md"))
BIB = ROOT / "assets/bibliography/neutral-atom-control.bib"


def prose_without_math(text: str) -> str:
    lines = []
    display = False
    for line in text.splitlines():
        if line.count("$$"):
            chunks = line.split("$$")
            kept = []
            for index, chunk in enumerate(chunks):
                if not display:
                    kept.append(chunk)
                if index < len(chunks) - 1:
                    display = not display
            line = "".join(kept)
        elif display:
            line = ""
        line = re.sub(r"(?<!\\)\$[^$]*?(?<!\\)\$", "", line)
        lines.append(line)
    return "\n".join(lines)


def main() -> None:
    failures: list[str] = []
    cited: set[str] = set()
    hostile = [
        (re.compile(r"\\[()]"), "Markdown-hostile inline delimiter"),
        (re.compile(r"(?:\\[A-Za-z]+|[A-Za-z])\*\{"), "malformed starred subscript"),
        (re.compile(r"\\[A-Za-z]+\*[A-Za-z0-9]"), "malformed starred symbol"),
    ]
    raw_tex = re.compile(
        r"\\(?:Omega|Delta|theta|phi|mathrm|operatorname|langle|rangle|mu|times|sqrt|dagger|mathcal)\b"
    )
    for page in PAGES:
        text = page.read_text()
        for pattern, label in hostile:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"{page.name}:{line}: {label}: {match.group(0)!r}")
        prose = prose_without_math(text)
        for match in raw_tex.finditer(prose):
            line = prose.count("\n", 0, match.start()) + 1
            failures.append(f"{page.name}:{line}: raw TeX outside math: {match.group(0)!r}")
        if text.count("$$") % 2:
            failures.append(f"{page.name}: unmatched display-math delimiter")
        for match in re.finditer(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", text, re.DOTALL):
            if "|" in match.group(1):
                line = text.count("\n", 0, match.start()) + 1
                failures.append(
                    f"{page.name}:{line}: raw vertical bar in inline math can be parsed as a Markdown table"
                )
        for match in re.finditer(r"\bG[1-7](?:\b|–)|\bnotebooks?\b", prose, re.IGNORECASE):
            line = prose.count("\n", 0, match.start()) + 1
            failures.append(
                f"{page.name}:{line}: reader-facing internal experiment label: {match.group(0)!r}"
            )
        cited.update(re.findall(r'<d-cite\s+key="([^"]+)"', text))
        for asset in re.findall(r'path="([^"]+)"', text):
            if not (ROOT / asset).exists():
                failures.append(f"{page.name}: missing figure asset {asset}")

    bib_keys = set(re.findall(r"@[A-Za-z]+\{([^,]+),", BIB.read_text()))
    for key in sorted(cited - bib_keys):
        failures.append(f"missing bibliography key: {key}")
    landing = (ROOT / "_projects/neutral-atom-control.md").read_text()
    if "series-overview" in landing:
        failures.append("landing page still references superseded series-overview")
    forbidden_reader_phrases = {
        "opentikz": "OpenTikZ reference remains",
        "An original TikZ schematic": "superseded cover-caption language remains",
        "local reproduction repository": "repository-centric prose remains",
    }
    public_sources = "\n".join(page.read_text() for page in PAGES) + "\n" + BIB.read_text()
    for phrase, label in forbidden_reader_phrases.items():
        if phrase.lower() in public_sources.lower():
            failures.append(label)
    if re.search(r"\bG[1-7]\b", public_sources):
        failures.append("reader-facing internal experiment label remains")
    for part in range(1, 6):
        if f"series_part: {part}" not in (ROOT / f"_projects/neutral-atom-control-part-{part}.md").read_text():
            failures.append(f"part {part}: missing series_part metadata")

    if failures:
        raise SystemExit("\n".join(failures))
    print(f"neutral-atom series lint passed: {len(PAGES)} pages, {len(cited)} citation keys")


if __name__ == "__main__":
    main()
