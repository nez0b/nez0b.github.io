#!/usr/bin/env python3
"""Check internal links and assets in a completed Jekyll build."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROUTES = [
    "/projects/neutral-atom-control/",
    "/projects/neutral-atom-control/part-1-foundations/",
    "/projects/neutral-atom-control/part-2-grape-krotov-crab/",
    "/projects/neutral-atom-control/part-3-collocation-piccolo/",
    "/projects/neutral-atom-control/part-4-noise-robustness/",
    "/projects/neutral-atom-control/part-5-hardware-bridge/",
]


class References(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.paths: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        attribute = "href" if tag == "a" else "src" if tag in {"img", "script", "source"} else None
        if attribute and values.get(attribute):
            self.paths.append(values[attribute] or "")


def target(build: Path, route: str, reference: str) -> Path | None:
    parsed = urlsplit(reference)
    if parsed.scheme or parsed.netloc or not parsed.path:
        return None
    path = unquote(parsed.path)
    if path.startswith("/"):
        candidate = build / path.lstrip("/")
    else:
        candidate = build / route.lstrip("/") / path
    if path.endswith("/"):
        candidate /= "index.html"
    elif not candidate.suffix and candidate.is_dir():
        candidate /= "index.html"
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", required=True, type=Path)
    args = parser.parse_args()
    missing: list[str] = []
    checked = 0
    for route in ROUTES:
        page = args.build_dir / route.lstrip("/") / "index.html"
        if not page.exists():
            missing.append(str(page))
            continue
        references = References()
        references.feed(page.read_text())
        for reference in references.paths:
            path = target(args.build_dir, route, reference)
            if path is None:
                continue
            checked += 1
            if not path.exists():
                missing.append(f"{route} -> {reference}")
    if missing:
        raise SystemExit("missing internal build targets:\n" + "\n".join(missing))
    print(f"build-link audit passed: {len(ROUTES)} pages and {checked} internal links/assets")


if __name__ == "__main__":
    main()
