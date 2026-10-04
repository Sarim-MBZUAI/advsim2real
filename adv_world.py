"""Render gate for injected pages (coherent page that shows the declared marker)."""
from __future__ import annotations

import re

_ELEM_RE = re.compile(r"\[\d+\]")


def page_is_complete(page: str | None, *, min_chars: int = 40, min_elems: int = 1) -> bool:
    if not page or len(page.strip()) < min_chars:
        return False
    return "rootwebarea" in page.lower() or len(_ELEM_RE.findall(page)) >= min_elems


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


def marker_hit(page: str | None, marker: str, *, frac: float = 0.6) -> bool:
    """Marker is a substring of the page, or >= frac of its tokens longer than 2 chars appear."""
    if not page or not marker:
        return False
    p, m = _norm(page), _norm(marker)
    if m in p:
        return True
    toks = [t for t in m.split() if len(t) > 2]
    if not toks:
        return m in p
    return sum(1 for t in toks if t in p) / len(toks) >= frac
