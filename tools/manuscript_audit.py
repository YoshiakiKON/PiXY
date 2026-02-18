"""Manuscript audit utilities for PiXY SoftwareX drafts.

This script is intentionally dependency-free and PowerShell-friendly.

It reports:
- Word counts for the English manuscript with common exclusions
- Abstract/Keywords/Highlights compliance-style stats
- Structural alignment checks between EN and JP drafts (headings, Impact items)

Usage (from repo root):
    python tools/manuscript_audit.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
EN_PATH = ROOT / "paper_softwarex.md"
JP_PATH = ROOT / "paper_softwarex_JP.md"


_YAML_FRONT_MATTER_RE = re.compile(r"\A---\s*\n.*?\n---\s*\n", flags=re.S)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def strip_yaml_front_matter(text: str) -> str:
    m = _YAML_FRONT_MATTER_RE.match(text)
    return text[m.end() :] if m else text


def word_count_englishish(text: str) -> int:
    tokens = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text)
    return len(tokens)


def extract_block(text_wo_yaml: str, start_heading: str, end_heading: str | None) -> str:
    lines = text_wo_yaml.splitlines()
    try:
        start_i = lines.index(start_heading)
    except ValueError:
        return ""
    start_i += 1

    if end_heading is None:
        body = lines[start_i:]
    else:
        try:
            end_i = lines.index(end_heading)
        except ValueError:
            end_i = len(lines)
        body = lines[start_i:end_i]

    return "\n".join(body).strip() + "\n"


def drop_metadata_table(text_wo_yaml: str) -> str:
    lines = text_wo_yaml.splitlines()
    metadata_idx = None
    first_section_idx = None
    for i, ln in enumerate(lines):
        if metadata_idx is None:
            s = ln.strip()
            if s == "Metadata" or "Metadata" in s:
                metadata_idx = i

        if first_section_idx is None:
            if re.match(r"^\s*1(?:\.|)\s+", ln):
                first_section_idx = i

    if (
        metadata_idx is not None
        and first_section_idx is not None
        and first_section_idx > metadata_idx
    ):
        kept = lines[:metadata_idx] + [""] + lines[first_section_idx:]
        return "\n".join(kept)
    return text_wo_yaml


def drop_references_section(text: str) -> str:
    m = re.search(r"(?m)^(References|参考文献).*$", text)
    if not m:
        return text
    return text[: m.start()].rstrip() + "\n"


def list_numbered_headings(text_wo_yaml: str) -> list[str]:
    hs: list[str] = []
    for ln in text_wo_yaml.splitlines():
        s = ln.strip()
        if re.match(r"^\d+(?:\.\d+)*(?:\.|)\s+", s):
            hs.append(s)
    return hs


def list_heading_numbers(text_wo_yaml: str) -> list[str]:
    nums: list[str] = []
    for ln in text_wo_yaml.splitlines():
        s = ln.strip()
        m = re.match(r"^(\d+(?:\.\d+)*)(?:\.|)\s+", s)
        if m:
            nums.append(m.group(1))
    return nums


def extract_impact_items(en_text_wo_yaml: str) -> list[str]:
    """Extract Impact item numbers from section 4.

    Works for both EN and JP drafts:
    - Locates the section by its number ("4.") rather than its title.
    - Accepts both ASCII parentheses "(1)" and Japanese full-width "（1）".
    """

    m = re.search(r"(?m)^4\.\s+.+$", en_text_wo_yaml)
    if not m:
        return []
    tail = en_text_wo_yaml[m.end() :]
    m2 = re.search(r"(?m)^5\.\s", tail)
    sec = tail[: m2.start()] if m2 else tail
    return re.findall(r"[\(\（](\d)[\)\）]", sec)


def extract_numbers(text: str) -> set[str]:
    """Extract simple Arabic numerals (integers/decimals) from text."""
    return set(re.findall(r"\b\d+(?:\.\d+)?\b", text))


@dataclass(frozen=True)
class HighlightsStats:
    bullets: list[str]

    @property
    def lengths(self) -> list[int]:
        return [len(b) for b in self.bullets]


def parse_highlights(text_wo_yaml: str) -> HighlightsStats:
    hl = extract_block(text_wo_yaml, "Highlights", "Metadata")
    bullets: list[str] = []
    for ln in hl.splitlines():
        s = ln.strip()
        if s.startswith("-"):
            bullets.append(s[1:].strip())
    return HighlightsStats(bullets=bullets)


def parse_keywords(text_wo_yaml: str) -> list[str]:
    kw = extract_block(text_wo_yaml, "Keywords", "Highlights")
    items: list[str] = []
    for ln in kw.splitlines():
        s = ln.strip()
        if s.startswith("-"):
            items.append(s[1:].strip())
    return items


def print_list(title: str, items: Iterable[str]) -> None:
    print(title)
    for it in items:
        print(f"  - {it}")


def main() -> int:
    if not EN_PATH.exists():
        print(f"ERROR: Missing {EN_PATH}")
        return 2
    if not JP_PATH.exists():
        print(f"ERROR: Missing {JP_PATH}")
        return 2

    en_raw = read_text(EN_PATH)
    jp_raw = read_text(JP_PATH)

    en_wo_yaml = strip_yaml_front_matter(en_raw)
    jp_wo_yaml = strip_yaml_front_matter(jp_raw)

    # --- Counts (EN) ---
    abstract = extract_block(en_wo_yaml, "Abstract", "Keywords")
    keywords = parse_keywords(en_wo_yaml)
    highlights = parse_highlights(en_wo_yaml)

    en_wo_yaml_wo_meta = drop_metadata_table(en_wo_yaml)
    en_body_wo_refs = drop_references_section(en_wo_yaml_wo_meta)

    print("== English manuscript: counts ==")
    print(f"File: {EN_PATH.name}")
    print(f"TOTAL words (incl. YAML, tables, refs): {word_count_englishish(en_raw)}")
    print(
        "WORDS excluding YAML+Metadata table+References section: "
        f"{word_count_englishish(en_body_wo_refs)}"
    )
    print(f"ABSTRACT words: {word_count_englishish(abstract)}")
    print(f"KEYWORDS count: {len(keywords)} ({', '.join(keywords)})")
    print(f"HIGHLIGHTS bullets: {len(highlights.bullets)}")
    for i, (b, n) in enumerate(zip(highlights.bullets, highlights.lengths), start=1):
        print(f"  {i}. {n:3d} chars: {b}")
    print("")

    # --- Alignment checks ---
    en_headings = list_numbered_headings(en_wo_yaml)
    jp_headings = list_numbered_headings(jp_wo_yaml)
    en_heading_nums = list_heading_numbers(en_wo_yaml)
    jp_heading_nums = list_heading_numbers(jp_wo_yaml)

    print("== EN/JP alignment: structure ==")
    print(f"EN numbered headings: {len(en_headings)}")
    print(f"JP numbered headings: {len(jp_headings)}")

    if en_heading_nums == jp_heading_nums:
        print("OK: Heading numbering structure matches (e.g., 1, 2, 3.1, 3.2, ...)")
    else:
        print("NOTE: Heading numbering structure differs:")
        print(f"  EN: {en_heading_nums}")
        print(f"  JP: {jp_heading_nums}")

    en_impact = extract_impact_items(en_wo_yaml)
    jp_impact = extract_impact_items(jp_wo_yaml)
    print(f"EN Impact items: {''.join(en_impact) if en_impact else '(none found)'}")
    print(f"JP Impact items: {''.join(jp_impact) if jp_impact else '(none found)'}")
    expected = list("12345")
    if en_impact:
        missing = [x for x in expected if x not in en_impact]
        extra = [x for x in en_impact if x not in expected]
        if missing or extra:
            print(f"NOTE: EN Impact item numbers unexpected. missing={missing}, extra={extra}")
        else:
            print("OK: EN Impact items include (1)-(5)")

    if jp_impact:
        missing = [x for x in expected if x not in jp_impact]
        extra = [x for x in jp_impact if x not in expected]
        if missing or extra:
            print(f"NOTE: JP Impact item numbers unexpected. missing={missing}, extra={extra}")
        else:
            print("OK: JP Impact items include (1)-(5)")

    print("")

    # --- Alignment checks: numeric anchors ---
    print("== EN/JP alignment: numeric tokens (quick sanity) ==")
    en_body = drop_references_section(drop_metadata_table(en_wo_yaml))
    jp_body = drop_references_section(drop_metadata_table(jp_wo_yaml))
    en_nums = extract_numbers(en_body)
    jp_nums = extract_numbers(jp_body)
    en_only = sorted(en_nums - jp_nums, key=lambda x: (len(x), x))
    jp_only = sorted(jp_nums - en_nums, key=lambda x: (len(x), x))
    print(f"Unique EN numbers: {len(en_nums)}; Unique JP numbers: {len(jp_nums)}")
    if en_only:
        print(f"Numbers only in EN (first 25): {en_only[:25]}")
    else:
        print("OK: No EN-only number tokens")
    if jp_only:
        print(f"Numbers only in JP (first 25): {jp_only[:25]}")
    else:
        print("OK: No JP-only number tokens")

    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
