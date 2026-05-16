"""
text_loader.py
--------------
Loads a storybook from a plain .txt file and returns clean text.

Why a dedicated module?
- Gutenberg files come with a long header/footer (license, metadata).
  We strip those so the rest of the pipeline sees ONLY the story.
- Real books have weird whitespace, page numbers, chapter markers, etc.
  We normalize all of that here so downstream code stays simple.
"""

from pathlib import Path
import re


# Gutenberg files have standard start/end markers. We detect them and
# slice out only the story body. If the markers aren't there, we return
# the whole file as-is.
GUTENBERG_START = re.compile(r"\*\*\* START OF.+?\*\*\*", re.IGNORECASE)
GUTENBERG_END = re.compile(r"\*\*\* END OF.+?\*\*\*", re.IGNORECASE)


def load_story(
    file_path: str | Path,
    section_start: str | None = None,
    section_end: str | None = None,
) -> str:
    """
    Read a story file from disk and return cleaned text.

    Parameters
    ----------
    file_path : str or Path
        Path to a .txt file containing the storybook.
    section_start, section_end : str, optional
        If the file is an anthology containing multiple stories, supply
        substring markers that bracket the single story to extract. We
        take the LAST occurrence of section_start (to skip the table of
        contents, which mentions every story title) and the FIRST
        occurrence of section_end that comes after it. Case-insensitive.

    Returns
    -------
    str
        The cleaned story text (Gutenberg headers/footers removed,
        optionally narrowed to a single section, whitespace normalized).
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Story file not found: {file_path}")

    # encoding='utf-8' with errors='ignore' — Gutenberg files sometimes
    # have stray non-UTF-8 bytes. We'd rather drop them than crash.
    raw_text = file_path.read_text(encoding="utf-8", errors="ignore")

    # Step 1: strip Gutenberg header/footer if present
    cleaned = _strip_gutenberg_boilerplate(raw_text)

    # Step 2: if this is an anthology, narrow to the single story
    if section_start or section_end:
        cleaned = _extract_section(cleaned, section_start, section_end)

    # Step 3: normalize whitespace
    cleaned = _normalize_whitespace(cleaned)

    return cleaned


def _extract_section(
    text: str,
    section_start: str | None,
    section_end: str | None,
) -> str:
    """
    Extract a single named section from an anthology file.

    We find the LAST occurrence of section_start because the first ones
    are almost always in the table of contents (which lists every story
    title near the top of the file). The actual story heading comes
    later, after the contents and any front-matter.

    For section_end, we use the FIRST occurrence that appears AFTER our
    chosen start position — that's the heading of the next story.

    If section_end isn't found after the start (i.e., this is the last
    story in the anthology), we just return everything from start to the
    end of the file.
    """
    # Case-insensitive search via .lower()
    lower = text.lower()

    start_idx = 0
    if section_start:
        marker = section_start.lower()
        # rfind = last occurrence. -1 if not found.
        found = lower.rfind(marker)
        if found == -1:
            raise ValueError(
                f"section_start marker not found in text: {section_start!r}"
            )
        # Skip past the marker itself so we don't include the title line
        # in the extracted body.
        start_idx = found + len(marker)

    end_idx = len(text)
    if section_end:
        marker = section_end.lower()
        # find = first occurrence at or after start_idx
        found = lower.find(marker, start_idx)
        if found != -1:
            end_idx = found
        # else: section_end not found after start — last story in anthology
        # is fine; keep end_idx at len(text).

    return text[start_idx:end_idx]


def _strip_gutenberg_boilerplate(text: str) -> str:
    """Cut everything before *** START *** and after *** END ***.

    Important: we re-search for END *after* slicing off the START header,
    because slicing shifts every byte offset. Using the original
    end_match.start() against the post-slice text gives wrong results
    (this was a real bug — caught it on the way in).
    """
    start_match = GUTENBERG_START.search(text)
    if start_match:
        text = text[start_match.end():]

    end_match = GUTENBERG_END.search(text)
    if end_match:
        text = text[:end_match.start()]

    return text.strip()


def _normalize_whitespace(text: str) -> str:
    """
    Fix common whitespace issues in book text:
    - Convert Windows/Mac line endings to Unix
    - Collapse 3+ blank lines into 2 (preserves paragraph breaks)
    - Strip trailing spaces on each line
    """
    # Unify line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Strip trailing spaces on each line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    # Collapse 3+ consecutive newlines into 2 (one blank line = paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def get_basic_stats(text: str) -> dict:
    """
    Quick sanity check on a loaded story.
    Useful for confirming the file loaded correctly before
    we run expensive downstream processing.
    """
    return {
        "characters": len(text),
        "words": len(text.split()),
        "lines": text.count("\n") + 1,
        "paragraphs": len([p for p in text.split("\n\n") if p.strip()]),
    }
