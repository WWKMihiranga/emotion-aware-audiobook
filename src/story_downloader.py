"""
story_downloader.py
-------------------
Downloads short public-domain stories from Project Gutenberg
for use as test data.

Why these specific stories?
- All public domain (free to use, no license worries)
- Short enough to process on M4 MacBook without timeouts
- Emotionally rich — good test cases for our emotion analysis later
- Well-known stories so we can sanity-check our output
"""

from pathlib import Path
import requests
from tqdm import tqdm


# Curated short stories from Project Gutenberg.
# Curated short stories from Project Gutenberg.
# We picked these because each has CLEAR emotional arcs:
#   - The Gift of the Magi: tenderness → surprise → bittersweet joy
#   - The Necklace:         pride → despair → bitter revelation
#   - The Tell-Tale Heart:  calm → mounting dread → frenzy
#
# Some Gutenberg files are anthologies (multiple stories in one .txt).
# For those, we specify `section_start` and `section_end` markers that
# bracket the single story we want, applied AFTER Gutenberg-boilerplate
# stripping. Markers are matched as plain substrings (case-insensitive).
TEST_STORIES = {
    "gift_of_the_magi": {
        "url": "https://www.gutenberg.org/cache/epub/7256/pg7256.txt",
        "title": "The Gift of the Magi",
        "author": "O. Henry",
        # Single-story file — no section extraction needed.
        "section_start": None,
        "section_end": None,
    },
    "the_necklace": {
        # PG 3090 — Original Short Stories Volume 1 (Maupassant)
        # contains "The Necklace" cleanly. (PG 3077 was the wrong volume.)
        "url": "https://www.gutenberg.org/files/3090/3090-0.txt",
        "title": "The Necklace",
        "author": "Guy de Maupassant",
        "section_start": "THE NECKLACE",
        # The story that comes right after in this collection
        "section_end": "THE PIECE OF STRING",
    },
    "tell_tale_heart": {
        # PG 2148 — Works of Poe Vol 2. Tell-Tale Heart sits between
        # "WILLIAM WILSON" and "BERENICE" in the table of contents.
        "url": "https://www.gutenberg.org/files/2148/2148-0.txt",
        "title": "The Tell-Tale Heart",
        "author": "Edgar Allan Poe",
        "section_start": "THE TELL-TALE HEART",
        "section_end": "BERENICE",
    },
}


def download_story(story_key: str, output_dir: str | Path) -> Path:
    """
    Download a single story by its key into output_dir.

    If the file already exists, skip the download (so reruns are fast).
    """
    if story_key not in TEST_STORIES:
        raise ValueError(
            f"Unknown story key: {story_key}. "
            f"Available: {list(TEST_STORIES.keys())}"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    info = TEST_STORIES[story_key]
    output_path = output_dir / f"{story_key}.txt"

    # Skip if cached — Gutenberg gets cranky about repeated hits
    if output_path.exists():
        print(f"[cached] {info['title']} -> {output_path}")
        return output_path

    print(f"Downloading: {info['title']} by {info['author']}")
    resp = requests.get(info["url"], timeout=30)
    resp.raise_for_status()

    # Gutenberg sometimes serves Latin-1 — we let requests sniff and
    # then re-encode to UTF-8 on disk so the rest of our pipeline can
    # assume UTF-8 everywhere.
    output_path.write_text(resp.text, encoding="utf-8")
    print(f"  saved -> {output_path} ({len(resp.text):,} chars)")

    return output_path


def download_all(output_dir: str | Path) -> dict[str, Path]:
    """
    Download every test story. Returns a dict mapping story_key -> Path.
    """
    output_dir = Path(output_dir)
    results = {}

    for key in tqdm(TEST_STORIES.keys(), desc="Stories"):
        try:
            results[key] = download_story(key, output_dir)
        except Exception as e:
            # We don't want one failed download to break the others.
            # Network issues, Gutenberg downtime, etc. are common.
            print(f"  [WARN] Failed to download {key}: {e}")

    return results


def load_story_by_key(story_key: str, raw_dir: str | Path) -> str:
    """
    Convenience helper: download (if needed) and load a registered story,
    applying its anthology section markers automatically.

    Returns the cleaned single-story text — same shape as
    text_loader.load_story(), just with the markers wired up for you.
    """
    # Imported here to avoid a circular import at module load time
    # (text_loader doesn't depend on us, but we use it here).
    from . import text_loader

    if story_key not in TEST_STORIES:
        raise ValueError(
            f"Unknown story key: {story_key}. "
            f"Available: {list(TEST_STORIES.keys())}"
        )

    info = TEST_STORIES[story_key]
    raw_path = Path(raw_dir) / f"{story_key}.txt"
    if not raw_path.exists():
        download_story(story_key, raw_dir)

    return text_loader.load_story(
        raw_path,
        section_start=info.get("section_start"),
        section_end=info.get("section_end"),
    )
