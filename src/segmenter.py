"""
segmenter.py
------------
Splits a cleaned story into segments suitable for emotion analysis.

Why two granularities?
- SENTENCES: precise unit for emotion labeling later (Phase 2).
- PARAGRAPHS: better unit for summarization context (Phase 3).
We compute both now so later phases can pick whichever they need.

We use NLTK's Punkt tokenizer because writing our own sentence splitter
is a rabbit hole — "Mr. Smith said 'Hello.' Then he left." has THREE
periods but only TWO sentences. Punkt handles that correctly.
"""

import nltk
from nltk.tokenize import sent_tokenize


def ensure_nltk_data():
    """
    Download NLTK's sentence tokenizer data the first time we run.
    Safe to call repeatedly — it's a no-op if already downloaded.
    """
    try:
        # punkt_tab is the newer version required by NLTK 3.9+
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        print("First-time setup: downloading NLTK sentence tokenizer (~5MB)...")
        nltk.download("punkt_tab", quiet=True)


def split_into_paragraphs(text: str) -> list[str]:
    """
    Split story text into paragraphs.

    A paragraph is anything separated by one or more blank lines.
    We filter out empty paragraphs so downstream code never has to.
    """
    paragraphs = text.split("\n\n")
    # Strip and remove empties
    return [p.strip() for p in paragraphs if p.strip()]


def split_into_sentences(text: str) -> list[str]:
    """
    Split story text into sentences using NLTK's Punkt tokenizer.
    Handles abbreviations, decimals, and quoted speech correctly.
    """
    ensure_nltk_data()

    # First flatten paragraphs into one continuous string — sent_tokenize
    # doesn't care about paragraph breaks, only sentence boundaries.
    # We replace paragraph breaks with single spaces so it doesn't see
    # "end.\n\nNext" as one sentence.
    flat = text.replace("\n\n", " ").replace("\n", " ")

    sentences = sent_tokenize(flat)
    # Clean up — strip whitespace, drop empties
    return [s.strip() for s in sentences if s.strip()]


def segment_story(text: str) -> dict:
    """
    Run all segmentation on a story at once and return everything
    downstream phases need.

    Returns a dict with:
        - 'sentences':  list[str]
        - 'paragraphs': list[str]
        - 'stats':      dict with counts
    """
    sentences = split_into_sentences(text)
    paragraphs = split_into_paragraphs(text)

    return {
        "sentences": sentences,
        "paragraphs": paragraphs,
        "stats": {
            "num_sentences": len(sentences),
            "num_paragraphs": len(paragraphs),
            "avg_sentence_length_words": (
                sum(len(s.split()) for s in sentences) / len(sentences)
                if sentences else 0
            ),
            "avg_paragraph_length_words": (
                sum(len(p.split()) for p in paragraphs) / len(paragraphs)
                if paragraphs else 0
            ),
        },
    }
