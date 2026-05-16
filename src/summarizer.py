"""
summarizer.py
-------------
Emotion-aware summarization using the "anchor and compress" strategy.

The core problem with standard summarization:
  - Off-the-shelf summarizers (BART, T5) optimize for *information density*.
  - They strip out emotional nuance because it isn't "informative" in the
    factual sense — but for a story, the emotional turning points ARE
    the point.

Our fix — anchor and compress:
  1. From Phase 2's emotion scores, find "anchor" sentences: the ones
     with the highest emotional intensity (top 20% by 1 − P(neutral)).
     These are the story's emotional peaks.
  2. For the spans of text BETWEEN anchors, run normal abstractive
     summarization to compress connective tissue.
  3. Stitch anchors (verbatim) and bridges (summarized) back together
     in order. The summary now keeps every emotional turn while
     dropping filler.

Model: sshleifer/distilbart-cnn-12-6 (~300MB)
  - Distilled BART, ~2x faster than facebook/bart-large-cnn
  - Trained on CNN/DailyMail, good at narrative compression
  - Fits comfortably in 16GB RAM
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import gc
import json
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"


def _pick_device() -> str:
    """Same device selection logic as emotion_analyzer."""
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# --------------------------------------------------------------------
# Step 1: anchor selection
# --------------------------------------------------------------------

@dataclass
class AnchorConfig:
    """
    Tunable parameters for emotional-peak selection.

    intensity_percentile: keep sentences above this percentile of intensity.
                         0.80 = top 20%.
    min_anchors:         floor — even very flat stories get this many.
    max_anchors:         ceiling — even very emotional stories cap here,
                         otherwise the "summary" approaches full length.
    min_intensity:       absolute floor on intensity. If the top sentence
                         in the story is below this, the story is too flat
                         to anchor on emotion and we fall back to spacing
                         anchors evenly.
    """
    intensity_percentile: float = 0.80
    min_anchors: int = 3
    max_anchors: int = 10
    min_intensity: float = 0.30


def compute_intensity(sentence_emotions: list[dict]) -> np.ndarray:
    """Per-sentence intensity = 1 − P(neutral). Shape: (n,)."""
    return np.array(
        [1.0 - s["scores"]["neutral"] for s in sentence_emotions]
    )


def select_anchor_indices(
    sentence_emotions: list[dict],
    config: AnchorConfig | None = None,
) -> list[int]:
    """
    Return sorted indices of sentences chosen as emotional anchors.

    Strategy:
      1. Compute intensity per sentence.
      2. Build the candidate pool: sentences above BOTH the percentile
         threshold AND the absolute min_intensity floor.
         (The floor matters because with mostly-neutral text the
         intensity distribution is bimodal — a pure percentile cutoff
         can let baseline-noise sentences slip in.)
      3. If too many candidates, keep only the top max_anchors.
         If too few, top up from the next-highest-intensity sentences
         that still pass min_intensity.
      4. If even after step 3 we don't have min_anchors AND the story
         is essentially flat (max intensity < min_intensity), fall back
         to evenly-spaced indices so we still produce a summary.
    """
    cfg = config or AnchorConfig()
    n = len(sentence_emotions)
    if n == 0:
        return []

    intensities = compute_intensity(sentence_emotions)

    # Flat-story fallback: nothing rises above the floor
    if intensities.max() < cfg.min_intensity:
        k = min(cfg.min_anchors, n)
        # np.linspace gives k evenly-spaced indices including endpoints
        return [int(i) for i in np.linspace(0, n - 1, k).round()]

    # Two gates: percentile AND absolute floor. Both must pass.
    percentile_threshold = float(
        np.quantile(intensities, cfg.intensity_percentile)
    )
    effective_threshold = max(percentile_threshold, cfg.min_intensity)
    candidates = [
        i for i, v in enumerate(intensities) if v >= effective_threshold
    ]

    # Too many: keep only the top-intensity ones (preserving order)
    if len(candidates) > cfg.max_anchors:
        candidates = sorted(
            candidates,
            key=lambda i: intensities[i],
            reverse=True,
        )[: cfg.max_anchors]
        candidates.sort()

    # Too few: top up with next-highest-intensity sentences that still
    # pass min_intensity. If even that doesn't get us to min_anchors,
    # we accept fewer anchors — better than padding with noise.
    if len(candidates) < cfg.min_anchors:
        remaining = [
            i for i in range(n)
            if i not in candidates and intensities[i] >= cfg.min_intensity
        ]
        remaining.sort(key=lambda i: intensities[i], reverse=True)
        need = cfg.min_anchors - len(candidates)
        candidates = sorted(candidates + remaining[:need])

    return candidates


# --------------------------------------------------------------------
# Step 2: abstractive summarization (for bridges between anchors)
# --------------------------------------------------------------------

class BridgeSummarizer:
    """
    Thin wrapper around DistilBART for summarizing chunks of bridge text.
    Use as a context manager so the ~300MB model gets unloaded promptly.
    """

    def __init__(self, model_name: str = SUMMARIZER_MODEL, device: str | None = None):
        self.model_name = model_name
        self.device = device or _pick_device()
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        if self._model is not None:
            return
        print(f"[summarizer] Loading {self.model_name} on device={self.device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
        print("[summarizer] Model loaded.")

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    @torch.inference_mode()
    def summarize(
        self,
        text: str,
        max_length: int = 80,
        min_length: int = 15,
    ) -> str:
        """
        Summarize a single text chunk. Returns the summary string.

        max_length/min_length are in TOKENS, not words. ~80 tokens is
        roughly 50-60 words — about right for a one-sentence bridge.
        """
        if self._model is None:
            self.load()

        # DistilBART has a 1024-token input limit. We truncate; for
        # bridges this rarely matters because they're already short.
        inputs = self._tokenizer(
            text,
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        # Generation params chosen for narrative text:
        # - num_beams=4: better quality than greedy, still fast
        # - length_penalty=1.0: neutral (don't bias toward long/short)
        # - no_repeat_ngram_size=3: prevents the model parroting phrases
        output_ids = self._model.generate(
            **inputs,
            max_length=max_length,
            min_length=min_length,
            num_beams=4,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
        return self._tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *args):
        self.unload()


# --------------------------------------------------------------------
# Step 3: stitch anchors + bridges
# --------------------------------------------------------------------

@dataclass
class SummarySegment:
    """One piece of the final summary — either a verbatim anchor or a
    summarized bridge."""
    text: str
    kind: str  # "anchor" or "bridge"
    source_indices: list[int] = field(default_factory=list)
    # For anchors, this contains the original sentence's emotion data,
    # which Phase 4 (TTS) will use to pick a voice tone.
    emotion: dict | None = None


def build_emotion_aware_summary(
    sentence_emotions: list[dict],
    anchor_config: AnchorConfig | None = None,
    summarizer: BridgeSummarizer | None = None,
    skip_short_bridges: int = 1,
    bridge_max_length: int = 80,
) -> list[SummarySegment]:
    """
    Build a list of SummarySegments forming the emotion-aware summary.

    Parameters
    ----------
    sentence_emotions:
        Output of Phase 2 — list of {text, top_label, top_score, scores}.
    anchor_config:
        How aggressively to pick anchors. None = sensible defaults.
    summarizer:
        Optional pre-loaded BridgeSummarizer. If None, one is loaded
        and unloaded inside this function (slower for repeated calls).
    skip_short_bridges:
        If a bridge span is <= this many sentences, skip summarization
        and include the original sentences verbatim (summarizing a single
        sentence often produces a near-copy or a garbled version anyway).
        Default 1 — only skip truly trivial single-sentence bridges.
    bridge_max_length:
        Token cap for each bridge summary.

    Returns
    -------
    Ordered list of SummarySegment, ready to be concatenated into prose
    or fed sentence-by-sentence to a TTS engine.
    """
    cfg = anchor_config or AnchorConfig()
    anchor_idxs = select_anchor_indices(sentence_emotions, cfg)
    if not anchor_idxs:
        return []

    sentences = [s["text"] for s in sentence_emotions]
    segments: list[SummarySegment] = []

    # Manage the summarizer lifecycle. If the caller passed one in,
    # we assume they manage it. Otherwise we load/unload ourselves.
    owns_summarizer = summarizer is None
    if owns_summarizer:
        summarizer = BridgeSummarizer()
        summarizer.load()

    try:
        # cursor walks through the sentence list. At each anchor we:
        #   1. Handle everything from cursor up to the anchor (bridge)
        #   2. Emit the anchor verbatim
        #   3. Advance cursor past the anchor
        cursor = 0
        for anchor_i in anchor_idxs:
            # Bridge before this anchor
            bridge_span = sentences[cursor:anchor_i]
            if len(bridge_span) > skip_short_bridges:
                # Long enough to summarize
                bridge_text = " ".join(bridge_span)
                summary = summarizer.summarize(
                    bridge_text, max_length=bridge_max_length
                )
                segments.append(SummarySegment(
                    text=summary,
                    kind="bridge",
                    source_indices=list(range(cursor, anchor_i)),
                ))
            elif bridge_span:
                # Too short to summarize meaningfully — include verbatim.
                # Summarizing 1-2 sentences usually returns either the
                # original or a slightly mangled version. Better to keep
                # the author's words.
                segments.append(SummarySegment(
                    text=" ".join(bridge_span),
                    kind="bridge",
                    source_indices=list(range(cursor, anchor_i)),
                ))

            # The anchor itself, verbatim
            segments.append(SummarySegment(
                text=sentence_emotions[anchor_i]["text"],
                kind="anchor",
                source_indices=[anchor_i],
                emotion={
                    "top_label": sentence_emotions[anchor_i]["top_label"],
                    "top_score": sentence_emotions[anchor_i]["top_score"],
                    "scores": sentence_emotions[anchor_i]["scores"],
                },
            ))
            cursor = anchor_i + 1

        # Trailing bridge after the last anchor
        tail = sentences[cursor:]
        if len(tail) > skip_short_bridges:
            tail_text = " ".join(tail)
            summary = summarizer.summarize(
                tail_text, max_length=bridge_max_length
            )
            segments.append(SummarySegment(
                text=summary,
                kind="bridge",
                source_indices=list(range(cursor, len(sentences))),
            ))
        elif tail:
            segments.append(SummarySegment(
                text=" ".join(tail),
                kind="bridge",
                source_indices=list(range(cursor, len(sentences))),
            ))
    finally:
        if owns_summarizer:
            summarizer.unload()

    return segments


# --------------------------------------------------------------------
# Convenience: full pipeline + serialization
# --------------------------------------------------------------------

def segments_to_text(segments: list[SummarySegment]) -> str:
    """Render summary segments as a single readable string."""
    return " ".join(seg.text for seg in segments).strip()


def segments_to_json(segments: list[SummarySegment]) -> list[dict]:
    """Serialize to JSON-friendly dicts for Phase 4 to consume."""
    return [
        {
            "text": seg.text,
            "kind": seg.kind,
            "source_indices": seg.source_indices,
            "emotion": seg.emotion,
        }
        for seg in segments
    ]


def summarize_analyzed_file(
    analyzed_path: str | Path,
    output_path: str | Path | None = None,
    anchor_config: AnchorConfig | None = None,
) -> dict:
    """
    Load an emotion-analyzed JSON file (Phase 2 output) and produce an
    emotion-aware summary. Optionally save to disk.

    Output JSON adds a "summary" field:
        {
          ... existing Phase 2 fields ...,
          "summary": {
              "segments":   [...],         # ordered list for TTS
              "text":       "...",         # rendered prose
              "anchor_indices": [...],     # which sentences were anchors
              "compression_ratio": 0.34,   # summary length / original
          }
        }
    """
    analyzed_path = Path(analyzed_path)
    with analyzed_path.open(encoding="utf-8") as f:
        data = json.load(f)

    if "sentence_emotions" not in data:
        raise ValueError(
            f"{analyzed_path} doesn't have 'sentence_emotions'. "
            f"Run emotion analysis first."
        )

    sentence_emotions = data["sentence_emotions"]
    segments = build_emotion_aware_summary(
        sentence_emotions, anchor_config=anchor_config
    )

    summary_text = segments_to_text(segments)
    original_text = " ".join(s["text"] for s in sentence_emotions)
    compression = (
        len(summary_text.split()) / max(len(original_text.split()), 1)
    )

    data["summary"] = {
        "segments": segments_to_json(segments),
        "text": summary_text,
        "anchor_indices": [
            seg.source_indices[0] for seg in segments if seg.kind == "anchor"
        ],
        "compression_ratio": compression,
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Saved -> {output_path}")
        print(f"  Compression: {compression:.0%} of original length")

    return data
