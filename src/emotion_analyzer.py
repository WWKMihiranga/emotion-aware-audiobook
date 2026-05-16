"""
emotion_analyzer.py
-------------------
Tags each sentence with an emotion + confidence using a pre-trained
DistilRoBERTa classifier.

Model: j-hartmann/emotion-english-distilroberta-base
  - 7 emotion labels: anger, disgust, fear, joy, neutral, sadness, surprise
  - ~330MB download, runs comfortably on CPU
  - Returns a softmax distribution over all 7 labels per input

Design choices:
  - We load the model lazily (only when first needed) so importing this
    module is cheap.
  - Batch inference (default 16) — much faster than one-at-a-time on
    long stories, but small enough not to blow up 16GB RAM.
  - We return ALL 7 label scores per sentence, not just the top one.
    Phase 3 uses the full distribution to compute "emotional intensity".
  - On Apple Silicon we try MPS (GPU acceleration). Falls back to CPU
    if torch wasn't built with MPS support.
"""

from __future__ import annotations

from pathlib import Path
import gc
import json
import os
from typing import Iterable

import torch
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# The labels the model outputs (in the order the model emits them).
# We hardcode them here so downstream code never has to dig into the
# config to know which index means what.
EMOTION_LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]


def _pick_device() -> str:
    """
    Choose the best available compute device.
    On M-series Macs: 'mps' (GPU via Metal). On CUDA boxes: 'cuda'.
    Otherwise: 'cpu'.
    """
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class EmotionAnalyzer:
    """
    Wraps the HuggingFace emotion classifier. Use as a context manager
    or call .unload() when done to free memory before loading the
    summarizer (matters on 16GB Macs).
    """

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self.device = device or _pick_device()
        self._model = None
        self._tokenizer = None

    # ----- model lifecycle -----

    def load(self) -> None:
        """Load tokenizer + model into memory. First call downloads ~330MB."""
        if self._model is not None:
            return  # already loaded

        print(f"[emotion] Loading {self.model_name} on device={self.device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self._model.to(self.device)
        self._model.eval()  # we never train this — disable dropout etc.
        print(f"[emotion] Model loaded ({self._param_count_m():.0f}M params).")

    def unload(self) -> None:
        """Free model memory. Critical before loading the summarizer."""
        self._model = None
        self._tokenizer = None
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            # MPS cache clearing — only some torch versions have this
            if hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()

    def _param_count_m(self) -> float:
        return sum(p.numel() for p in self._model.parameters()) / 1e6

    # ----- inference -----

    @torch.inference_mode()
    def classify_batch(self, sentences: list[str]) -> list[dict]:
        """
        Classify a batch of sentences. Returns one dict per sentence:

            {
              "top_label": "joy",
              "top_score": 0.87,
              "scores": {"anger": 0.01, "disgust": 0.00, ...all 7...},
            }
        """
        if self._model is None:
            self.load()

        # Tokenize — pad to longest in batch, truncate to model max (512).
        # Most book sentences are well under 100 tokens; the truncation
        # safety-net is for occasional run-on sentences.
        encoded = self._tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        logits = self._model(**encoded).logits  # (batch, 7)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for row in probs:
            scores = {label: float(row[i]) for i, label in enumerate(EMOTION_LABELS)}
            top_label = max(scores, key=scores.get)
            results.append({
                "top_label": top_label,
                "top_score": scores[top_label],
                "scores": scores,
            })
        return results

    def classify_sentences(
        self,
        sentences: list[str],
        batch_size: int = 16,
        show_progress: bool = True,
    ) -> list[dict]:
        """
        Classify a long list of sentences in batches. Returns one dict
        per input sentence, in the same order.
        """
        results: list[dict] = []
        # range(0, n, batch_size) — standard mini-batch iteration
        iterator = range(0, len(sentences), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Emotion analysis", unit="batch")

        for start in iterator:
            batch = sentences[start : start + batch_size]
            results.extend(self.classify_batch(batch))
        return results

    # ----- context manager sugar -----

    def __enter__(self) -> "EmotionAnalyzer":
        self.load()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unload()


# --------------------------------------------------------------------
# Convenience functions — most notebooks will use these, not the class
# --------------------------------------------------------------------

def analyze_segmented_file(
    segmented_path: str | Path,
    output_path: str | Path | None = None,
    batch_size: int = 16,
) -> dict:
    """
    Run emotion analysis on the JSON output of Phase 1's segmenter.

    Reads:  {"sentences": [...], "paragraphs": [...], "stats": {...}}
    Writes: same JSON + per-sentence emotion tags. Structure:

        {
          "sentences":       [...],
          "paragraphs":      [...],
          "stats":           {...},
          "sentence_emotions": [
              {"text": "...", "top_label": "...", "top_score": ..., "scores": {...}},
              ...
          ],
        }

    If output_path is given, also saves the result to disk.
    """
    segmented_path = Path(segmented_path)
    with segmented_path.open(encoding="utf-8") as f:
        data = json.load(f)

    sentences = data["sentences"]
    print(f"Analyzing {len(sentences)} sentences from {segmented_path.name}...")

    with EmotionAnalyzer() as analyzer:
        emotion_results = analyzer.classify_sentences(
            sentences, batch_size=batch_size
        )

    # Pair each result with its source sentence
    data["sentence_emotions"] = [
        {"text": sent, **result}
        for sent, result in zip(sentences, emotion_results)
    ]

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Saved -> {output_path}")

    return data


def emotion_distribution(sentence_emotions: list[dict]) -> dict[str, float]:
    """
    Roll up per-sentence labels into a story-level distribution.
    Returns: {emotion_label: fraction_of_sentences}.
    """
    if not sentence_emotions:
        return {label: 0.0 for label in EMOTION_LABELS}

    counts = {label: 0 for label in EMOTION_LABELS}
    for s in sentence_emotions:
        counts[s["top_label"]] += 1

    total = len(sentence_emotions)
    return {label: counts[label] / total for label in EMOTION_LABELS}
