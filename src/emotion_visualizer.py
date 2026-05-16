"""
emotion_visualizer.py
---------------------
Plots the emotional arc of a story.

Two complementary views:
  - plot_intensity_arc:   one line, "how emotional is the story right now"
                          (= 1 - P(neutral)), smoothed. Good for seeing
                          the overall narrative shape at a glance.
  - plot_emotion_arc:     stacked-area chart of the 7 emotion probabilities
                          over time. Good for seeing WHICH emotion is
                          dominant at each point.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .emotion_analyzer import EMOTION_LABELS


# Fixed color per emotion so plots are visually consistent across
# different stories and notebooks. Picked so each emotion's color
# matches its "feel" (red=anger, blue=sadness, etc.).
EMOTION_COLORS = {
    "anger":    "#d62728",  # red
    "disgust":  "#8c564b",  # brown
    "fear":     "#9467bd",  # purple
    "joy":      "#ffd92f",  # yellow
    "neutral":  "#bdbdbd",  # grey
    "sadness":  "#1f77b4",  # blue
    "surprise": "#ff7f0e",  # orange
}


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Simple moving-average smoothing. Reduces sentence-to-sentence noise
    so the underlying arc is visible. window=5 is a good default for
    short stories; bump to 10–20 for novels.
    """
    if window <= 1 or len(values) < window:
        return values
    # 'same' mode keeps the output length equal to input length
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="same")


def plot_intensity_arc(
    sentence_emotions: list[dict],
    title: str = "Emotional Intensity Arc",
    smooth_window: int = 5,
    save_path: str | Path | None = None,
):
    """
    Plot story-wide emotional intensity = 1 − P(neutral) per sentence.
    High values = sentence is emotionally charged (regardless of which
    emotion). The dominant emotion at each point colors the marker.
    """
    n = len(sentence_emotions)
    intensities = np.array(
        [1.0 - s["scores"]["neutral"] for s in sentence_emotions]
    )
    smoothed = _smooth(intensities, smooth_window)

    fig, ax = plt.subplots(figsize=(12, 4))

    # Background: smoothed intensity line
    ax.plot(smoothed, color="#333333", linewidth=1.5, label="Intensity (smoothed)")
    ax.fill_between(range(n), smoothed, alpha=0.15, color="#333333")

    # Foreground: scatter, colored by dominant non-neutral emotion
    for i, s in enumerate(sentence_emotions):
        if s["top_label"] == "neutral":
            continue  # skip — those are just baseline
        ax.scatter(
            i, intensities[i],
            color=EMOTION_COLORS.get(s["top_label"], "#000000"),
            s=20, alpha=0.7, edgecolors="none",
        )

    ax.set_xlabel("Sentence index")
    ax.set_ylabel("Emotional intensity (1 − P(neutral))")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    # Build a legend showing only emotions actually present in the story
    present_labels = {s["top_label"] for s in sentence_emotions} - {"neutral"}
    if present_labels:
        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=EMOTION_COLORS[lbl], markersize=8, label=lbl)
            for lbl in sorted(present_labels)
        ]
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved plot -> {save_path}")

    return fig, ax


def plot_emotion_arc(
    sentence_emotions: list[dict],
    title: str = "Emotion Distribution Over Time",
    smooth_window: int = 5,
    save_path: str | Path | None = None,
):
    """
    Stacked-area chart showing all 7 emotion probabilities over time.
    Each band's height is that emotion's smoothed probability at that
    sentence index. Lets you see which emotion dominates where.
    """
    n = len(sentence_emotions)
    if n == 0:
        raise ValueError("No sentences to plot.")

    # Build a (7, n) array of probabilities, then smooth each row
    matrix = np.zeros((len(EMOTION_LABELS), n))
    for j, s in enumerate(sentence_emotions):
        for i, label in enumerate(EMOTION_LABELS):
            matrix[i, j] = s["scores"][label]

    smoothed = np.array([_smooth(row, smooth_window) for row in matrix])

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.stackplot(
        range(n),
        smoothed,
        labels=EMOTION_LABELS,
        colors=[EMOTION_COLORS[l] for l in EMOTION_LABELS],
        alpha=0.85,
    )
    ax.set_xlabel("Sentence index")
    ax.set_ylabel("Emotion probability")
    ax.set_title(title)
    ax.set_xlim(0, n - 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    plt.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"  Saved plot -> {save_path}")

    return fig, ax
