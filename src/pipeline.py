"""
pipeline.py
-----------
End-to-end orchestrator. Wraps every phase (1 through 5+polish) behind
a single entry point so you can produce an audiobook from a raw .txt
in one command.

Usage as a library:
    from src.pipeline import run_pipeline
    result = run_pipeline("my_story.txt", output_dir="outputs/my_story")

Usage as a CLI:
    python -m src.pipeline my_story.txt
    python -m src.pipeline my_story.txt --no-tts          # stop after Phase 3
    python -m src.pipeline my_story.txt --no-bark         # XTTS-only
    python -m src.pipeline my_story.txt --title "My Story" --author "J. Doe"

Phases the pipeline runs (in order):
    1. Load + segment the input .txt
    2. Run emotion analysis (Phase 2 model)
    3. Build emotion-aware summary (Phase 3 model)
    4. Optionally: synthesize TTS for every summary segment (needs GPU)
    5. Optionally: assemble final MP3 + SRT, embed cover art

Each phase writes its intermediate output to disk and skips itself on
rerun if the output is already there. This means you can:
    - run phases 1-3 on your Mac (CPU is fine, ~5 min)
    - sync the project to Colab
    - run phases 4-5 there (needs GPU)
    - sync back, run again with --polish-only to add cover art
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

# All other src imports are deferred to inside run_pipeline() so that
# `python -m src.pipeline --help` doesn't pay the cost of loading
# torch / transformers / pydub just to print a help message.


# --------------------------------------------------------------------
# Result type
# --------------------------------------------------------------------

@dataclass
class PipelineResult:
    """What run_pipeline returns. Paths may be None if a phase was skipped."""
    story_key: str
    segmented_json: Path
    analyzed_json: Path | None = None
    summary_json: Path | None = None
    intensity_png: Path | None = None
    arc_png: Path | None = None
    mp3_path: Path | None = None
    srt_path: Path | None = None
    phases_run: list[str] = field(default_factory=list)
    phases_skipped: list[str] = field(default_factory=list)


# --------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------

def run_pipeline(
    story_path: str | Path,
    output_dir: str | Path,
    *,
    story_key: str | None = None,
    section_start: str | None = None,
    section_end: str | None = None,
    run_tts: bool = True,
    use_bark: bool = True,
    title: str | None = None,
    author: str | None = None,
    embed_cover: bool = True,
    force_rerun: bool = False,
) -> PipelineResult:
    """
    Run the full pipeline on a single story file.

    Parameters
    ----------
    story_path
        Path to the input .txt file.
    output_dir
        Where to write all intermediate JSON, plots, WAVs, and the final
        MP3+SRT. Created if missing.
    story_key
        Identifier used for output filenames. Defaults to the story
        file's stem (e.g. "my_story.txt" -> "my_story").
    section_start, section_end
        If the file is an anthology, these are passed through to
        text_loader.load_story to extract a single story section.
    run_tts
        Whether to do Phase 4 (TTS) and Phase 5 (assembly). Set False
        to stop after Phase 3 (useful for CPU-only environments).
    use_bark
        Whether to route anchor segments through Bark (vs XTTS-only).
        Ignored if run_tts is False.
    title, author
        Used for ID3 tags + filename. Defaults: story_key as title,
        unknown author.
    embed_cover
        Whether to embed the emotion-arc PNG as MP3 cover art. Only
        kicks in if run_tts is True (no MP3 to tag otherwise).
    force_rerun
        If True, ignore caches and redo every phase. Default False keeps
        reruns fast.

    Returns
    -------
    PipelineResult with paths to every output produced.
    """
    # Local imports — keeps `--help` fast
    from . import (
        text_loader,
        segmenter,
        emotion_analyzer,
        emotion_visualizer,
        summarizer,
        audio_assembler,
    )

    story_path = Path(story_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not story_path.exists():
        raise FileNotFoundError(f"Story file not found: {story_path}")

    if story_key is None:
        story_key = story_path.stem

    if title is None:
        # Convert "the_necklace" -> "The Necklace" for display purposes
        title = story_key.replace("_", " ").title()

    result = PipelineResult(
        story_key=story_key,
        segmented_json=output_dir / f"{story_key}_segmented.json",
    )

    print(f"\n=== Pipeline: {story_key} ===")
    print(f"  input:  {story_path}")
    print(f"  output: {output_dir}")

    # ----- Phase 1: segmentation -----
    seg_path = result.segmented_json
    if seg_path.exists() and not force_rerun:
        print(f"  [skip] Phase 1 (already segmented)")
        result.phases_skipped.append("segment")
    else:
        print(f"  [run]  Phase 1: load + segment")
        text = text_loader.load_story(
            story_path,
            section_start=section_start,
            section_end=section_end,
        )
        segs = segmenter.segment_story(text)
        with seg_path.open("w", encoding="utf-8") as f:
            json.dump(segs, f, ensure_ascii=False, indent=2)
        print(f"         -> {segs['stats']['num_sentences']} sentences, "
              f"{segs['stats']['num_paragraphs']} paragraphs")
        result.phases_run.append("segment")

    # ----- Phase 2: emotion analysis -----
    ana_path = output_dir / f"{story_key}_analyzed.json"
    result.analyzed_json = ana_path
    if ana_path.exists() and not force_rerun:
        print(f"  [skip] Phase 2 (already analyzed)")
        result.phases_skipped.append("emotion")
    else:
        print(f"  [run]  Phase 2: emotion analysis")
        emotion_analyzer.analyze_segmented_file(seg_path, output_path=ana_path)
        result.phases_run.append("emotion")

    # ----- Phase 2.5: emotional arc plots -----
    intensity_png = output_dir / f"{story_key}_intensity.png"
    arc_png = output_dir / f"{story_key}_arc.png"
    result.intensity_png = intensity_png
    result.arc_png = arc_png
    if intensity_png.exists() and arc_png.exists() and not force_rerun:
        print(f"  [skip] Phase 2.5 (plots cached)")
        result.phases_skipped.append("plots")
    else:
        print(f"  [run]  Phase 2.5: emotional-arc plots")
        with ana_path.open(encoding="utf-8") as f:
            ana_data = json.load(f)
        # matplotlib import is heavy; defer until we actually plot.
        # We import inside the visualizer module too, but keep it lazy.
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend — no GUI needed
        emotion_visualizer.plot_intensity_arc(
            ana_data["sentence_emotions"],
            title=f"Emotional Intensity — {title}",
            save_path=intensity_png,
        )
        emotion_visualizer.plot_emotion_arc(
            ana_data["sentence_emotions"],
            title=f"Emotion Distribution — {title}",
            save_path=arc_png,
        )
        result.phases_run.append("plots")

    # ----- Phase 3: emotion-aware summarization -----
    sum_path = output_dir / f"{story_key}_summary.json"
    result.summary_json = sum_path
    if sum_path.exists() and not force_rerun:
        print(f"  [skip] Phase 3 (already summarized)")
        result.phases_skipped.append("summary")
    else:
        print(f"  [run]  Phase 3: emotion-aware summarization")
        summarizer.summarize_analyzed_file(ana_path, output_path=sum_path)
        result.phases_run.append("summary")

    if not run_tts:
        print("\n[stop] run_tts=False — skipping Phases 4 and 5.")
        return result

    # ----- Phase 4: TTS -----
    # tts_engine has heavy deps (Coqui TTS, transformers/Bark) so we
    # only import when actually running.
    from . import tts_engine

    wavs_dir = output_dir / f"{story_key}_wavs"
    expected_count = _count_summary_segments(sum_path)
    existing_wavs = sorted(wavs_dir.glob("*.wav")) if wavs_dir.exists() else []

    if (len(existing_wavs) == expected_count
            and expected_count > 0
            and not force_rerun):
        print(f"  [skip] Phase 4 (all {expected_count} WAVs cached)")
        result.phases_skipped.append("tts")
        tts_results = _reload_tts_results(sum_path, wavs_dir)
    else:
        print(f"  [run]  Phase 4: TTS ({expected_count} segments)")
        with sum_path.open(encoding="utf-8") as f:
            sum_data = json.load(f)
        tts_results = tts_engine.synthesize_segments(
            sum_data["summary"]["segments"],
            output_dir=wavs_dir,
            use_bark_for_anchors=use_bark,
        )
        result.phases_run.append("tts")

    # ----- Phase 5: assembly + polish -----
    print(f"  [run]  Phase 5: assemble MP3 + SRT")
    assembly = audio_assembler.assemble_audiobook(
        tts_results,
        output_dir=output_dir,
        title=story_key,
    )
    result.mp3_path = assembly.mp3_path
    result.srt_path = assembly.srt_path
    result.phases_run.append("assemble")

    print(f"         -> MP3: {assembly.mp3_path.name} "
          f"({assembly.total_duration_s:.1f}s)")
    print(f"         -> SRT: {assembly.srt_path.name}")

    # ----- Phase 6: cover art + ID3 -----
    if embed_cover:
        print(f"  [run]  Phase 6: embed cover art + ID3 tags")
        # Prefer the intensity PNG as cover — it's the most visually
        # interesting plot (line + colored markers) compared to the
        # stacked-area arc which can look muddy at thumbnail size.
        cover = intensity_png if intensity_png.exists() else arc_png
        try:
            audio_assembler.embed_cover_art(
                assembly.mp3_path,
                cover,
                title=title,
                artist=author or "Unknown",
                album="Emotional Audiobook",
            )
            print(f"         -> cover: {cover.name}")
            result.phases_run.append("cover_art")
        except Exception as e:
            # Cover art is cosmetic — never let it fail the whole pipeline.
            print(f"  [WARN] Cover art embedding failed: {e}")

    print(f"\n✅ Pipeline complete. "
          f"Ran: {result.phases_run}. Skipped: {result.phases_skipped}.")
    return result


# --------------------------------------------------------------------
# Internal helpers
# --------------------------------------------------------------------

def _count_summary_segments(summary_path: Path) -> int:
    """How many segments are in a summary JSON? Used to detect whether
    a previous TTS run completed for all of them."""
    with summary_path.open(encoding="utf-8") as f:
        data = json.load(f)
    return len(data["summary"]["segments"])


def _reload_tts_results(summary_path: Path, wavs_dir: Path) -> list:
    """
    Rebuild a list of TTSResult objects from cached WAVs on disk.
    Used when Phase 4 is skipped — we still need TTSResult objects for
    Phase 5 to read.
    """
    from . import tts_engine
    from scipy.io import wavfile

    with summary_path.open(encoding="utf-8") as f:
        data = json.load(f)
    segments = data["summary"]["segments"]

    # WAVs are named "<NNNN>_<slug>.wav" — sort numerically by index prefix.
    wavs = sorted(wavs_dir.glob("*.wav"))
    if len(wavs) != len(segments):
        raise RuntimeError(
            f"Cached WAV count ({len(wavs)}) doesn't match summary "
            f"segment count ({len(segments)}). Rerun with --force-rerun."
        )

    results = []
    for seg, wav_path in zip(segments, wavs):
        # Get duration cheaply by reading WAV header (scipy returns full
        # data, but we only need shape — first element after rate).
        sr, data_arr = wavfile.read(wav_path)
        duration = len(data_arr) / sr
        emotion_label = seg["emotion"]["top_label"] if seg.get("emotion") else None
        # We don't know which engine produced each cached WAV (filename
        # doesn't encode it). For anchor segments assume bark, bridges xtts.
        # Worst case for getting this wrong: Phase 5's trim_bark_tails
        # runs on an XTTS clip (harmless — XTTS clips don't have trailing
        # noise to trim, so nothing changes) or doesn't run on a Bark clip
        # (you'd see a slightly long anchor segment). Acceptable.
        engine = "bark" if seg["kind"] == "anchor" else "xtts"
        results.append(tts_engine.TTSResult(
            text=seg["text"],
            kind=seg["kind"],
            engine=engine,
            emotion=emotion_label,
            wav_path=wav_path,
            duration_s=duration,
            sample_rate=sr,
        ))
    return results


# --------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Convert a story .txt into an emotion-aware audiobook.",
    )
    p.add_argument("story_path", type=str, help="Input .txt file")
    p.add_argument(
        "-o", "--output-dir", type=str, default=None,
        help="Output directory. Defaults to outputs/<story_stem>/."
    )
    p.add_argument(
        "--story-key", type=str, default=None,
        help="Identifier used in output filenames. Defaults to the input file's stem."
    )
    p.add_argument(
        "--section-start", type=str, default=None,
        help="If the file is an anthology, marker for the section start."
    )
    p.add_argument(
        "--section-end", type=str, default=None,
        help="If the file is an anthology, marker for the section end."
    )
    p.add_argument(
        "--no-tts", action="store_true",
        help="Stop after Phase 3 (no TTS, no MP3). Useful for CPU-only environments."
    )
    p.add_argument(
        "--no-bark", action="store_true",
        help="Use XTTS for every segment (faster, less expressive). Ignored with --no-tts."
    )
    p.add_argument(
        "--title", type=str, default=None,
        help="Title for ID3 tags. Defaults to the story key, title-cased."
    )
    p.add_argument(
        "--author", type=str, default=None,
        help="Author/artist for ID3 tags."
    )
    p.add_argument(
        "--no-cover", action="store_true",
        help="Skip embedding the emotion-arc PNG as MP3 cover art."
    )
    p.add_argument(
        "--force-rerun", action="store_true",
        help="Ignore caches; redo every phase from scratch."
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    story_path = Path(args.story_path)
    story_key = args.story_key or story_path.stem
    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / story_key

    try:
        run_pipeline(
            story_path=story_path,
            output_dir=output_dir,
            story_key=story_key,
            section_start=args.section_start,
            section_end=args.section_end,
            run_tts=not args.no_tts,
            use_bark=not args.no_bark,
            title=args.title,
            author=args.author,
            embed_cover=not args.no_cover,
            force_rerun=args.force_rerun,
        )
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}", file=sys.stderr)
        # Re-raise so tracebacks are useful when debugging
        raise

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
