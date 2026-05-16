"""
audio_assembler.py
------------------
Stitches per-segment WAVs into a single MP3 audiobook, with a synced
.srt subtitle file so any player (VLC, QuickTime, browser <video>) can
display the current text as it plays.

Why SRT?
  - Universally supported. Drop the .srt next to the .mp3 in the same
    folder with the same basename; VLC auto-loads it.
  - Plain text, human-readable, easy to debug.
  - One subtitle entry per segment, so anchors and bridges each get
    their own visible chunk.

Design choices:
  - Inter-segment silence: 350ms between bridges, 600ms after anchors
    (the dramatic pause). Tunable via `silence_ms_*`.
  - We use pydub's AudioSegment for stitching because it's pure-Python
    on top of ffmpeg and handles MP3 export cleanly.
  - Subtitle timestamps are computed from cumulative audio durations,
    not from the per-segment "duration_s" field — measuring after
    stitching catches silence drift exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pydub import AudioSegment

from .tts_engine import TTSResult


# --------------------------------------------------------------------
# SRT timestamp formatting
# --------------------------------------------------------------------

def _ms_to_srt_timestamp(ms: int) -> str:
    """
    Convert milliseconds to SRT's HH:MM:SS,mmm format.

    SRT uses a comma (not period) before the millisecond field — this
    bites people coming from ffmpeg or VTT. Get it wrong and most
    players silently fail to parse the file.
    """
    hours = ms // 3_600_000
    ms %= 3_600_000
    minutes = ms // 60_000
    ms %= 60_000
    seconds = ms // 1_000
    ms %= 1_000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


# --------------------------------------------------------------------
# Stitching
# --------------------------------------------------------------------

@dataclass
class AssemblyResult:
    """Where everything went and how long each part took."""
    mp3_path: Path
    srt_path: Path
    total_duration_s: float
    num_segments: int


def assemble_audiobook(
    tts_results: list[TTSResult],
    output_dir: str | Path,
    title: str,
    *,
    silence_ms_bridge: int = 350,
    silence_ms_anchor: int = 600,
    mp3_bitrate: str = "128k",
    trim_bark_tails: bool = True,
) -> AssemblyResult:
    """
    Stitch per-segment WAVs into a single MP3 + matching SRT.

    Parameters
    ----------
    tts_results
        From tts_engine.synthesize_segments, in playback order.
    output_dir
        Where the .mp3 and .srt land. Created if missing.
    title
        Basename for the output files: "<title>.mp3" and "<title>.srt".
    silence_ms_bridge, silence_ms_anchor
        Silence inserted AFTER a segment based on its kind. Anchors get
        a longer pause to let the emotional moment land before moving on.
    mp3_bitrate
        128k is plenty for narration. Bump to 192k if you care about
        Bark's non-verbal sounds being crisp.
    trim_bark_tails
        If True, run _trim_trailing_silence on every Bark-engine segment
        before stitching. Bark sometimes adds 1-3 sec of trailing noise
        (humming, breath, hallucinated tones) that this cleans up.
        XTTS segments are left alone — they don't have this problem.

    Returns
    -------
    AssemblyResult with file paths and total length.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not tts_results:
        raise ValueError("No TTS results to assemble.")

    # Build the master audio one segment at a time, recording where each
    # segment starts/ends in the final timeline. We track times in
    # milliseconds (pydub's native unit) and convert to SRT at the end.
    master = AudioSegment.silent(duration=0)
    srt_entries: list[tuple[int, int, int, str]] = []  # (index, start_ms, end_ms, text)

    for i, result in enumerate(tts_results, start=1):
        # The segment audio itself
        segment = AudioSegment.from_wav(str(result.wav_path))

        # Optionally trim trailing noise from Bark segments. We do this
        # before computing start_ms/end_ms so the SRT timestamps line up
        # with the actual (trimmed) audio, not the original.
        if trim_bark_tails and result.engine == "bark":
            original_len = len(segment)
            segment = _trim_trailing_silence(segment)
            trimmed = original_len - len(segment)
            if trimmed > 0:
                # Only log when we actually trim something — verbose mode
                print(f"  [trim] segment {i}: dropped {trimmed}ms of trailing noise")

        # Record its position in the master timeline BEFORE appending
        start_ms = len(master)
        master = master + segment
        end_ms = len(master)

        # Inter-segment silence — appended AFTER, not counted in the
        # segment's subtitle window
        silence_ms = (
            silence_ms_anchor if result.kind == "anchor"
            else silence_ms_bridge
        )
        master = master + AudioSegment.silent(duration=silence_ms)

        srt_entries.append((i, start_ms, end_ms, result.text))

    # Export MP3
    safe_title = "".join(c if c.isalnum() or c in "-_" else "_" for c in title)
    mp3_path = output_dir / f"{safe_title}.mp3"
    master.export(mp3_path, format="mp3", bitrate=mp3_bitrate)

    # Export SRT
    srt_path = output_dir / f"{safe_title}.srt"
    srt_lines = []
    for idx, start_ms, end_ms, text in srt_entries:
        srt_lines.append(str(idx))
        srt_lines.append(
            f"{_ms_to_srt_timestamp(start_ms)} --> {_ms_to_srt_timestamp(end_ms)}"
        )
        # SRT does support basic line breaks within a cue. Long texts
        # are easier to read if wrapped at ~80 chars.
        srt_lines.append(_wrap_for_srt(text))
        srt_lines.append("")  # blank line between cues
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8")

    total_duration_s = len(master) / 1000.0

    return AssemblyResult(
        mp3_path=mp3_path,
        srt_path=srt_path,
        total_duration_s=total_duration_s,
        num_segments=len(tts_results),
    )


def _wrap_for_srt(text: str, max_line_chars: int = 80) -> str:
    """
    Soft-wrap long text into ≤max_line_chars-wide lines for readable
    subtitles. Uses simple word-boundary wrapping; doesn't break inside
    words.
    """
    words = text.split()
    if not words:
        return ""
    lines = []
    current = words[0]
    for w in words[1:]:
        if len(current) + 1 + len(w) <= max_line_chars:
            current = current + " " + w
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return "\n".join(lines)


# --------------------------------------------------------------------
# Trailing-silence trimming (for Bark cleanup)
# --------------------------------------------------------------------

def _trim_trailing_silence(
    audio: AudioSegment,
    silence_thresh_db: float = -40.0,
    min_silence_ms: int = 200,
    keep_ms: int = 100,
) -> AudioSegment:
    """
    Trim trailing silence/low-energy noise from the end of a clip.

    Walks backward from the end in small windows. The trim point is the
    last window whose dBFS exceeds `silence_thresh_db`. We then keep
    `keep_ms` of audio after that point so the cut doesn't sound abrupt.

    This exists because Bark sometimes hallucinates 1-3 sec of trailing
    noise (humming, breath sounds, random tones) after the speech ends.
    XTTS doesn't have this problem so we only run this on Bark segments
    by default.

    Parameters
    ----------
    audio
        The pydub AudioSegment to trim.
    silence_thresh_db
        Anything quieter than this is considered silence. -40 dBFS is
        a good default for narration; lower it to be more aggressive.
    min_silence_ms
        Don't bother trimming unless there's at least this much trailing
        silence. Prevents over-trimming clips that legitimately end soft.
    keep_ms
        Keep this much audio after the detected speech-end, so the cut
        doesn't sound like a hard chop.

    Returns
    -------
    A trimmed AudioSegment. If no trailing silence is detected, returns
    the input unchanged.
    """
    if len(audio) == 0:
        return audio

    # Walk backward in 50ms windows. Stop when we find a non-silent one.
    window_ms = 50
    cursor = len(audio)
    while cursor > window_ms:
        window = audio[cursor - window_ms : cursor]
        if window.dBFS > silence_thresh_db:
            # Found the end of real audio. Apply the keep buffer and
            # bail out.
            new_end = min(cursor + keep_ms, len(audio))
            silence_trimmed = len(audio) - new_end
            # Only return trimmed version if we actually saved something
            # meaningful — otherwise return original to avoid jitter on
            # clips that end cleanly already.
            if silence_trimmed >= min_silence_ms:
                return audio[:new_end]
            return audio
        cursor -= window_ms

    # Entire clip is silent. Return as-is rather than empty (downstream
    # code is happier with a short silent segment than a zero-length one).
    return audio


# --------------------------------------------------------------------
# ID3 cover art embedding
# --------------------------------------------------------------------

def embed_cover_art(
    mp3_path: str | Path,
    image_path: str | Path,
    *,
    title: str | None = None,
    artist: str | None = None,
    album: str | None = None,
) -> None:
    """
    Embed an image as MP3 cover art using ID3 tags. Optionally also set
    Title / Artist / Album.

    We use mutagen (pure Python, no ffmpeg gymnastics). The image is
    read as bytes and attached as an APIC frame, which every modern
    player supports (Apple Music, VLC, Spotify-imported tracks, etc.).

    Lazy-imports mutagen so importing this module doesn't fail when
    mutagen isn't installed — only the user of this function pays.

    Parameters
    ----------
    mp3_path
        The .mp3 file to tag. Modified in place.
    image_path
        Image to embed. PNG or JPEG. We detect MIME type from extension.
    title, artist, album
        Optional ID3 text fields. Common audiobook convention is to
        use story title as Title, author as Artist, "Emotional Audiobook"
        as Album.
    """
    # Lazy import — mutagen is in requirements.txt but if a user runs
    # an older venv we'd rather give a clear error than fail at import time.
    try:
        from mutagen.id3 import ID3, ID3NoHeaderError, APIC, TIT2, TPE1, TALB
    except ImportError as e:
        raise RuntimeError(
            "mutagen is required for cover art. "
            "Install with: pip install mutagen"
        ) from e

    mp3_path = Path(mp3_path)
    image_path = Path(image_path)
    if not mp3_path.exists():
        raise FileNotFoundError(f"MP3 not found: {mp3_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_data = image_path.read_bytes()
    # Detect MIME type from extension. mutagen needs this for APIC frames.
    ext = image_path.suffix.lower()
    if ext == ".png":
        mime = "image/png"
    elif ext in (".jpg", ".jpeg"):
        mime = "image/jpeg"
    else:
        raise ValueError(f"Unsupported cover image format: {ext}")

    # Load or create ID3 tag block on the MP3
    try:
        tags = ID3(str(mp3_path))
    except ID3NoHeaderError:
        # File has no ID3 header yet (common for freshly-exported pydub MP3s)
        tags = ID3()

    # Remove any existing cover art so we don't end up with duplicates.
    # APIC frame keys are like "APIC:" or "APIC:Cover".
    tags.delall("APIC")

    tags.add(APIC(
        encoding=3,    # 3 = UTF-8
        mime=mime,
        type=3,        # 3 = "Cover (front)" per ID3 spec
        desc="Cover",
        data=image_data,
    ))

    # Optional text metadata
    if title is not None:
        tags.delall("TIT2")
        tags.add(TIT2(encoding=3, text=title))
    if artist is not None:
        tags.delall("TPE1")
        tags.add(TPE1(encoding=3, text=artist))
    if album is not None:
        tags.delall("TALB")
        tags.add(TALB(encoding=3, text=album))

    tags.save(str(mp3_path))


