"""
tts_engine.py
-------------
Dual-engine TTS for the emotion-aware audiobook system.

Strategy:
  - XTTS-v2 (Coqui) synthesizes every segment as the BASELINE narrator.
    Fast (~1-3s/segment on T4 GPU), consistent voice, decent prosody.
  - Bark (Suno) re-synthesizes ANCHOR segments only. Slower
    (~10-30s/segment) but produces dramatically more emotional output
    when nudged with [sighs], [gasps], etc. cues.

The result: the audiobook has consistent narration throughout, with
emotional peaks given to a more expressive model. This is roughly how
human audiobook narrators work — they read most of the text in a
neutral register and only "perform" at the dramatic moments.

GPU REQUIRED. CPU-only is technically possible but practically too slow
(minutes per sentence for Bark). Use Colab's free T4 tier or equivalent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import gc
import hashlib
import time
from typing import Optional

import numpy as np

# Lazy imports — we don't want module import to fail just because torch
# isn't installed (e.g., during testing on machines without it).
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# --------------------------------------------------------------------
# Emotion → Bark cue mapping
# --------------------------------------------------------------------
# Bark responds to bracketed cues in the input text. We don't go
# overboard — just one well-placed cue per anchor, since piling them on
# tends to confuse the model.
EMOTION_BARK_CUES = {
    "anger":    "[sighs deeply] ",       # frustrated exhale before
    "disgust":  "[sighs] ",
    "fear":     "[gasps] ",
    "joy":      "[laughs softly] ",
    "neutral":  "",                       # no cue
    "sadness":  "[sighs sadly] ",
    "surprise": "[gasps] ",
}

# Default voice preset names. These can be overridden per-engine.
DEFAULT_BARK_SPEAKER = "v2/en_speaker_6"   # steady male narrator
DEFAULT_XTTS_LANGUAGE = "en"
# XTTS-v2 needs a 6-30 second reference clip to clone a voice. We use
# a public-domain narrator sample bundled with TTS package by default.
DEFAULT_XTTS_SPEAKER = "Andrew Chipper"


# --------------------------------------------------------------------
# XTTS-v2 wrapper
# --------------------------------------------------------------------

class XTTSEngine:
    """
    Wraps Coqui XTTS-v2.

    Notes on quirks:
    - First load downloads ~1.8GB of model files. Cached after.
    - Output is 24kHz mono float32 numpy array.
    - We deliberately don't pass a speaker_wav by default — we use the
      built-in speakers since they're already at the right sample rate
      and don't need normalization.
    """

    SAMPLE_RATE = 24000

    def __init__(
        self,
        model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        speaker: str = DEFAULT_XTTS_SPEAKER,
        language: str = DEFAULT_XTTS_LANGUAGE,
        device: str | None = None,
    ):
        if not _TORCH_AVAILABLE:
            raise RuntimeError(
                "PyTorch is required. Install it via "
                "`pip install torch` (or use the project's requirements.txt)."
            )
        self.model_name = model_name
        self.speaker = speaker
        self.language = language
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None

    def load(self) -> None:
        if self._model is not None:
            return
        # Lazy import — TTS pulls a lot of deps; only load when actually needed.
        from TTS.api import TTS

        # XTTS-v2 has a license-acceptance prompt the first time it loads.
        # Setting this env var auto-accepts (Coqui's non-commercial license).
        import os
        os.environ.setdefault("COQUI_TOS_AGREED", "1")

        print(f"[xtts] Loading {self.model_name} on {self.device}...")
        # progress_bar=False keeps notebook output clean
        self._model = TTS(self.model_name, progress_bar=False).to(self.device)
        print("[xtts] Loaded.")

    def unload(self) -> None:
        self._model = None
        gc.collect()
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def synthesize(
        self,
        text: str,
        emotion: str | None = None,
    ) -> np.ndarray:
        """
        Generate a float32 mono waveform at 24kHz.

        XTTS doesn't take an emotion parameter directly. The `emotion`
        argument is accepted for API symmetry with the Bark engine, but
        is only used here to subtly vary temperature: emotional segments
        get slightly higher temperature for more prosodic variation.
        """
        if self._model is None:
            self.load()

        # Temperature tweak — XTTS's default is 0.65. Bump for emotional
        # peaks so the prosody varies more. Capped to avoid garbled output.
        temperature = 0.65
        if emotion and emotion != "neutral":
            temperature = 0.80

        # tts() returns a list of floats (waveform). We coerce to numpy.
        wav = self._model.tts(
            text=text,
            speaker=self.speaker,
            language=self.language,
            temperature=temperature,
        )
        return np.asarray(wav, dtype=np.float32)


# --------------------------------------------------------------------
# Bark wrapper
# --------------------------------------------------------------------

class BarkEngine:
    """
    Wraps Suno's Bark model via the transformers library.

    Notes on quirks:
    - Bark is slow: ~10-30s per short sentence on a T4 GPU.
    - Output is 24kHz mono.
    - Voice consistency across calls is so-so even with a pinned speaker
      preset. We pin one anyway; it helps but isn't perfect.
    - Sometimes Bark hallucinates extra audio after the spoken text
      (humming, background noise, etc.). Not much we can do without
      post-processing; the audio_assembler module trims trailing silence.
    """

    SAMPLE_RATE = 24000

    def __init__(
        self,
        speaker: str = DEFAULT_BARK_SPEAKER,
        device: str | None = None,
    ):
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for Bark.")
        self.speaker = speaker
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoProcessor, BarkModel

        print(f"[bark] Loading suno/bark on {self.device}...")
        self._processor = AutoProcessor.from_pretrained("suno/bark")
        self._model = BarkModel.from_pretrained("suno/bark").to(self.device)
        # Bark supports half-precision on GPU — uses half the VRAM,
        # comparable quality. Skip on CPU (fp16 is slow there).
        if self.device == "cuda":
            self._model = self._model.to(torch.float16)
        self._model.eval()
        print("[bark] Loaded.")

    def unload(self) -> None:
        self._model = None
        self._processor = None
        gc.collect()
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def synthesize(
        self,
        text: str,
        emotion: str | None = None,
    ) -> np.ndarray:
        """
        Generate audio for `text`, prefixed with an emotion cue if
        `emotion` is one of the supported labels.
        """
        if self._model is None:
            self.load()

        cue = EMOTION_BARK_CUES.get(emotion, "") if emotion else ""
        prompted_text = cue + text

        inputs = self._processor(
            prompted_text,
            voice_preset=self.speaker,
            return_tensors="pt",
        ).to(self.device)

        audio = self._model.generate(**inputs)
        # Bark returns shape (1, samples). Squeeze and convert to numpy.
        wav = audio.cpu().float().numpy().squeeze()
        return wav.astype(np.float32)


# --------------------------------------------------------------------
# Dual-engine orchestrator
# --------------------------------------------------------------------

@dataclass
class TTSResult:
    """One synthesized segment + metadata."""
    text: str
    kind: str          # "anchor" or "bridge"
    engine: str        # "xtts" or "bark"
    emotion: str | None
    wav_path: Path     # disk path to the .wav file
    duration_s: float  # length in seconds
    sample_rate: int   # always 24000 for us


def _safe_filename(text: str, max_len: int = 40) -> str:
    """
    Build a short, deterministic filename from a text snippet.
    We hash the full text so different segments never collide, and
    include a slug for human readability.
    """
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    slug = "".join(c if c.isalnum() else "_" for c in text[:max_len]).strip("_")
    return f"{slug}_{digest}"


def synthesize_segments(
    segments: list[dict],
    output_dir: str | Path,
    *,
    xtts: XTTSEngine | None = None,
    bark: BarkEngine | None = None,
    use_bark_for_anchors: bool = True,
) -> list[TTSResult]:
    """
    Synthesize each segment with the appropriate engine.

    Routing:
        - kind == "anchor" AND use_bark_for_anchors AND bark is loadable
          -> Bark (emotional). Falls back to XTTS on failure.
        - everything else -> XTTS.

    Audio is written to `output_dir/<segment_index>_<slug>.wav` so each
    segment can be played in isolation and the order is preserved by
    sorting filenames.

    Parameters
    ----------
    segments : list[dict]
        From summarizer.segments_to_json — each has text, kind, emotion.
    output_dir : Path
        Directory to write .wav files into.
    xtts, bark : optional preloaded engines
        If None, we instantiate them as needed. Passing a loaded engine
        lets you reuse it across multiple stories.
    use_bark_for_anchors : bool
        Set False to skip Bark entirely (XTTS-only mode). Useful for
        quick iteration or if Bark setup is failing.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import — keep top-level import light
    from scipy.io import wavfile

    # We load engines lazily as needed
    owns_xtts = xtts is None
    owns_bark = (bark is None) and use_bark_for_anchors

    results: list[TTSResult] = []

    try:
        for i, seg in enumerate(segments):
            text = seg["text"]
            kind = seg["kind"]
            emotion_label = (
                seg["emotion"]["top_label"] if seg.get("emotion") else None
            )

            # Decide engine for this segment
            use_bark = (
                use_bark_for_anchors
                and kind == "anchor"
                and emotion_label not in (None, "neutral")
            )
            intended_engine = "bark" if use_bark else "xtts"

            # ----- per-segment cache check -----
            # Each WAV gets a sidecar .meta.json recording the engine that
            # produced it. We can skip regeneration only if the cached
            # file's engine matches what we'd use THIS run. Flipping
            # USE_BARK then re-runs anchor segments through Bark but keeps
            # every XTTS bridge cached.
            filename = f"{i:04d}_{_safe_filename(text)}.wav"
            wav_path = output_dir / filename
            meta_path = wav_path.with_suffix(".meta.json")

            if wav_path.exists() and meta_path.exists():
                try:
                    import json as _json
                    cached_meta = _json.loads(meta_path.read_text())
                    if cached_meta.get("engine") == intended_engine:
                        # Cache hit. Reuse the WAV — measure duration so
                        # downstream code (assembler, SRT) is consistent.
                        cached_sr, cached_data = wavfile.read(wav_path)
                        duration = len(cached_data) / cached_sr
                        print(
                            f"  [{i:3d}] {intended_engine:6s} (cached) "
                            f"{text[:60]!r}"
                        )
                        results.append(TTSResult(
                            text=text,
                            kind=kind,
                            engine=intended_engine,
                            emotion=emotion_label,
                            wav_path=wav_path,
                            duration_s=duration,
                            sample_rate=cached_sr,
                        ))
                        continue
                except Exception:
                    # Bad metadata file — fall through to regeneration
                    pass

            # ----- need to synthesize -----
            # Lazy-load engines only when we actually need them. This
            # matters when EVERY segment is cached — we never load the
            # 1.8GB XTTS model just to find out it wasn't needed.
            if xtts is None:
                xtts = XTTSEngine()
                xtts.load()

            engine_used = intended_engine
            wav: np.ndarray | None = None

            if use_bark:
                if bark is None:
                    try:
                        bark = BarkEngine()
                        bark.load()
                    except Exception as e:
                        print(f"  [WARN] Bark failed to load: {e}")
                        print(f"  Falling back to XTTS for all segments.")
                        use_bark_for_anchors = False
                        use_bark = False
                        engine_used = "xtts"

                if use_bark:
                    try:
                        t0 = time.time()
                        wav = bark.synthesize(text, emotion=emotion_label)
                        print(
                            f"  [{i:3d}] bark   ({emotion_label:8s}) "
                            f"{time.time() - t0:5.1f}s  "
                            f"{text[:60]!r}"
                        )
                    except Exception as e:
                        # Bark sometimes OOMs or hangs on long inputs.
                        # Fall back to XTTS for this one segment.
                        print(
                            f"  [{i:3d}] bark FAILED ({e!r}) — "
                            f"falling back to XTTS for this segment."
                        )
                        engine_used = "xtts"
                        wav = None  # force XTTS path below

            if wav is None:
                t0 = time.time()
                wav = xtts.synthesize(text, emotion=emotion_label)
                # Show progress consistently even when Bark not used
                if not use_bark:
                    print(
                        f"  [{i:3d}] xtts   ({(emotion_label or '-'):8s}) "
                        f"{time.time() - t0:5.1f}s  "
                        f"{text[:60]!r}"
                    )

            # Normalize and save. XTTS sometimes overshoots [-1, 1] and
            # clips on disk; we rescale anything that exceeds the bounds.
            peak = float(np.max(np.abs(wav))) if wav.size else 0.0
            if peak > 1.0:
                wav = wav / peak

            # scipy expects int16 for WAV. Convert from float32 [-1, 1].
            wav_int16 = (wav * 32767.0).astype(np.int16)
            wavfile.write(wav_path, XTTSEngine.SAMPLE_RATE, wav_int16)

            # Write the metadata sidecar so the next run knows which
            # engine produced this WAV.
            import json as _json
            meta_path.write_text(_json.dumps({
                "engine": engine_used,
                "kind": kind,
                "emotion": emotion_label,
                "text_preview": text[:200],
            }))

            duration = len(wav) / XTTSEngine.SAMPLE_RATE
            results.append(TTSResult(
                text=text,
                kind=kind,
                engine=engine_used,
                emotion=emotion_label,
                wav_path=wav_path,
                duration_s=duration,
                sample_rate=XTTSEngine.SAMPLE_RATE,
            ))

    finally:
        if owns_xtts and xtts is not None:
            xtts.unload()
        if owns_bark and bark is not None:
            bark.unload()

    return results
