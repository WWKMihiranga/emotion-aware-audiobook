"""
Emotional Audiobook — source package.

Phase 1 modules:
    - story_downloader  : fetch public-domain test stories
    - text_loader       : read story files, strip boilerplate, normalize
    - segmenter         : split stories into sentences and paragraphs

Phase 2 modules:
    - emotion_analyzer  : tag each sentence with emotion + confidence
    - emotion_visualizer: plot the story's emotional arc

Phase 3 modules:
    - summarizer        : emotion-aware "anchor and compress" summary

Phase 4 modules:
    - tts_engine        : XTTS-v2 + Bark dual-engine TTS

Phase 5 modules:
    - audio_assembler   : stitch WAVs into MP3 + matched SRT subtitles

Phase 6 modules:
    - pipeline          : end-to-end orchestrator. NOT imported here on
                          purpose — `python -m src.pipeline` would
                          double-import otherwise and emit a
                          RuntimeWarning. Use `from src import pipeline`
                          or `from src.pipeline import run_pipeline`.
"""

from . import story_downloader
from . import text_loader
from . import segmenter
from . import emotion_analyzer
from . import emotion_visualizer
from . import summarizer
from . import tts_engine
from . import audio_assembler

__all__ = [
    "story_downloader",
    "text_loader",
    "segmenter",
    "emotion_analyzer",
    "emotion_visualizer",
    "summarizer",
    "tts_engine",
    "audio_assembler",
]
