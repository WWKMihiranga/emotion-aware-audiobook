# Emotional Audiobook

> Turning storybooks into audiobooks that *preserve emotion* — an AI pipeline that
> reads a story, understands its emotional arc, and narrates it with a voice that
> follows that arc.

Most AI summarizers and text-to-speech systems flatten emotion. Summarizers
optimize for information density and throw away the emotional turning points.
TTS engines read everything in the same even tone. This project is an
experiment in doing the opposite: keeping the *feeling* of a story intact all
the way from raw text to final audio.

---

## What it does

Given a plain-text story, the pipeline:

1. **Cleans and segments** the text into sentences and paragraphs
2. **Classifies the emotion** of every sentence (7 emotions + confidence)
3. **Builds an emotion-aware summary** that keeps the emotional peaks verbatim
   and compresses only the connective tissue
4. **Synthesizes expressive audio** — a fast engine narrates the bridges, a
   more expressive engine performs the emotional peaks
5. **Assembles a final MP3** with synced subtitles and emotion-arc cover art

The result is a condensed audiobook that still *feels* like the original story.

---

## The core idea: "Anchor and Compress"

The interesting part of this project is the summarization strategy. Standard
abstractive summarizers (BART, T5, etc.) are trained to maximize information
density — which means they systematically discard emotional nuance, because a
sentence like *"She wept"* carries little *information* even though it may be
the emotional core of a scene.

This pipeline takes a different approach:

```
1. Score every sentence by emotional intensity      (intensity = 1 − P(neutral))
2. Select "anchor" sentences — the emotional peaks   (top 20%, with floors/ceilings)
3. Summarize only the text BETWEEN anchors           (DistilBART on the "bridges")
4. Stitch anchors (verbatim) + bridges (compressed)  back together in order
```

Every emotional turning point survives word-for-word. Only the narrative
connective tissue gets compressed. The summary reads faster but keeps the
story's emotional shape.

This same anchor/bridge structure then drives the TTS stage: anchors are routed
to a more expressive voice engine, bridges to a faster one — roughly mirroring
how a human narrator reads filler evenly and "performs" the dramatic moments.

---

## Pipeline architecture

```
            ┌──────────────────────────────────────────────┐
 story.txt ─▶  Phase 1 — Text loading & segmentation        ─▶ segmented.json
            │  Strip boilerplate, extract single story from │
            │  anthologies, split into sentences (NLTK)     │
            └──────────────────────────────────────────────┘
                              │
            ┌──────────────────────────────────────────────┐
            │  Phase 2 — Emotion analysis                   │
 analyzed.json ◀  DistilRoBERTa 7-emotion classifier;       ─▶ emotion arc plots
            │  per-sentence label + full distribution       │
            └──────────────────────────────────────────────┘
                              │
            ┌──────────────────────────────────────────────┐
            │  Phase 3 — Emotion-aware summarization        │
 summary.json ◀  "Anchor and compress": keep emotional      │
            │  peaks verbatim, summarize bridges (DistilBART)│
            └──────────────────────────────────────────────┘
                              │
            ┌──────────────────────────────────────────────┐
            │  Phase 4 — Dual-engine TTS                    │
   wavs/ ◀──   Bridges → XTTS-v2 (fast, consistent)         │
            │  Anchors → Bark (expressive, emotion cues)    │
            └──────────────────────────────────────────────┘
                              │
            ┌──────────────────────────────────────────────┐
            │  Phase 5 + 6 — Assembly & polish              │
 story.mp3 ◀  Stitch segments, generate synced SRT          ─▶ story.srt
            │  subtitles, embed emotion-arc cover art       │
            └──────────────────────────────────────────────┘
```

---

## Models used

| Stage | Model | Why |
|-------|-------|-----|
| Emotion analysis | `j-hartmann/emotion-english-distilroberta-base` | 7-emotion classifier, ~330 MB, CPU-friendly |
| Summarization | `sshleifer/distilbart-cnn-12-6` | Distilled BART, ~2× faster than full BART, good at narrative compression |
| TTS (bridges) | Coqui XTTS-v2 | Fast, consistent narrator voice |
| TTS (anchors) | Suno Bark | More expressive; supports non-verbal cues (`[sighs]`, `[gasps]`) |

All models are open-source and run locally — no paid APIs required.

---

## Project structure

```
emotional_audiobook/
├── src/
│   ├── story_downloader.py    # Fetch public-domain test stories (Project Gutenberg)
│   ├── text_loader.py         # Clean text, strip boilerplate, extract anthology sections
│   ├── segmenter.py           # Sentence/paragraph segmentation (NLTK)
│   ├── emotion_analyzer.py    # Phase 2 — per-sentence emotion classification
│   ├── emotion_visualizer.py  # Emotional-arc plots (intensity + stacked area)
│   ├── summarizer.py          # Phase 3 — "anchor and compress" summarization
│   ├── tts_engine.py          # Phase 4 — XTTS-v2 + Bark dual-engine TTS
│   ├── audio_assembler.py     # Phase 5/6 — MP3 stitching, SRT subtitles, cover art
│   └── pipeline.py            # End-to-end orchestrator + CLI
├── notebooks/
│   ├── week1_setup_and_foundation.ipynb     # Phase 1 walkthrough
│   ├── week2_3_emotion_and_summary.ipynb    # Phases 2-3 walkthrough
│   ├── week4_5_tts_and_assembly.ipynb       # Phases 4-5 walkthrough
│   └── week6_full_pipeline_demo.ipynb       # Full end-to-end demo
├── data/                      # Story files & intermediate JSON (gitignored)
├── outputs/                   # Generated audio & plots (gitignored)
└── requirements.txt
```

---

## Getting started

### Requirements

- Python 3.10+
- ffmpeg (`brew install ffmpeg` on macOS, `apt-get install ffmpeg` on Linux)
- For the TTS stages: a GPU is strongly recommended (Google Colab's free tier
  works well). Phases 1-3 run fine on CPU.

### Setup

```bash
git clone https://github.com/YOUR_USERNAME/emotional-audiobook.git
cd emotional-audiobook

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Run the pipeline

As a command-line tool:

```bash
# Full pipeline (needs GPU for TTS)
python -m src.pipeline data/raw/gift_of_the_magi.txt --title "The Gift of the Magi"

# Phases 1-3 only — emotion analysis + summary, no audio (CPU is fine)
python -m src.pipeline data/raw/gift_of_the_magi.txt --no-tts

# XTTS only — skip the slower Bark engine
python -m src.pipeline data/raw/gift_of_the_magi.txt --no-bark
```

Or explore step by step in the notebooks — start with
`notebooks/week6_full_pipeline_demo.ipynb` for the full picture.

---

## Design decisions & engineering notes

A few choices worth calling out:

- **Memory discipline.** The emotion model and summarizer are never held in
  RAM at the same time — each is explicitly loaded and unloaded. This keeps the
  pipeline runnable on a 16 GB machine.

- **Caching everywhere.** Every phase writes its output to disk and skips itself
  on re-run. TTS results are cached *per segment and per engine*, so switching
  from XTTS-only to the full Bark run only re-synthesizes the ~10 anchor
  sentences rather than the whole story.

- **Anthology handling.** Public-domain stories on Project Gutenberg often come
  bundled in multi-story anthology files. The text loader can extract a single
  named story section, so "The Tell-Tale Heart" doesn't get processed as the
  entire collected works of Poe.

- **Graceful degradation.** If the expressive TTS engine fails on a segment
  (out-of-memory, model hiccup), that segment falls back to the faster engine
  automatically rather than failing the whole run.

- **Bimodal-distribution fix in anchor selection.** A naive percentile cutoff
  for "emotional peaks" breaks on mostly-neutral text, where the intensity
  distribution is bimodal — the cutoff lets baseline-noise sentences through.
  Anchor selection requires *both* a percentile threshold and an absolute
  intensity floor.

---

## Test stories

The pipeline is developed and tested on three public-domain short stories, each
chosen for a distinct emotional arc:

- **The Gift of the Magi** (O. Henry) — tenderness → surprise → bittersweet joy
- **The Necklace** (Guy de Maupassant) — pride → despair → bitter revelation
- **The Tell-Tale Heart** (Edgar Allan Poe) — calm → mounting dread → frenzy

All are fetched automatically from Project Gutenberg by `story_downloader.py`.

---

## Limitations & honest caveats

- The emotion model is trained on general English and has known quirks on
  older/formal prose — it tends to over-predict `disgust` on 19th-century
  narration, for example.
- Bark's voice consistency across separate calls is imperfect even with a
  pinned speaker preset; the narrator can shift slightly between anchor
  segments.
- TTS quality and speed depend heavily on hardware. On CPU, Bark can take
  minutes per sentence.
- This is a learning/portfolio project, not production software — there's no
  hosted service, no test suite beyond manual verification, and the model
  choices favor "runs on a laptop" over "best possible quality."

---

## Possible future work

- Fine-tune the emotion classifier on literary/narrative text
- Per-segment voice-tone control instead of binary engine routing
- A hosted web version where users process their own stories
- Better trailing-silence detection for Bark output

---

## Acknowledgements

- Story texts from [Project Gutenberg](https://www.gutenberg.org/)
- Emotion model by [j-hartmann](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
- Summarization model by [sshleifer](https://huggingface.co/sshleifer/distilbart-cnn-12-6)
- TTS via [Coqui TTS](https://github.com/coqui-ai/TTS) and [Suno Bark](https://github.com/suno-ai/bark)

---

## License

This project is released under the MIT License — see `LICENSE` for details.

*Note: the underlying models and story texts have their own licenses. Project
Gutenberg texts are public domain in the US; the Coqui XTTS-v2 model is under a
non-commercial license. Check each before any commercial use.*
