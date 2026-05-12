# 🎙️ VoiceAgent — Multilingual Agentic TTS

> Portfolio project for ML Engineer roles in voice AI and multilingual NLP.  
> Uses **pretrained Hugging Face models only** — no training required.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER INPUT                              │
│                                                                 │
│   Text prompt    ──┐                                            │
│   Audio (mic)   ──►  Whisper ASR  ──► text + language code      │
│   Language flag  ──┘                                            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AGENT ORCHESTRATOR (ReAct loop)              │
│                                                                 │
│   Thought → Action (tool call) → Observation → repeat           │
│                                                                 │
│   Tools available:                                              │
│     • web_search      (DuckDuckGo, no API key)                  │
│     • retrieve_memory (FAISS semantic search over past turns)   │
│     • detect_language (text → ISO 639-1 code)                   │
│     • get_current_time                                          │
└───────────────────────────┬─────────────────────────────────────┘
                            │  final answer text + language code
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                       TTS ROUTER                                │
│                                                                 │
│   lang == "en"          →  SpeechT5 (microsoft/speecht5_tts)    │
│   lang in MMS_SUPPORTED →  MMS-TTS  (facebook/mms-tts-{lang})  │
│   lang unknown          →  Bark     (suno/bark-small)           │
│                                                                 │
│   Key insight: each model is a specialist. We route to the      │
│   best model per language — no cross-lingual attention needed.  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
                     WAV audio output
                  (played in browser or saved)
```

---

## Why this architecture

**Language routing instead of a single polyglot model** — MMS-TTS trains a separate VITS model per language using IPA phoneme input. Routing by language code is the correct abstraction: each specialist model handles its language's phoneme inventory, prosody, and script natively. This mirrors how production multilingual voice assistants handle language diversity.

**ReAct agent pattern** — the agent alternates between reasoning (what do I know?) and acting (call a tool). Each step is logged, making the agent's behavior inspectable and debuggable — critical for voice systems where errors are hard for users to skim past.

**FAISS episodic memory** — past Q&A turns are stored as sentence embeddings. Relevant past context is retrieved and injected into each new prompt, giving the agent conversational memory without a database.

---

## Models used (all free, Hugging Face)

| Task | Model | Coverage |
|------|-------|----------|
| ASR + language ID | `openai/whisper-base` | 99 languages |
| TTS — English | `microsoft/speecht5_tts` | English |
| TTS — Multilingual | `facebook/mms-tts-{lang}` | 1100+ languages |
| TTS — Expressive fallback | `suno/bark-small` | English + others |
| Speaker embeddings | `Matthijs/cmu-arctic-xvectors` | English voices |
| Sentence embeddings | `sentence-transformers/all-MiniLM-L6-v2` | Multilingual |

---

## Project structure

```
voiceagent/
├── environment.yml              ← conda environment (start here)
├── Dockerfile                   ← for cloud deployment
│
├── src/
│   ├── api.py                   ← FastAPI backend (local + Vercel/HF Spaces)
│   ├── agent/
│   │   ├── orchestrator.py      ← ReAct agent loop
│   │   └── tools.py             ← web_search, memory, lang_detect tools
│   ├── tts/
│   │   ├── router.py            ← language → model selector
│   │   ├── speecht5.py          ← English TTS wrapper
│   │   ├── mms.py               ← Multilingual TTS wrapper (1100+ langs)
│   │   └── bark.py              ← Expressive TTS fallback
│   ├── asr/
│   │   └── whisper_asr.py       ← Whisper transcription + language detection
│   ├── memory/
│   │   └── vector_store.py      ← FAISS episodic memory
│   └── utils/
│       └── lang_detect.py       ← text language detection
│
├── frontend/
│   └── index.html               ← web demo UI (plain HTML, no framework)
│
├── demos/
│   ├── demo_en.py               ← English TTS, no API key needed
│   └── demo_multilingual.py     ← synthesize in 6 languages, save WAVs
│
├── tests/
│   ├── test_tts_router.py       ← routing logic tests (no model downloads)
│   └── test_lang_detect.py      ← language detection tests
│
├── assets/
│   └── sample_outputs/          ← pre-generated WAV files for the demo
│
├── theory.md                    ← TTS architecture deep-dive (interview prep)
├── glossary.md                  ← key terms: phoneme, vocoder, prosody, ReAct…
└── resources.md                 ← videos, papers, and deployment guide
```

---

## Quickstart

```bash
# 1. Create and activate the conda environment
conda env create -f environment.yml
conda activate voiceagent

# 2. Test English TTS (downloads ~500MB of models on first run)
python demos/demo_en.py

# 3. Test multilingual TTS
python demos/demo_multilingual.py            # all 6 languages
python demos/demo_multilingual.py --lang es  # Spanish only

# 4. Run the web demo locally
uvicorn src.api:app --reload --port 8000
# Open http://localhost:8000/frontend/index.html in your browser

# 5. Run tests (no model downloads required)
pytest tests/ -v
```

---

## Deployment options

| Platform | Cost | Notes |
|----------|------|-------|
| **Local** | Free | `uvicorn src.api:app --reload` |
| **Hugging Face Spaces** | Free (GPU available) | Best for ML portfolio visibility |
| **Render / Railway** | Free tier | Python-native, easy setup |
| **Vercel** | Free | Frontend only — pair with Render for backend |
| **Docker (any cloud)** | Varies | `Dockerfile` included |

**Recommended for portfolio:** Hugging Face Spaces. Free GPU quota, visible to ML engineers, easy to link from a model card or resume.

---

## Concepts covered

See `theory.md` for deep dives and `glossary.md` for quick definitions:

- Neural TTS architectures: Tacotron 2 → FastSpeech 2 → VITS → SpeechT5 → MMS → Bark
- G2P (grapheme-to-phoneme) and phoneme inventory differences across languages
- Neural vocoders: WaveNet → HiFi-GAN → EnCodec
- Speaker embeddings and voice cloning (x-vectors, ECAPA-TDNN)
- Multilingual challenges: tonal languages, code-switching, script diversity
- Agentic AI: ReAct pattern, tool use, RAG, episodic memory with FAISS
- Whisper: multilingual ASR with built-in language identification
- TTS evaluation: MOS, UTMOS, WER intelligibility, RTF

---

## What I would build next

- Streaming TTS: flush sentence-by-sentence to hit a <2s latency budget
- Speaker embedding injection for consistent voice identity across languages
- Fine-tune MMS on a custom voice with 10 minutes of data
- UTMOS-based automated quality evaluation across all supported languages
- Cross-lingual voice cloning: same speaker timbre in a different language

---

*Built as portfolio work for ML Engineer (voice AI / multilingual NLP) roles.*

---

## How the pipeline works — step by step

This section explains exactly what happens when you run `python demos/demo_en.py`, and why each decision was made.

---

### Step 1 — Download the pretrained model

```python
SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
```

We do not train anything. We download weights that Microsoft already trained on hundreds of hours of English speech. `from_pretrained()` pulls the model from Hugging Face Hub and caches it locally in `~/.cache/huggingface/`. Next run it loads from disk instantly.

Three components are downloaded:
- **Processor** — tokenizes text into input IDs (character tokens)
- **SpeechT5ForTextToSpeech** — the acoustic model, maps tokens → mel spectrogram
- **SpeechT5HifiGan** — the vocoder, maps mel spectrogram → waveform

---

### Step 2 — Load the speaker embedding

```python
table = pq.read_table(...)          # parquet file from HuggingFace datasets
xvector = table["xvector"][7306]    # row 7306 = BDL (US male)
embedding = torch.tensor(xvector).unsqueeze(0)  # shape (1, 512)
```

**Why do we need a speaker embedding?**

SpeechT5 is a *multi-speaker* model — it was trained on many different voices. Without telling it which voice to use, it has no default. The speaker embedding is a 512-dimensional vector that encodes one speaker's vocal identity (their timbre, accent, speaking style). It was extracted from real audio recordings using a speaker verification model (SpeechBrain's x-vector model).

**Why row 7306?**

The CMU ARCTIC dataset has 7,931 recordings across 7 speakers. Row 7306 happens to be a BDL (US male) utterance. Any row from the same speaker would give a similar voice. We pick one and reuse it for all synthesis — that gives us a consistent voice identity.

**Why a 512-dimensional vector?**

The x-vector model compresses a variable-length audio recording into a fixed 512-dimensional vector. This means you can describe any speaker's voice in the same compact format, regardless of how long their reference audio is. The TTS decoder uses this vector to condition every step of speech generation.

**What happened when we used a silent dummy embedding?**

Silence has no vocal identity — the x-vector model extracts noise. The decoder receives a meaningless conditioning signal and produces garbled, robotic output. This is why the first attempt sounded bad: the architecture was correct, the input was wrong.

---

### Step 3 — Language detection and model routing

```python
# In TTSRouter
if lang == "en":
    return SpeechT5TTS()          # best English quality
elif lang in MMS_SUPPORTED:
    return MMSTTS(lang=lang)      # facebook/mms-tts-{lang}
else:
    return BarkTTS()              # expressive fallback
```

**Why not one model for all languages?**

Every language has a different phoneme inventory, prosody system, and writing script. A model trained only on English has learned English phoneme distributions — it cannot produce a Spanish trill /r/ or a Mandarin tone correctly because it never saw those patterns during training.

Meta's MMS project solved this by training a separate VITS model per language using IPA (International Phonetic Alphabet) as the input. IPA is a universal phoneme notation — it represents sounds, not letters. So each language is first converted to IPA (via a G2P tool), then synthesized by its specialist model.

**The routing is the multilingual system.** There is no cross-attention between languages, no polyglot decoder, no language embedding. Just: detect language → select the specialist → synthesize. Clean, modular, and each model is as good as it can be for its language.

**How language detection works:**
- For text input: `langdetect` library (Google's language detection algorithm)
- For audio input: Whisper ASR — the first decoder token it predicts is a language ID token (e.g. `<|es|>` for Spanish), so we get transcription and language in one forward pass

---

### Step 4 — Synthesis

```python
inputs = processor(text=text, return_tensors="pt")
speech = model.generate_speech(
    inputs["input_ids"],      # tokenized text
    speaker_embedding,         # who is speaking
    vocoder=vocoder,           # waveform generator
)
```

The model:
1. Encodes the text tokens into a hidden representation
2. Autoregressively decodes mel spectrogram frames, conditioning on the speaker embedding at each step
3. Passes the mel spectrogram through HiFi-GAN to produce a waveform

Output is a float32 numpy array at 16,000 Hz. We write it to a WAV file with `soundfile`.

---

### Why pretrained models and not training from scratch?

Training a TTS model from scratch requires:
- 20–100+ hours of clean, single-speaker audio with transcripts
- A GPU with 16–40GB VRAM
- Days to weeks of training time
- Expert knowledge of loss functions, learning rate schedules, and evaluation

Using pretrained models gives us:
- State-of-the-art quality immediately
- No data collection
- CPU-compatible inference
- 1100+ languages (via MMS) that would be impossible to train independently

For production systems like Siri, models are trained from scratch on proprietary data. For a portfolio project demonstrating understanding of the architecture and pipeline, pretrained models are the correct choice — they let you focus on the system design and integration rather than the training infrastructure.