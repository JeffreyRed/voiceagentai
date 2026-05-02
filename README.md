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
