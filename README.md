# 🎙️ VoiceAgent — Multilingual Agentic TTS System

> A portfolio project demonstrating expertise in **text-to-speech**, **agentic AI**, and **multilingual NLP** using pretrained Hugging Face models — no training required.

---

## What this project does

VoiceAgent is a conversational AI agent that:

1. Accepts text or audio input (via Whisper ASR)
2. Understands the user's intent and language
3. Uses tools (web search, memory, language detection) to form a response
4. Speaks the answer back in the user's language using the best available TTS model

The system can answer questions in **English, Spanish, French, German, Chinese, and 1000+ other languages** by routing to the appropriate pretrained TTS model.

---

## Why this matters for multilingual voice AI

Building voice assistants for multiple languages is hard because:

- Different languages have different phoneme inventories, prosody patterns, and writing systems
- A single TTS model rarely covers all languages well
- The agent must detect language, adapt its response, and select the right voice model — all at runtime

This project shows a modular architecture that mirrors how production multilingual voice assistants (like Siri) solve these problems.

---

## Models used (all free, Hugging Face)

| Task | Model | Languages |
|------|-------|-----------|
| ASR | `openai/whisper-base` | 99 languages |
| TTS (English) | `microsoft/speecht5_tts` | English |
| TTS (Multilingual) | `facebook/mms-tts-*` | 1100+ languages |
| TTS (Expressive) | `suno/bark-small` | English + some multilingual |
| LM backbone | `mistralai/Mistral-7B-Instruct-v0.2` (via API) | Multilingual |
| Language detection | `langdetect` + `pycld2` | 80+ languages |

---

## Project structure

```
voiceagent/
├── src/
│   ├── agent/
│   │   ├── orchestrator.py      # Main agent loop (ReAct pattern)
│   │   ├── tools.py             # Tool definitions (search, memory, detect_lang)
│   │   └── prompts.py           # System prompts per language
│   ├── tts/
│   │   ├── router.py            # Language → model selector
│   │   ├── speecht5.py          # SpeechT5 wrapper
│   │   ├── mms.py               # MMS-TTS wrapper (1100 langs)
│   │   └── bark.py              # Bark wrapper (expressive)
│   ├── asr/
│   │   └── whisper_asr.py       # Whisper transcription
│   ├── memory/
│   │   ├── vector_store.py      # FAISS-based retrieval memory
│   │   └── session.py           # Per-session short-term memory
│   └── utils/
│       ├── audio_io.py          # WAV/MP3 read-write helpers
│       └── lang_detect.py       # Language identification utilities
├── demos/
│   ├── demo_en.py               # English Q&A with voice output
│   ├── demo_multilingual.py     # Switch languages mid-conversation
│   └── demo_agentic.py          # Full agent loop with tool use
├── tests/
│   ├── test_tts_router.py
│   ├── test_asr.py
│   └── test_agent_tools.py
├── docs/
│   └── architecture.png         # System diagram
├── assets/
│   └── sample_outputs/          # Pre-generated audio samples (WAV)
├── theory.md                    # Deep-dive: TTS, attention, multilingual NLP
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/YOUR_USERNAME/voiceagent.git
cd voiceagent
pip install -r requirements.txt

# 2. Run the English demo (no API key needed — uses local models)
python demos/demo_en.py

# 3. Run the multilingual demo
python demos/demo_multilingual.py --lang es   # Spanish
python demos/demo_multilingual.py --lang zh   # Chinese

# 4. Run the full agentic demo (needs HF token for Mistral)
export HF_TOKEN=your_token_here
python demos/demo_agentic.py
```

---

## Key design decisions

**TTS routing over a single model** — Rather than using one model for all languages, the router selects the specialist model for each language code. This mirrors how production systems handle language diversity without model bloat.

**ReAct agent pattern** — The agent alternates between Reasoning (what do I know?) and Acting (call a tool). This makes the agent's decision trace inspectable and debuggable, which is critical in a voice context where errors are hard to catch.

**FAISS vector memory** — The agent stores past Q&A pairs as embeddings. On new queries it retrieves semantically similar past turns, simulating episodic memory without a database.

**Whisper for ASR** — Using Whisper as the front-end means the system accepts voice input and auto-detects the input language, feeding that signal to the TTS router for response language selection.

---

## Sample outputs

Pre-generated audio samples in `assets/sample_outputs/`:

- `en_weather_query.wav` — English agent answering a weather question
- `es_greeting.wav` — Spanish TTS via MMS
- `fr_factual.wav` — French TTS via MMS
- `en_expressive_story.wav` — Bark model with prosody variation

---

## What I'd build next with more time

- Fine-tune a speaker embedding for consistent voice identity across languages
- Add cross-lingual voice cloning (same speaker, different language)
- Streaming TTS output (generate and play audio chunk-by-chunk)
- Evaluation harness: MOS scoring, WER benchmarks per language

---

## Concepts covered (see `theory.md`)

- Neural TTS architectures: Tacotron2, FastSpeech2, VITS, SpeechT5
- Multilingual phoneme systems and G2P (grapheme-to-phoneme)
- Cross-attention in sequence-to-sequence TTS
- Agentic AI: ReAct, tool use, memory
- Speaker embeddings and voice cloning
- ASR with Whisper: architecture and multilingual handling
- MOS (Mean Opinion Score) and TTS evaluation

---

## Author

Built as part of a portfolio for ML Engineer roles in voice AI / multilingual NLP.
# voiceagentai
