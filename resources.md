# Resources — Learning TTS & Voice AI

Curated list of videos, papers, and articles to get you interview-ready.
Estimated total time if you follow the priority track: ~12–15 hours over one week.

---

## 🎬 Videos to watch (priority order)

### TTS fundamentals

**1. "Neural Text-to-Speech Synthesis" — Heiga Zen (Google Brain)**
- Where to find: search "Heiga Zen TTS tutorial INTERSPEECH" on YouTube
- Why: one of the clearest walkthroughs of how Tacotron and WaveNet connect, from someone who built production TTS at Google. Covers the attention alignment problem in depth.
- Time: ~45 min

**2. Yannic Kilcher — "FastSpeech 2: Fast and High-Quality End-to-End Text to Speech"**
- YouTube: search "Yannic Kilcher FastSpeech 2"
- Why: Yannic is excellent at breaking down paper architecture. You'll understand why duration predictors replaced attention for TTS.
- Time: ~30 min

**3. Andrej Karpathy — "Let's build GPT: from scratch, in code, spelled out"**
- YouTube: Andrej Karpathy channel
- Why: not TTS-specific, but the single best explanation of transformer attention. Once you understand self-attention from this video, cross-attention in TTS makes immediate sense.
- Time: ~2 hours (worth every minute — watch at 1.5x)

**4. AssemblyAI — "How Whisper Works: The Details"**
- YouTube: AssemblyAI channel
- Why: clear walkthrough of Whisper's encoder-decoder architecture, language detection token, and how it handles multilingual data. Directly relevant to this project.
- Time: ~20 min

**5. "VITS: Conditional Variational Autoencoder with Adversarial Learning" — paper walkthrough**
- Search on YouTube for "VITS TTS paper explained"
- Why: VITS is the architecture behind MMS-TTS and many production systems. Understanding its three components (VAE + flow + GAN) is key.
- Time: ~30 min

**6. AI Coffee Break with Letitia — "Normalizing Flows explained"**
- YouTube: AI Coffee Break channel
- Why: normalizing flows are used in VITS and are a common interview topic in generative modeling. This is the clearest visual explanation.
- Time: ~15 min

---

### Multilingual and production voice AI

**7. Meta AI — "MMS: Scaling Speech Technology to 1000+ Languages" (talk)**
- Search "Meta MMS speech 1000 languages" on YouTube
- Why: directly covers the architecture and training decisions behind MMS-TTS, the main multilingual model in this project. Very relevant to Siri multilingual work.
- Time: ~25 min

**8. "How Siri sounds like a human" — any Apple WWDC session on TTS**
- developer.apple.com/videos — search "Siri voice" or "text to speech"
- Why: shows what production-level considerations Apple engineers think about — latency, on-device inference, language coverage. Context for the role you're applying to.
- Time: ~20 min

---

### Agentic AI

**9. LangChain — "Building Agents" tutorial**
- YouTube: LangChain official channel
- Why: hands-on introduction to the ReAct loop and tool use. The code directly mirrors the architecture in this project.
- Time: ~30 min

**10. "ReAct: Synergizing Reasoning and Acting in Language Models" — paper walkthrough**
- Search "ReAct paper explained" on YouTube
- Why: the theoretical foundation for agentic AI. 15-minute read of the paper is enough; a video walkthrough helps if the paper feels dense.
- Time: ~20 min

---

## 📄 Papers to read (not memorize — understand the key idea)

| Paper | Key idea | Read this section |
|-------|----------|-------------------|
| Tacotron 2 (Shen et al. 2018) | Location-sensitive attention for TTS alignment | Sections 1, 2, 3 |
| FastSpeech 2 (Ren et al. 2020) | Duration/pitch/energy predictors, parallel TTS | Sections 1, 3 |
| VITS (Kim et al. 2021) | End-to-end TTS with VAE + flow + GAN | Sections 1, 3.1, 3.2 |
| SpeechT5 (Ao et al. 2021) | Unified speech-text pre-training | Sections 1, 3 |
| Whisper (Radford et al. 2022) | Weakly supervised multilingual ASR | Sections 1, 2, 3 |
| MMS (Pratap et al. 2023) | 1100+ language TTS/ASR from bible recordings | Sections 1, 2 |
| HiFi-GAN (Kong et al. 2020) | Multi-period/scale discriminators for vocoders | Sections 1, 3 |
| ReAct (Yao et al. 2022) | Reason + Act agent loop | Full paper (~6 pages) |

All papers are free on arxiv.org — search by title.

---

## 📚 Articles and blog posts

**"The Illustrated Transformer" — Jay Alammar**
- jalammar.github.io/illustrated-transformer
- The best visual explanation of transformer attention on the internet. Read this before reading any TTS paper.

**"Speech Synthesis: A Tutorial" — Simon King**
- Search "Simon King speech synthesis tutorial PDF"
- A thorough academic tutorial covering the history from formant synthesis through neural TTS. Good for interview depth.

**"How does Bark work?" — Suno blog**
- suno.ai blog or their GitHub README
- Explains the three-stage codec language model approach. Directly relevant to understanding Bark's architecture.

**"Massively Multilingual Speech" — Meta AI blog**
- ai.meta.com/blog/mms-massively-multilingual-speech
- The non-technical companion to the MMS paper. Good for explaining the project at a high level in an interview.

**"Building LLM-powered agents" — Lilian Weng (OpenAI)**
- lilianweng.github.io/posts/2023-06-23-agent
- The canonical reference for agentic AI concepts: ReAct, tool use, memory, planning. Bookmark and re-read sections as needed.

---

## 🛠️ Hugging Face spaces to try (live demos, no setup)

These let you experience the models before you run them locally.

| Space | What it demos |
|-------|---------------|
| `facebook/MMS` | MMS-TTS in 1100+ languages |
| `microsoft/speecht5-tts` | SpeechT5 English TTS |
| `suno/bark` | Bark expressive TTS |
| `openai/whisper` | Whisper ASR + language detection |

Go to huggingface.co/spaces and search each name.

---

## 🧠 Interview prep: questions you should be able to answer

Based on a typical ML Engineer interview for a multilingual voice role:

**TTS architecture**
- Walk me through how Tacotron 2 generates speech, step by step.
- Why did FastSpeech replace attention with a duration predictor? What problem does this solve?
- What is HiFi-GAN's MPD and why do we need multiple discriminators?
- What is a normalizing flow, and why is it useful in VITS?

**Multilingual**
- How does MMS handle 1100 languages with one architecture?
- What is the difference between a tonal and stress-timed language, and why does this matter for TTS?
- How would you detect code-switching in real time?
- What is G2P and why is it harder for Chinese than Spanish?

**Speaker & voice**
- What is a speaker embedding? How would you use one to clone a voice?
- What is ECAPA-TDNN and what does it improve over x-vectors?

**ASR / Whisper**
- How does Whisper detect the input language? (Answer: the first decoder token is a language ID token — the model predicts it from the encoder output)
- What are Whisper's known failure modes? (Answer: hallucination on silence, degraded WER on rare languages)

**Agentic AI**
- Describe the ReAct loop and draw the Thought/Action/Observation cycle.
- What is RAG? How is FAISS used in this project?
- What are the latency bottlenecks in a voice agent, and how do you mitigate them?

**System design**
- Design a voice assistant that supports 10 languages with a 2-second latency target.
- How would you evaluate TTS quality without human annotators?

---

## 🌐 Deployment options

**Local testing**
Run `uvicorn src.api:app --reload` and open `localhost:8000`. The FastAPI backend serves the demo UI and handles audio upload (WAV/MP3) + TTS response.

**Vercel (frontend only)**
Vercel is great for the Next.js or plain HTML frontend. The Python backend (FastAPI) cannot run on Vercel's serverless functions directly. Options:
- Deploy frontend to Vercel + backend to Render.com or Railway.app (free tier, supports Python)
- Or deploy everything to Hugging Face Spaces (free, supports Gradio/FastAPI, GPU available)

**Recommended for portfolio: Hugging Face Spaces**
- Free GPU (A10G, 1hr/day on free tier)
- Supports Gradio or FastAPI
- Automatically displays in the HF model card ecosystem — relevant audience for ML engineers
- Deploy with: `huggingface-cli upload-large-folder your-username/voiceagent .`

**Docker (for any cloud)**
A `Dockerfile` is included in the project for containerized deployment to any cloud (GCP Cloud Run, AWS Lambda container, etc.).
