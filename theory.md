# Theory — Multilingual TTS & Agentic AI

> This document covers the core concepts behind this project. Written as interview preparation for ML Engineer roles in voice AI (e.g. Siri multilingual).

---

## 1. Text-to-Speech: the problem

TTS converts a text string into a natural-sounding waveform. The core challenge is that the mapping is **one-to-many**: the text "read" can be pronounced two ways ("reed" vs "red"), and even a single pronunciation can vary in pitch, speed, and emotion across speakers and contexts. The system must resolve all of this without being told.

There are three sub-problems every TTS system must solve:

1. **Linguistic analysis** — what does this text mean phonetically? (G2P, prosody prediction)
2. **Acoustic modeling** — what does that phonetic sequence sound like? (mel spectrogram generation)
3. **Vocoding** — how do we turn a spectrogram into an audio waveform? (neural vocoder)

---

## 2. TTS architectures — evolution

### 2.1 Tacotron 2 (2018)

Tacotron 2 was the first neural TTS system to achieve near-human naturalness on English.

**Architecture:**
- Encoder: a stack of CNN layers + bidirectional LSTM that encodes the input character sequence into a hidden representation
- Decoder: an autoregressive LSTM that predicts mel spectrogram frames one at a time, attending to the encoder output at each step
- Attention: location-sensitive attention — a weighted sum over encoder states, biased toward monotonic left-to-right alignment (critical for TTS, where the text must be read in order)
- Vocoder: WaveNet (originally), later replaced by WaveRNN or HiFi-GAN for speed

**Key insight:** The attention mechanism handles the alignment problem implicitly. The model learns that character "h" in position 3 corresponds to spectrogram frames 8-12, without being told. This is powerful but fragile — attention can collapse (repeat a word) or skip (drop a syllable), especially on long inputs.

**Limitation:** Autoregressive decoding is slow — each frame depends on the previous one, so you cannot parallelize.

---

### 2.2 FastSpeech 2 (2020)

FastSpeech 2 replaces autoregressive decoding with a **parallel, non-autoregressive** architecture.

**Architecture:**
- Feed-Forward Transformer (FFT) blocks encode the phoneme sequence
- A **duration predictor** explicitly predicts how many spectrogram frames each phoneme should produce
- Phoneme representations are repeated (length-regulated) to match the predicted duration
- **Pitch predictor** and **energy predictor** add prosody information at the phoneme level
- Parallel mel-spectrogram generation — all frames produced in one forward pass

**Key insight:** By making duration, pitch, and energy explicit and predictable, FastSpeech 2 gives you fine-grained control over prosody. You can change the pitch of a single phoneme without re-running the whole model.

**Trade-off:** Quality is slightly lower than the best autoregressive models (less contextual flexibility), but inference is 10-100x faster.

---

### 2.3 VITS (2021) — Variational Inference with adversarial learning for end-to-end TTS

VITS is the first model to go **end-to-end from text to waveform** in a single model, eliminating the separate vocoder.

**Architecture:**
- Posterior encoder: encodes the ground-truth waveform into a latent distribution z
- Prior encoder: predicts that same latent distribution from the text
- Flow-based decoder: normalizing flow that maps z to a waveform
- Discriminator (from GAN training): ensures the waveform sounds real
- Stochastic duration predictor: samples duration from a distribution rather than predicting a single value — introduces natural rhythm variability

**Key insight:** VITS learns a latent space that captures all acoustic variability (speaker identity, pitch, rhythm) jointly. The flow allows exact likelihood computation, making training stable.

**Why it matters:** VITS outperforms Tacotron 2 + WaveGlow in MOS scores while running in real time. It is the architecture behind many open-source voice cloning systems.

---

### 2.4 SpeechT5 (Microsoft, 2022)

SpeechT5 is a unified pre-trained model for **speech and text**, inspired by T5.

**Architecture:**
- Shared encoder-decoder Transformer pre-trained on both speech and text
- For TTS: text tokens → shared encoder → speech decoder → HiFi-GAN vocoder
- For ASR: mel spectrogram → speech encoder → text decoder
- Speaker conditioning: an x-vector speaker embedding is injected into the decoder, enabling multi-speaker synthesis

**Key insight:** By pre-training on both modalities simultaneously, SpeechT5 learns a shared representation space. Speech and text tokens live in the same embedding space. This enables cross-modal transfer learning.

**Why it matters for multilingual:** The same pre-trained backbone can be fine-tuned for new languages with minimal data, because the model has already learned general speech-text correspondences.

---

### 2.5 MMS-TTS (Meta, 2023)

Meta's Massively Multilingual Speech project trained VITS-based TTS models for **1107 languages** using data from the New Testament audio bible recordings.

**How they handled 1107 languages:**
- Language-specific models: a separate VITS model per language (not a single model that speaks all languages)
- Shared architecture, different weights
- Common phoneme representation using the International Phonetic Alphabet (IPA)

**Key insight:** The phoneme-based input (IPA) is language-agnostic. The model never sees raw characters — it sees phoneme tokens. This means adding a new language only requires a phonemizer (G2P tool) for that language, not a new architecture.

---

### 2.6 Bark (Suno, 2023)

Bark is a GPT-style generative audio model trained on diverse audio data.

**Architecture:**
- Three-stage language model: semantic tokens → coarse acoustic tokens → fine acoustic tokens
- Codec: EnCodec (Meta) as the neural audio codec — audio is tokenized into discrete codes
- No explicit phoneme step — the model learns pronunciation from raw text
- Supports non-speech sounds: laughter, music, background noise

**Key insight:** By treating audio generation as a language modeling problem over codec tokens, Bark can produce highly natural and expressive speech, including paralinguistic cues. The cost is speed — it is the slowest model here.

---

## 3. Cross-attention in TTS

Cross-attention is the mechanism that connects the encoder (text) and decoder (audio) in seq2seq TTS.

At each decoder timestep t, cross-attention computes:

```
Attention(Q, K, V) = softmax(QKᵀ / √d_k) · V
```

Where:
- **Q** (query) comes from the decoder state at step t — "what am I looking for?"
- **K, V** (key, value) come from the encoder output — the full text representation
- The result is a weighted sum of encoder values, biased toward the most relevant text positions

In TTS, the attention should be **monotonically aligned** — the decoder should read the text left-to-right. Tacotron 2 enforces this softly via location-sensitive attention, which biases the attention weights toward positions near the previous attention peak. VITS avoids the problem entirely by using a duration predictor instead.

**Why this matters for multilingual:** Different languages have different text-to-speech alignment characteristics. Languages with complex morphology (e.g. Finnish) or logographic writing (e.g. Chinese) require careful tokenization before the attention mechanism can align correctly.

---

## 4. G2P — Grapheme-to-Phoneme conversion

G2P converts written text into phoneme sequences. This is the first step in traditional TTS pipelines.

**Why it's hard:**
- English is notoriously irregular: "read" (present) vs "read" (past), "lead" (verb) vs "lead" (metal)
- Chinese characters do not encode pronunciation at all — you need a separate dictionary or model
- Arabic and Hebrew are written without short vowels — the G2P model must infer them from context

**Approaches:**
- Rule-based: hand-crafted phoneme rules (Festival, eSpeak)
- Statistical: joint-sequence models trained on pronunciation dictionaries
- Neural: seq2seq transformers trained on (word, phoneme) pairs (e.g. `charsiu_g2p`)
- Multilingual: a single model covering many languages using IPA as the output space (used by MMS)

---

## 5. Neural vocoders

The vocoder converts a mel spectrogram (a compact frequency-domain representation) back into a time-domain waveform.

| Vocoder | Architecture | Speed | Quality |
|---------|--------------|-------|---------|
| WaveNet | Dilated causal CNN, autoregressive | Very slow | Excellent |
| WaveRNN | RNN, autoregressive | Slow | Good |
| WaveGlow | Normalizing flow | Fast | Good |
| HiFi-GAN | GAN (generator + discriminators) | Very fast | Excellent |
| EnCodec | VQ-VAE neural codec | Fast | High (codec artifacts) |

**HiFi-GAN** is the current standard for most open-source TTS. It uses multiple discriminators operating at different frequency resolutions to catch both fine-grained artifacts and coarse prosodic errors.

---

## 6. Speaker embeddings and voice cloning

Speaker embeddings are fixed-dimensional vectors that encode a speaker's vocal identity, extracted from a few seconds of reference audio.

**Common approaches:**
- **d-vectors**: embeddings from a speaker verification model (GE2E loss)
- **x-vectors**: TDNN-based speaker embeddings from Kaldi/SpeechBrain
- **ECAPA-TDNN**: state-of-the-art speaker embeddings used in modern TTS

**Voice cloning workflow:**
1. Extract speaker embedding from 3-30 seconds of reference audio
2. Inject the embedding into the TTS decoder (concatenated to every decoder input, or as a conditioning vector via FiLM layers)
3. Generate speech in that speaker's voice for arbitrary text

**Zero-shot vs few-shot cloning:**
- Zero-shot: clone from a single reference clip, no fine-tuning (e.g. VALL-E, StyleTTS2)
- Few-shot: fine-tune the TTS model on 10-100 utterances from the target speaker

---

## 7. Multilingual TTS challenges

### 7.1 Phoneme inventory coverage

Every language has a different set of phonemes. English has ~44 phonemes, Mandarin has ~21 consonants + 5 tones, Swahili is largely phonetically regular. A multilingual TTS model must either:

- Use a universal phoneme set (IPA) and train one model per language
- Use a language-agnostic token space and train a single model that generalizes

### 7.2 Prosody variation

Prosody (rhythm, pitch, stress) differs dramatically across languages:
- English is **stress-timed**: stressed syllables occur at roughly equal intervals
- French is **syllable-timed**: all syllables have roughly equal duration
- Mandarin is **tonal**: pitch encodes lexical meaning (four tones + neutral)

A model trained only on English will have wrong prosodic assumptions when generating other languages.

### 7.3 Code-switching

Code-switching occurs when a speaker switches languages mid-sentence. This is common in multilingual communities (e.g. Spanglish). Production voice assistants must detect code-switches and handle them gracefully — either by using a code-switch-aware model or by detecting the switch and routing to a different TTS engine.

### 7.4 Script diversity

Languages use different writing systems:
- Latin script (English, Spanish, French, Swahili)
- CJK (Chinese, Japanese, Korean) — require word segmentation before G2P
- Arabic/Hebrew — right-to-left, short vowels usually omitted
- Devanagari (Hindi, Sanskrit) — syllabic (akshara) units, not alphabetic

The tokenizer and G2P module must handle each script correctly before the TTS model ever sees the input.

---

## 8. TTS evaluation — MOS and beyond

**Mean Opinion Score (MOS):** Listeners rate audio naturalness on a 1-5 scale. Scores above 4.0 are considered near-human. MOS is subjective and expensive to collect.

**Automatic metrics:**
- **WER (Word Error Rate):** Run ASR on the generated audio and compare to the input text. Measures intelligibility.
- **Mel Cepstral Distortion (MCD):** L2 distance in mel cepstrum space. Measures acoustic similarity to ground truth.
- **UTMOS:** An automatic MOS predictor trained on human ratings (neural proxy for MOS).
- **Speaker similarity cosine distance:** Compare speaker embeddings of generated vs reference audio.

---

## 9. Agentic AI fundamentals

### 9.1 What makes an AI "agentic"?

An agent is an AI system that:
- Has **goals** (not just inputs/outputs)
- Takes **actions** that affect the world (tool calls, API requests)
- Observes **feedback** from those actions
- **Plans** over multiple steps

A simple chatbot answers one question. An agent can search the web, read a document, compute something, and compose a multi-step answer — choosing its own path.

### 9.2 ReAct (Reason + Act)

ReAct is the most common agentic pattern. The model alternates between:

```
Thought: I need to find the current weather in Madrid.
Action: web_search("Madrid weather today")
Observation: Madrid is 22°C and sunny.
Thought: I have the answer. I'll respond in Spanish since the user asked in Spanish.
Action: tts_speak("Hoy en Madrid hace 22 grados y está soleado.", lang="es")
```

Each cycle produces a thought (internal reasoning), an action (tool call), and an observation (tool result). This loop continues until the agent decides it has a final answer.

**Why it works:** Forcing the model to articulate its reasoning before acting reduces errors and makes the agent's behavior inspectable and debuggable.

### 9.3 Tool use

Tools are functions the agent can call. In LangChain they are defined with:
- A name (used in the prompt)
- A description (tells the LLM when to use it)
- A schema (defines input parameters)
- An implementation (Python function)

In this project, tools include:
- `web_search(query)` — DuckDuckGo
- `retrieve_memory(query)` — FAISS semantic search over past turns
- `detect_language(text)` — returns an ISO 639-1 language code
- `speak(text, lang)` — routes to the appropriate TTS model

### 9.4 Memory types in agents

| Type | Description | Implementation |
|------|-------------|----------------|
| In-context | Everything in the current prompt window | Python list of messages |
| Episodic | Past turns retrieved by semantic similarity | FAISS + sentence-transformers |
| Semantic | General world knowledge | Pretrained LLM weights |
| Procedural | How to use tools | Tool definitions in the prompt |

For a voice assistant, episodic memory is the most important: "you asked about the weather in Madrid three turns ago" should inform how the agent answers "what about tomorrow?"

### 9.5 Why agentic + TTS is hard

When TTS is the output modality of an agentic system, several new challenges appear:

1. **Latency**: the agent must think AND generate audio. Each adds hundreds of milliseconds.
2. **Error recovery**: if the agent generates wrong information and speaks it, the user cannot "skim" past it — they have to listen.
3. **Prosody and intent**: the TTS model must know *how* to say something, not just *what* to say. An error message should sound different from an excited answer.
4. **Streaming**: production systems start speaking before the full response is generated. This requires chunk-wise TTS with sentence-level flushing.

---

## 10. Whisper — multilingual ASR

Whisper (OpenAI, 2022) is a seq2seq Transformer trained on 680,000 hours of weakly supervised audio data from the internet.

**Architecture:**
- Encoder: CNN frontend (converts 30s mel spectrogram chunks to feature maps) + Transformer encoder
- Decoder: autoregressive Transformer decoder that generates text tokens
- Special tokens: `<|en|>`, `<|es|>`, `<|transcribe|>`, `<|translate|>` — the decoder is told what task to perform via prompt tokens

**Key features:**
- Transcribes in the source language or translates to English
- Language identification as a side effect of the first decoder step
- Robust to noise, accents, and non-native speakers
- Works out of the box for 99 languages

**Why this matters here:** Whisper gives us both the transcription *and* the detected language code in a single pass. We feed that language code directly to the TTS router to ensure the agent responds in the same language the user spoke in.

---

## 11. FAISS — vector similarity search

FAISS (Facebook AI Similarity Search) is a library for efficient nearest-neighbor search in high-dimensional vector spaces.

In this project, we use it to implement **episodic memory**:

1. Each Q&A turn is encoded into a 384-dimensional embedding (via `sentence-transformers/all-MiniLM-L6-v2`)
2. Embeddings are stored in a FAISS flat L2 index
3. On a new query, the query is embedded and the k=3 most similar past turns are retrieved
4. Those past turns are injected into the agent's context window

This is a lightweight RAG (Retrieval-Augmented Generation) setup without a database.

---

## 12. Concepts to know cold for the interview

- What is the difference between mel spectrogram, MFCC, and raw waveform as audio representations?
- How does attention alignment work in Tacotron 2, and why does it sometimes fail?
- Why is FastSpeech 2 faster than Tacotron 2? What does it trade off?
- What is a normalizing flow and why is it used in VITS?
- What is an x-vector / speaker embedding, and how is it used for multi-speaker TTS?
- How does Whisper detect the language of the input?
- What is the ReAct pattern? Draw the thought/action/observation loop.
- What is FAISS and what is its role in RAG systems?
- What is MOS and what are its limitations as a TTS evaluation metric?
- What is code-switching and why is it hard for TTS systems?
- What is G2P, and why is it more complex for Chinese than for Spanish?
- What is HiFi-GAN, and what do its multiple discriminators each catch?
- What is the difference between zero-shot and few-shot voice cloning?
- What are the main latency bottlenecks in a voice agent system?
