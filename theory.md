# Theory — Multilingual TTS & Agentic Voice AI

> Deep-dive reference for interview prep. Companion to `glossary.md` (quick definitions) and `resources.md` (videos and papers).

---

## 1. The three sub-problems every TTS system solves

TTS maps text to waveform. This involves three distinct problems:

**Linguistic analysis** — what does this text sound like phonetically?
- Text normalization: "Dr." → "Doctor", "$42" → "forty-two dollars"
- G2P (grapheme-to-phoneme): convert characters to phoneme symbols
- Prosody prediction: which words are stressed? Where do pauses go?

**Acoustic modeling** — what does that phoneme sequence sound like?
- Maps phonemes → mel spectrogram (a 2D time-frequency representation)
- Must model pitch (F0), duration, energy, and speaker identity

**Vocoding** — how do we turn a spectrogram into a waveform?
- Mel spectrogram → time-domain audio
- Neural vocoders (HiFi-GAN) have replaced all classical approaches

---

## 2. TTS architecture evolution

### Tacotron 2 (Google, 2018)

The first neural TTS system to achieve near-human naturalness on English.

- Encoder: CNN layers + bidirectional LSTM → encodes character sequence
- Decoder: autoregressive LSTM → predicts mel frames one at a time
- Attention: location-sensitive attention aligns decoder to encoder positions
- Vocoder: WaveNet (original), later HiFi-GAN

**The alignment problem:** Attention must read text left-to-right. If it drifts — repeating a position (word repetition) or skipping (dropped syllables) — the output is broken. Tacotron 2 uses location-sensitive attention to bias toward monotonic alignment, but it still fails on long inputs.

**Bottleneck:** autoregressive decoding — frame N depends on frame N-1. Cannot parallelize. Slow.

---

### FastSpeech 2 (Microsoft, 2020)

Replaces attention with an **explicit duration predictor**.

- Feed-Forward Transformer blocks encode phonemes
- Duration predictor: predicts how many mel frames each phoneme needs
- Length regulator: repeats phoneme representations to match predicted duration
- Pitch predictor: explicit F0 contour per phoneme
- Energy predictor: explicit loudness per phoneme
- All frames generated in parallel → 10–100× faster than Tacotron 2

**Key insight:** by making duration, pitch, and energy explicit scalar predictions, FastSpeech 2 gives fine-grained prosody control. You can raise the pitch of a single phoneme by overriding its pitch value at inference time.

**Trade-off:** quality is slightly lower than the best autoregressive models because there is no step-by-step self-feedback.

---

### VITS (Kakao, 2021)

**End-to-end** TTS from text to waveform in a single model — no separate vocoder.

Three components trained jointly:

1. **Posterior encoder (VAE):** encodes the ground-truth waveform into a latent distribution z
2. **Prior encoder (normalizing flow):** predicts that same distribution from the text
3. **Decoder (upsampler):** decodes z to waveform (similar to HiFi-GAN generator)
4. **GAN discriminator:** ensures the waveform sounds real

A **stochastic duration predictor** samples durations from a distribution (not a single value), introducing natural rhythm variability. This is why VITS speech sounds less robotic than FastSpeech 2.

**Normalizing flow** maps a simple Gaussian prior (easy to sample from) to a complex acoustic posterior (what the spectrogram actually looks like for this text). The flow is invertible and has exact likelihood — unlike GANs, training is stable.

**Why it matters for multilingual:** VITS is the backbone of MMS-TTS. Training a new language requires only a VITS checkpoint and a phonemizer for that language. The architecture generalizes.

---

### SpeechT5 (Microsoft, 2022)

A unified pre-trained model for **both speech and text**, inspired by T5.

- Pre-trained jointly on speech and text corpora
- Text tokens and speech tokens share the same encoder-decoder Transformer
- For TTS: text → shared encoder → speech decoder → HiFi-GAN vocoder
- Speaker conditioning: x-vector embedding injected into the decoder

**Why joint pre-training helps:** the model learns that "dog" in text and the acoustic realization of "dog" in speech are related representations. Fine-tuning for new languages requires less data because of this shared foundation.

**Used in this project** for English — it produces the highest-quality English output among our free models.

---

### MMS-TTS (Meta, 2023)

VITS-based TTS models covering **1107 languages**, trained on New Testament audio recordings.

**How 1107 languages are handled:**
- Separate model checkpoint per language (same VITS architecture, different weights)
- IPA phoneme input — script-agnostic. Every language is converted to IPA before the model sees it
- This means adding a new language requires only a G2P/phonemizer for that language
- No cross-lingual attention or language ID embedding needed at the model level

**The routing insight (core to this project):** because each language has its own model, the "multilingual" problem becomes a routing problem. Detect the language, select the model, synthesize. Clean separation of concerns.

**Data source:** religious text recordings are far from ideal (narrow prosodic range, formal register), but they were the only consistently multilingual parallel audio source at this scale.

---

### Bark (Suno, 2023)

A GPT-style **generative audio model** that treats speech as a language modeling problem.

Three-stage pipeline (all autoregressive language models):
1. Text → semantic tokens (high-level content)
2. Semantic tokens → coarse acoustic tokens (EnCodec codebook 1-2)
3. Coarse tokens → fine acoustic tokens (EnCodec codebook 3-8)

**Neural codec:** EnCodec (Meta) compresses audio into 8 parallel streams of discrete tokens (residual vector quantization). Bark generates these tokens, then EnCodec decodes them to a waveform.

**What Bark can do that others can't:**
- Paralinguistic cues: `[laughter]`, `[sighs]`, `[clears throat]`
- Music and non-speech sounds
- Expressive, contextually appropriate prosody (not just pitch/energy, but style)

**Cost:** very slow on CPU (RTF ≈ 3–10). Used as an expressive fallback in this project.

---

## 3. Cross-attention vs routing — why we chose routing

**Cross-attention in seq2seq TTS** (like Tacotron 2) lets the decoder attend to all encoder positions at each step. This is powerful but has problems:

- Attention can be non-monotonic (the model can look backward in text), causing repetition
- Attention alignment is fragile on long inputs
- A single cross-attention model cannot be multilingual without explicit language conditioning

**Language routing** is the correct design choice when:
- Each language has meaningfully different phoneme inventories and prosodic rules
- You have specialist models (MMS) that were trained per-language
- You want predictable behavior — a Spanish model always produces Spanish-quality speech

The router in this project detects language first, then selects the model. No cross-attention alignment issues, no language confusion, and each model is as good as it can be for its language.

---

## 4. G2P — Grapheme-to-Phoneme

G2P converts written text to phoneme sequences. Complexity varies by language:

| Language | Regularity | Challenge |
|----------|------------|-----------|
| Spanish | Very high | Almost fully regular spelling-pronunciation correspondence |
| German | High | Regular with a few exceptions |
| English | Low | Highly irregular: "read/read", "bow/bow", proper nouns |
| French | Medium | Silent letters, liaison rules |
| Arabic | Low | Short vowels usually omitted in text |
| Chinese | None | Characters don't encode pronunciation at all — requires dictionary or model |
| Japanese | Medium | Three scripts (hiragana, katakana, kanji) with different rules |

**MMS approach:** convert all languages to IPA first, using language-specific G2P tools (eSpeak, espeak-ng, language-specific dictionaries). The VITS model then works entirely in IPA space.

---

## 5. Neural vocoders

The vocoder converts mel spectrogram → waveform.

| Vocoder | Type | RTF (CPU) | Quality |
|---------|------|-----------|---------|
| WaveNet | Dilated causal CNN, AR | ~1000× slower than real time | Excellent |
| WaveRNN | RNN, AR | Slow | Good |
| WaveGlow | Normalizing flow | ~20× real time | Good |
| HiFi-GAN | GAN | ~50× faster than real time | Excellent |
| EnCodec | VQ-VAE codec | Fast | High (with codec artifacts) |

**HiFi-GAN** is the current standard. Its discriminators:
- **Multi-Period Discriminator (MPD):** looks at waveform samples with strides of 2, 3, 5, 7, 11 — catches periodic patterns at different frequencies
- **Multi-Scale Discriminator (MSD):** looks at the waveform at 3 different resolutions — catches both fine-grained waveform details and coarse spectral structure

Together they ensure the generated waveform is realistic at every time scale.

---

## 6. Speaker embeddings

Speaker embeddings encode a speaker's vocal identity (timbre, accent, speaking style) in a fixed-dimensional vector.

**Extraction:** a neural speaker verification model trained with a loss that pushes same-speaker embeddings together and different-speaker embeddings apart (GE2E, ArcFace, etc.).

**Conditioning in TTS:** the embedding is injected into the decoder at every step — either concatenated to the input or via FiLM (feature-wise linear modulation: scale and shift the layer activations using the embedding).

**Zero-shot voice cloning:** extract speaker embedding from 3–30 seconds of reference audio, inject it into a multi-speaker TTS model. No fine-tuning. Output sounds like the reference speaker saying arbitrary text.

**Limitation:** the cloned voice has the reference speaker's timbre but the model's prosody patterns. Emotional range and style transfer are still active research areas.

---

## 7. Multilingual challenges

### Phoneme inventory

Every language has a different phoneme set. English ~44, Mandarin ~21 consonants + 4 tones, Hawaiian ~13. A multilingual TTS model using a shared phoneme space must either:
- Use a universal set (IPA — all languages' phonemes represented)
- Use language-specific sets with language conditioning

MMS uses IPA per language. SpeechT5 uses character-level input for English.

### Prosody typology

**Stress-timed:** English, German, Dutch. Stressed syllables at roughly equal intervals. Unstressed syllables compressed.

**Syllable-timed:** French, Spanish, Italian. All syllables roughly equal duration.

**Mora-timed:** Japanese. The mora (not syllable) is the timing unit.

**Tonal:** Mandarin (4 tones), Cantonese (6 tones), Thai (5 tones), Vietnamese (6 tones). Pitch encodes lexical meaning — the TTS model must produce the correct F0 contour per syllable.

A model trained only on English will impose English rhythm on other languages, producing unnatural-sounding output.

### Code-switching

Code-switching is the alternation between languages within a conversation or sentence. Hard for TTS because:
- The language detector must identify the switch boundary
- A different TTS model must be selected mid-utterance
- Prosody must transition naturally across the language boundary

Production handling: run a sliding-window language detector, flush the current TTS buffer at the switch point, re-instantiate with the new language model.

---

## 8. Whisper — ASR and language detection

**Architecture:**
- Audio → 80-channel mel spectrogram (30-second chunks)
- CNN frontend extracts local features
- Transformer encoder produces audio representations
- Transformer decoder generates text autoregressively
- Special tokens control task: `<|en|>` (language), `<|transcribe|>` (task), `<|notimestamps|>`

**Language detection mechanism:** the first token the decoder predicts is the language ID token (e.g. `<|es|>` for Spanish). This is trained discriminatively — Whisper sees labeled (audio, language) pairs. Language detection accuracy is high for languages with substantial training data (~98% for major European languages) and lower for rare languages.

**Why this matters for this project:** Whisper gives us both the transcription and the language code in a single forward pass. We feed the language code to the TTS router directly — no separate language identification model needed.

---

## 9. Agentic AI fundamentals

### ReAct pattern

ReAct (Reason + Act, Yao et al. 2022) structures agent behavior as an alternating sequence:

```
Thought: I need to find current weather in Madrid.
Action: web_search
Action Input: Madrid weather today
Observation: 22°C, sunny skies.
Thought: I have enough to answer. The user asked in Spanish, so I'll respond in Spanish.
Final Answer: Hoy en Madrid hace 22 grados y el cielo está despejado.
```

Each thought-action-observation cycle is one step. The agent terminates when it produces a "Final Answer".

**Why it works:** forcing the model to write its reasoning before acting reduces errors (the model "thinks before acting") and produces an inspectable decision trace.

### Tool design

A good tool has:
- A precise, unambiguous name
- A description that tells the LLM *when* to use it (not just what it does)
- A simple string → string interface (no complex schemas)
- Graceful error handling (returns an error string, not an exception)

### Memory taxonomy

| Type | What it stores | Where |
|------|----------------|-------|
| Working | Current conversation | Context window |
| Episodic | Past interactions, retrieved by similarity | FAISS + embeddings |
| Semantic | World knowledge | LLM weights |
| Procedural | How to use tools | Tool definitions in system prompt |

### Latency budget for voice agents

Target: < 2 seconds from end of user speech to start of agent speech.

| Component | Typical latency |
|-----------|----------------|
| Whisper ASR (base, CPU) | 200–500ms |
| LLM first token (API) | 300–800ms |
| LLM full response | 500–2000ms |
| TTS (SpeechT5, CPU) | 200–500ms |
| TTS (MMS, CPU) | 200–500ms |
| TTS (Bark, CPU) | 5–20 seconds |

**Streaming TTS** is required to hit the budget: flush the first sentence to TTS as soon as the LLM generates it, start playing audio, generate the rest of the response concurrently.

---

## 10. TTS evaluation

**Mean Opinion Score (MOS):** humans rate naturalness 1–5. MOS ≥ 4.0 is near-human. Expensive.

**UTMOS:** neural MOS predictor trained on human ratings. Cheap proxy.

**WER intelligibility test:** run Whisper on the TTS output, compute WER against the input text. Measures whether the synthesized speech is understandable. WER > 5% indicates synthesis errors.

**MCD (Mel Cepstral Distortion):** L2 distance in mel cepstrum space vs ground truth. Correlates imperfectly with perceived quality.

**RTF (Real-Time Factor):** RTF = synthesis time / audio duration. RTF < 1.0 required for production. HiFi-GAN: RTF ≈ 0.01. Bark: RTF ≈ 3–10 on CPU.

**Speaker similarity:** cosine similarity between ECAPA-TDNN embeddings of synthesized vs reference audio. Used for voice cloning evaluation.

---

## 11. Questions to be ready to answer cold

**TTS architecture**
- Walk me through Tacotron 2 step by step.
- Why did FastSpeech replace attention with a duration predictor?
- What is a normalizing flow? Why is it used in VITS?
- What do HiFi-GAN's MPD and MSD discriminators each catch?
- What does it mean for TTS to be "end-to-end"?

**Multilingual**
- How does MMS handle 1107 languages with one architecture? (Separate VITS checkpoints per language, IPA input)
- What is G2P and why is Chinese harder than Spanish?
- What is code-switching and how would you handle it in a voice assistant?
- What is the difference between stress-timed and syllable-timed prosody?

**Speaker and voice**
- What is a speaker embedding? How is it injected into a TTS model?
- What is zero-shot voice cloning?

**ASR**
- How does Whisper detect the language of the input? (First decoder token is the language ID token)
- What are Whisper's known failure modes?

**Agentic AI**
- Describe the ReAct loop. Draw the Thought/Action/Observation cycle.
- What is RAG? How does FAISS fit in?
- What are the latency bottlenecks in a voice agent, and how do you mitigate them?

**System design**
- Design a voice assistant that supports 10 languages with a 2-second latency target.
- How would you evaluate TTS quality at scale without human annotators?
