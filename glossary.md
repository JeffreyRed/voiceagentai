# Glossary — TTS & Agentic Voice AI

Key terms you should be able to define confidently in a technical interview.
Organized by topic. Cross-references to `theory.md` where relevant.

---

## Audio & Signal Processing

**Waveform**
The raw time-domain audio signal — a sequence of amplitude values sampled at a fixed rate (e.g. 22,050 samples/second for speech). Everything a TTS system ultimately produces is a waveform.

**Sample rate (Hz)**
How many audio samples are captured per second. 16 kHz is standard for speech models (Whisper, SpeechT5). 22.05 kHz or 24 kHz is common for TTS output. Higher = better quality but more compute.

**Mel spectrogram**
A 2D time-frequency representation of audio, where the frequency axis is warped to the mel scale (logarithmic, matching human hearing). TTS acoustic models almost always predict mel spectrograms as an intermediate representation, which a vocoder then converts to a waveform.

**MFCC (Mel-Frequency Cepstral Coefficients)**
A compressed version of the mel spectrogram, computed by applying a DCT. Used heavily in classical ASR (HMM/GMM era). Less common in modern deep learning TTS, which prefers raw mel spectrograms.

**Mel scale**
A perceptual frequency scale where equal distances correspond to equal perceived pitch differences. Humans hear differences between 100 Hz and 200 Hz much more clearly than between 5000 Hz and 5100 Hz. The mel scale reflects this.

**Fundamental frequency (F0 / pitch)**
The rate at which the vocal cords vibrate, perceived as pitch. F0 contour (how pitch changes over time) is a key component of prosody and is explicitly predicted by FastSpeech 2 and VITS.

**Prosody**
The suprasegmental features of speech: rhythm, stress, intonation, and tempo. Two sentences with identical words but different prosody carry different meanings ("Really?" vs "Really."). Hard to control in TTS without explicit prosody prediction modules.

**Phoneme**
The smallest unit of sound that changes meaning in a language. English has ~44 phonemes. /p/ and /b/ are different phonemes ("pat" vs "bat"). Phonemes are language-specific — the same acoustic sound can be two different phonemes in one language and one phoneme in another.

**Allophone**
A variant pronunciation of a phoneme that does not change meaning. The /p/ in "pin" is aspirated; the /p/ in "spin" is not. Both are allophones of /p/ in English.

**Phoneme inventory**
The complete set of phonemes a language uses. English ≈ 44, Mandarin ≈ 21 consonants + 4 tones, Hawaiian ≈ 13. A multilingual TTS model must handle the union of all target languages' inventories.

**IPA (International Phonetic Alphabet)**
A standardized notation for phonemes across all human languages. Used as the intermediate representation in MMS-TTS so the model is script-agnostic. `/θ/` is the IPA symbol for the "th" sound in "think".

**Grapheme**
A written character or letter. The grapheme "c" can produce phonemes /k/ (cat) or /s/ (city) depending on context.

**G2P (Grapheme-to-Phoneme)**
The conversion of written text to a phoneme sequence. Straightforward in Spanish (highly regular spelling), complex in English (many exceptions), and requires a dictionary or model in Chinese (characters don't encode pronunciation).

**Voiced / Unvoiced**
Voiced sounds involve vocal cord vibration (/b/, /d/, /z/). Unvoiced do not (/p/, /t/, /s/). Important for TTS models to get right — confusing voiced/unvoiced pairs is a common synthesis error.

---

## TTS Architectures

**End-to-end TTS**
A TTS system where a single model takes text input and produces a waveform directly, with no separate vocoder. VITS is the main example. Contrast with pipeline TTS (acoustic model + vocoder).

**Acoustic model**
The component that maps a phoneme or text sequence to an acoustic representation (usually a mel spectrogram). Examples: Tacotron 2 decoder, FastSpeech 2, the SpeechT5 decoder.

**Vocoder**
The component that converts a mel spectrogram (or other intermediate representation) back into a waveform. Examples: WaveNet, HiFi-GAN, WaveRNN.

**Autoregressive (AR) model**
A model that generates output one step at a time, where each step depends on all previous outputs. Tacotron 2 and Bark are autoregressive. Slow to generate but often high quality because the model can condition on its own past outputs.

**Non-autoregressive (NAR) model**
A model that generates all output steps in parallel. FastSpeech 2 is non-autoregressive. Much faster than AR models, but quality can be slightly lower because there is no step-by-step feedback.

**Duration predictor**
A module (used in FastSpeech 2, VITS) that predicts how many spectrogram frames each phoneme should produce. Replaces the attention alignment mechanism of Tacotron 2. Makes synthesis speed more predictable and controllable.

**HiFi-GAN**
The dominant neural vocoder for open-source TTS. A GAN architecture with one generator (upsamples mel to waveform) and multiple discriminators: a multi-period discriminator (MPD) and multi-scale discriminator (MSD). Together they catch both fine-grained waveform artifacts and coarse spectral errors.

**VITS (Variational Inference with adversarial learning for end-to-end TTS)**
An end-to-end TTS model that combines a variational autoencoder (for latent acoustic modeling), normalizing flows (for flexible posterior modeling), and a GAN discriminator (for waveform quality). Produces high-quality, expressive speech in real time.

**Normalizing flow**
A generative model that learns an invertible mapping between a simple distribution (e.g. Gaussian) and a complex data distribution. VITS uses flows to model the relationship between the text-conditioned prior and the speech posterior. Flows allow exact likelihood computation, unlike VAEs.

**SpeechT5**
Microsoft's unified speech-text transformer. Pre-trained jointly on both speech and text data. The shared encoder allows cross-modal transfer — the model understands relationships between text tokens and speech tokens. Used in this project for English TTS.

**MMS-TTS (Massively Multilingual Speech TTS)**
Meta's VITS-based TTS models covering 1100+ languages, trained on New Testament audio recordings. Uses IPA phoneme input, making it script-agnostic. Each language is a separate model checkpoint with the same architecture. Key insight: routing by language code, not a single polyglot model.

**Bark**
Suno's GPT-style generative audio model. Generates speech by predicting discrete audio codec tokens (from EnCodec) in three stages: semantic → coarse acoustic → fine acoustic. Supports paralinguistic cues (laughter, sighs). Slow but expressive.

**Neural codec**
A model that compresses audio into discrete tokens (like a codebook of audio "words"). Examples: EnCodec (Meta), SoundStream (Google). Used as the tokenizer in Bark and AudioLM. Enables treating audio generation as a language modeling problem.

---

## Speaker & Voice

**Speaker embedding**
A fixed-dimensional vector (typically 192–512 dimensions) that encodes a speaker's vocal identity — timbre, accent, speaking style. Extracted from a few seconds of reference audio. Used to condition multi-speaker TTS models.

**x-vector**
A speaker embedding produced by a TDNN (Time Delay Neural Network) trained on speaker verification. Used by SpeechT5 for multi-speaker conditioning.

**d-vector**
A speaker embedding from a deep neural speaker verification model trained with GE2E loss. The first widely-used neural speaker embedding for TTS conditioning.

**ECAPA-TDNN**
State-of-the-art speaker embedding model. Combines channel- and context-dependent statistics pooling with multi-scale feature aggregation. Produces robust embeddings from as little as 3 seconds of audio.

**Voice cloning**
Synthesizing a target speaker's voice for arbitrary text, using only a short reference recording (zero-shot) or a small fine-tuning set (few-shot). The speaker embedding from the reference is injected into the TTS model to condition the output.

**FiLM (Feature-wise Linear Modulation)**
A conditioning mechanism where a conditioning vector (e.g. speaker embedding) is used to scale and shift the activations of a neural network layer. Allows efficient injection of speaker identity or language information into intermediate representations without concatenation.

---

## Multilingual

**Code-switching**
When a speaker alternates between two or more languages within a single conversation or sentence ("I'm going al supermercado después"). A challenge for both ASR and TTS — the system must detect the switch and handle it gracefully.

**Language identification (LID)**
Automatically detecting which language a piece of text or audio is in. Done in this project with `langdetect` (text) and Whisper's decoder (audio). The output language code drives TTS model selection.

**Tonal language**
A language where pitch (tone) distinguishes word meaning. Mandarin has 4 tones + neutral (mā/má/mǎ/mà = mother/hemp/horse/scold). Thai has 5 tones. The TTS model must produce the correct F0 contour per syllable, or the meaning changes.

**Syllable-timed vs stress-timed**
Prosodic rhythm typology. Stress-timed languages (English, German) have roughly equal intervals between stressed syllables. Syllable-timed languages (French, Spanish) have roughly equal syllable duration. A TTS model trained only on English will impose wrong rhythm on French output.

**Diacritics**
Accent marks that modify a letter's pronunciation (é, ñ, ü, ç). Critical for correct G2P. Missing diacritics in TTS input often cause mispronunciation.

**Morphological richness**
Languages like Finnish, Turkish, and Arabic express grammatical relationships through word endings and internal changes (inflection, agglutination). This increases vocabulary size, making character-level or phoneme-level models more appropriate than word-level ones for these languages.

---

## ASR (Automatic Speech Recognition)

**Whisper**
OpenAI's seq2seq ASR model trained on 680k hours of weakly supervised audio. Takes 30-second mel spectrogram chunks as input, auto-detects language, and produces text. Key for this project: the first decoder token is the language ID, giving us language detection for free.

**WER (Word Error Rate)**
The standard ASR evaluation metric. WER = (Substitutions + Deletions + Insertions) / Total reference words. Lower is better. Whisper base achieves ~10% WER on English.

**CTC (Connectionist Temporal Classification)**
A loss function and decoding algorithm for sequence alignment without explicit frame-level labels. Used in many ASR models (wav2vec 2.0, HuBERT). Not used in Whisper (which uses cross-entropy seq2seq loss).

**Beam search**
A decoding strategy that keeps the top-k most probable partial sequences at each step, rather than greedily picking the single best token. Used in Whisper decoding. Produces better transcriptions than greedy decoding at the cost of compute.

---

## Agentic AI

**ReAct (Reason + Act)**
An agentic prompting pattern where the LLM alternates between Thought (reasoning about what to do), Action (calling a tool), and Observation (receiving the tool result). Produces inspectable, debuggable agent behavior.

**Tool use**
Allowing an LLM to call external functions (web search, calculators, APIs) during generation. The model generates a structured tool call, the runtime executes it, and the result is fed back into the context.

**RAG (Retrieval-Augmented Generation)**
Augmenting an LLM's context with relevant documents retrieved from an external store. In this project: past Q&A turns are embedded and stored in FAISS; relevant past turns are retrieved and injected into the prompt before the LLM generates a response.

**FAISS (Facebook AI Similarity Search)**
A library for efficient approximate nearest-neighbor search in high-dimensional vector spaces. Used here to store and retrieve sentence embeddings for episodic memory.

**Vector embedding**
A dense numerical representation of text (or audio) in a high-dimensional space where semantically similar items are geometrically close. Produced by sentence-transformers (e.g. `all-MiniLM-L6-v2`).

**Episodic memory**
Memory of specific past events or interactions, retrieved based on similarity to the current context. Contrast with semantic memory (general world knowledge, baked into LLM weights) and working memory (current conversation in the context window).

**Agent loop**
The repeated cycle of: observe state → reason → act → observe new state. The agent terminates when it decides it has a final answer or hits a step limit.

**Hallucination**
When an LLM generates confident but factually incorrect statements. Particularly dangerous in a voice assistant because users cannot easily verify audio output. Tool use (web search) and RAG reduce hallucination by grounding responses in retrieved facts.

**Latency budget**
In a voice agent: the total acceptable delay between user speech ending and agent speech beginning. Typical target: under 2 seconds. Components: ASR (~200ms), LLM generation (~500ms–2s), TTS (~200ms–1s). Streaming (chunk-wise TTS) is necessary to hit this budget.

**Streaming TTS**
Generating and playing audio in chunks as the LLM produces text, rather than waiting for the full response. Requires sentence-level flushing: send the first complete sentence to TTS, start playing, generate the rest concurrently.

---

## Evaluation

**MOS (Mean Opinion Score)**
The gold standard for TTS evaluation. Human listeners rate naturalness on a 1–5 scale. MOS ≥ 4.0 is considered near-human. Expensive and slow to collect.

**UTMOS**
An automatic MOS predictor trained on human ratings. Acts as a proxy for MOS without needing human annotators. Useful for comparing TTS systems at scale.

**MCD (Mel Cepstral Distortion)**
An objective metric measuring the distance between the predicted mel cepstrum and the ground truth. Lower = closer to reference. Correlates imperfectly with perceived quality.

**RTF (Real-Time Factor)**
RTF = generation time / audio duration. RTF < 1.0 means the model generates audio faster than real time (required for production). HiFi-GAN RTF ≈ 0.01; Bark RTF ≈ 3–10 (slower than real time on CPU).

**Speaker similarity**
Cosine similarity between speaker embeddings of generated and reference audio. Used to evaluate voice cloning quality. Range: −1 to 1, with > 0.85 considered good cloning.

**Intelligibility**
Whether the synthesized speech can be correctly transcribed. Measured by running an ASR model on TTS output and computing WER against the input text. An intelligibility WER > 5% indicates significant synthesis errors.
