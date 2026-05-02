"""
src/tts/router.py

Routes (text, language_code) to the best pretrained TTS model.

Design decision — why routing instead of one polyglot model:
  MMS-TTS trains a separate VITS model per language using IPA phoneme input.
  Routing by language code is the correct abstraction: each model is a
  specialist, not a generalist. This mirrors how production multilingual
  voice assistants (including Siri) manage language-specific acoustic models.

Priority:
  1. SpeechT5 for English  — highest quality for en, native 16kHz output
  2. MMS-TTS for the 1100+ supported languages
  3. Bark as expressive fallback for unsupported languages
"""

from __future__ import annotations

SPEECHT5_LANGS = {"en"}

# Subset of MMS-TTS supported ISO 639-1 codes (full list: 1100+ ISO 639-3)
MMS_SUPPORTED = {
    "es", "fr", "de", "it", "pt", "nl", "ru", "ar",
    "hi", "zh", "ja", "ko", "tr", "pl", "vi", "th",
    "sw", "yo", "ig", "ha",
}


class TTSRouter:
    """
    Lazily loads and caches TTS model instances by language.
    Models are only downloaded on first use.
    """

    def __init__(self):
        self._cache: dict = {}

    def synthesize(self, text: str, lang: str = "en") -> bytes:
        """
        Synthesize text in the given language. Returns raw WAV bytes.

        Args:
            text: The text to synthesize.
            lang: ISO 639-1 language code (e.g. "en", "es", "zh").
        """
        import io
        import soundfile as sf
        import numpy as np

        model = self._get_model(lang)
        waveform: np.ndarray = model.synthesize(text)
        sample_rate = self._sample_rate(lang)

        buf = io.BytesIO()
        sf.write(buf, waveform, samplerate=sample_rate, format="WAV")
        return buf.getvalue()

    def _get_model(self, lang: str):
        if lang not in self._cache:
            self._cache[lang] = self._load_model(lang)
        return self._cache[lang]

    def _load_model(self, lang: str):
        if lang in SPEECHT5_LANGS:
            from src.tts.speecht5 import SpeechT5TTS
            return SpeechT5TTS()
        elif lang in MMS_SUPPORTED:
            from src.tts.mms import MMSTTS
            return MMSTTS(lang=lang)
        else:
            from src.tts.bark import BarkTTS
            return BarkTTS()

    def _sample_rate(self, lang: str) -> int:
        if lang in SPEECHT5_LANGS:
            return 16_000
        elif lang in MMS_SUPPORTED:
            return 16_000  # MMS models output 16kHz
        else:
            return 24_000  # Bark outputs 24kHz
