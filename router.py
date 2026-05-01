"""
tts/router.py

Routes a (text, language_code) pair to the best available TTS model.

Priority:
  1. SpeechT5 for English (highest quality for en)
  2. MMS-TTS for the 1100+ supported languages (best multilingual coverage)
  3. Bark as expressive fallback
"""

from __future__ import annotations

SPEECHT5_LANGS = {"en"}

MMS_SUPPORTED = {
    # subset shown — MMS covers 1100+ ISO 639-3 codes
    "es", "fr", "de", "it", "pt", "nl", "ru", "ar",
    "hi", "zh", "ja", "ko", "tr", "pl", "vi", "th",
    "sw", "yo", "ig", "ha",
}


class TTSRouter:
    """Select and cache TTS model instances by language."""

    def __init__(self):
        self._cache: dict = {}

    def synthesize(self, text: str, lang: str = "en") -> bytes:
        """Return synthesized audio as raw WAV bytes."""
        model = self._get_model(lang)
        return model.synthesize(text)

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
