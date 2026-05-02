"""
src/tts/mms.py

Multilingual TTS using Meta's MMS-TTS models (VITS-based, 1100+ languages).
Each language is a separate HuggingFace checkpoint: facebook/mms-tts-{lang}

Language codes use ISO 639-3 (3-letter). This module handles the mapping
from common ISO 639-1 (2-letter) codes to the MMS checkpoint names.
"""

import numpy as np
import torch
from transformers import VitsModel, VitsTokenizer

# Mapping from ISO 639-1 (2-letter) → MMS checkpoint suffix (ISO 639-3)
LANG_MAP = {
    "es": "spa",  # Spanish
    "fr": "fra",  # French
    "de": "deu",  # German
    "it": "ita",  # Italian
    "pt": "por",  # Portuguese
    "nl": "nld",  # Dutch
    "ru": "rus",  # Russian
    "ar": "ara",  # Arabic
    "hi": "hin",  # Hindi
    "zh": "cmn",  # Mandarin Chinese
    "ja": "jpn",  # Japanese
    "ko": "kor",  # Korean
    "tr": "tur",  # Turkish
    "pl": "pol",  # Polish
    "vi": "vie",  # Vietnamese
    "th": "tha",  # Thai
    "sw": "swh",  # Swahili
    "yo": "yor",  # Yoruba
    "ig": "ibo",  # Igbo
    "ha": "hau",  # Hausa
}


class MMSTTS:
    """
    Wrapper for facebook/mms-tts-{lang} checkpoints.
    Loads model lazily on first call per language.
    """

    def __init__(self, lang: str):
        """
        Args:
            lang: ISO 639-1 language code (e.g. "es", "fr", "zh")
        """
        iso3 = LANG_MAP.get(lang, lang)  # fall back to the code itself
        self.model_id = f"facebook/mms-tts-{iso3}"
        self.lang = lang
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is None:
            print(f"[MMS-TTS] Loading {self.model_id}...")
            self._tokenizer = VitsTokenizer.from_pretrained(self.model_id)
            self._model = VitsModel.from_pretrained(self.model_id)
            self._model.eval()
            print(f"[MMS-TTS] {self.model_id} ready.")

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize text to a numpy float32 waveform.

        Returns:
            numpy array, shape (N,), sample rate varies by model (~16kHz).
        """
        self._load()
        inputs = self._tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output = self._model(**inputs)
        waveform = output.waveform[0].numpy()  # (N,) float32
        return waveform
