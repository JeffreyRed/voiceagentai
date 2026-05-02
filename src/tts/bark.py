"""
src/tts/bark.py

Expressive TTS using suno/bark-small (or suno/bark for higher quality).
Used as fallback when MMS does not cover the requested language.

Note: Bark is slow on CPU (RTF ≈ 3–10). Use bark-small for development.
"""

import numpy as np

BARK_SAMPLE_RATE = 24000


class BarkTTS:
    """Wrapper for Bark generative TTS."""

    def __init__(self, model_size: str = "small"):
        self._model_size = model_size
        self._preload = None  # lazy load

    def _load(self):
        if self._preload is None:
            from transformers import AutoProcessor, BarkModel
            model_id = "suno/bark-small" if self._model_size == "small" else "suno/bark"
            print(f"[Bark] Loading {model_id}... (this may take a while)")
            self._processor = AutoProcessor.from_pretrained(model_id)
            self._model = BarkModel.from_pretrained(model_id)
            self._model.eval()
            self._preload = True
            print("[Bark] Ready.")

    def synthesize(self, text: str, voice_preset: str = "v2/en_speaker_6") -> np.ndarray:
        """
        Synthesize text to a numpy float32 waveform at 24000 Hz.

        Args:
            text: Input text. Supports [laughter], [sighs], [music] tokens.
            voice_preset: Bark speaker preset. See bark documentation for options.

        Returns:
            numpy array, shape (N,), sample rate 24000 Hz.
        """
        self._load()
        import torch
        inputs = self._processor(text, voice_preset=voice_preset)
        with torch.no_grad():
            audio_array = self._model.generate(**inputs)
        return audio_array.cpu().numpy().squeeze()
