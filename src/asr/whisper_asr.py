"""
src/asr/whisper_asr.py

Speech-to-text using openai/whisper-base.
Returns both the transcribed text and the detected language code,
which is fed directly to the TTS router to ensure response language matches input.

Key insight: Whisper detects language as a side effect of its first decoder step.
The model predicts a special <|lang_code|> token before generating any text.
This gives us language identification for free — no separate LID model needed for audio.
"""

import whisper
import numpy as np


class WhisperASR:
    """Wrapper for Whisper ASR with language detection."""

    def __init__(self, model_size: str = "base"):
        """
        Args:
            model_size: One of 'tiny', 'base', 'small', 'medium', 'large'.
                        'base' is a good balance for local development (~150MB).
        """
        print(f"[Whisper] Loading model '{model_size}'...")
        self.model = whisper.load_model(model_size)
        print("[Whisper] Ready.")

    def transcribe(self, audio_path: str) -> dict:
        """
        Transcribe an audio file and detect its language.

        Args:
            audio_path: Path to a WAV, MP3, or FLAC file.

        Returns:
            {
                "text": str,           # transcribed text
                "language": str,       # ISO 639-1 code (e.g. "en", "es")
                "language_prob": float # confidence of language detection
            }
        """
        result = self.model.transcribe(audio_path, task="transcribe")
        return {
            "text": result["text"].strip(),
            "language": result["language"],
            "language_prob": result.get("language_probs", {}).get(result["language"], 1.0),
        }

    def transcribe_array(self, audio: np.ndarray, sample_rate: int = 16000) -> dict:
        """
        Transcribe a numpy audio array.

        Args:
            audio: float32 numpy array, any sample rate (resampled internally to 16kHz).
            sample_rate: Sample rate of the input array.
        """
        import torch
        if sample_rate != 16000:
            import torchaudio
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
            audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, 16000)
            audio = audio_tensor.squeeze(0).numpy()

        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
        _, probs = self.model.detect_language(mel)
        detected_lang = max(probs, key=probs.get)

        result = self.model.transcribe(audio, task="transcribe")
        return {
            "text": result["text"].strip(),
            "language": detected_lang,
            "language_prob": probs[detected_lang],
        }
