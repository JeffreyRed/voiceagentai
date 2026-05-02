"""
src/tts/speecht5.py

English TTS using microsoft/speecht5_tts + HiFi-GAN vocoder.
Speaker conditioned on a default CMU ARCTIC x-vector embedding.
"""

import numpy as np
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset


MODEL_ID = "microsoft/speecht5_tts"
VOCODER_ID = "microsoft/speecht5_hifigan"
EMBEDDINGS_DATASET = "Matthijs/cmu-arctic-xvectors"
DEFAULT_SPEAKER_IDX = 7306  # BDL (male) from CMU ARCTIC


class SpeechT5TTS:
    """Wrapper around SpeechT5 for English TTS."""

    def __init__(self):
        print(f"[SpeechT5] Loading {MODEL_ID}...")
        self.processor = SpeechT5Processor.from_pretrained(MODEL_ID)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_ID)
        self.vocoder = SpeechT5HifiGan.from_pretrained(VOCODER_ID)
        self.speaker_embedding = self._load_speaker_embedding()
        self.model.eval()
        self.vocoder.eval()
        print("[SpeechT5] Ready.")

    def _load_speaker_embedding(self) -> torch.Tensor:
        """Load a default x-vector speaker embedding from CMU ARCTIC dataset."""
        embeddings_ds = load_dataset(EMBEDDINGS_DATASET, split="validation")
        row = embeddings_ds[DEFAULT_SPEAKER_IDX]
        embedding = torch.tensor(row["xvector"]).unsqueeze(0)  # (1, 512)
        return embedding

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize text to a numpy float32 waveform at 16000 Hz.

        Args:
            text: Input text (English). Max ~600 characters for clean output.

        Returns:
            numpy array, shape (N,), sample rate 16000 Hz.
        """
        inputs = self.processor(text=text, return_tensors="pt")
        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"],
                self.speaker_embedding,
                vocoder=self.vocoder,
            )
        return speech.numpy()  # float32, 16kHz
