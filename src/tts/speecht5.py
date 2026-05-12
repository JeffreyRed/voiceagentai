"""
src/tts/speecht5.py

English TTS using microsoft/speecht5_tts + HiFi-GAN vocoder.
Speaker embedding loaded from the parquet file in the dataset
(the only format that actually exists and works).
"""

import numpy as np
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan


MODEL_ID = "microsoft/speecht5_tts"
VOCODER_ID = "microsoft/speecht5_hifigan"


class SpeechT5TTS:
    """Wrapper around SpeechT5 for English TTS."""

    def __init__(self):
        print(f"[SpeechT5] Loading {MODEL_ID}...")
        self.processor = SpeechT5Processor.from_pretrained(MODEL_ID)
        self.model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_ID)
        self.vocoder = SpeechT5HifiGan.from_pretrained(VOCODER_ID)
        self.speaker_embedding = self._get_speaker_embedding()
        self.model.eval()
        self.vocoder.eval()
        print("[SpeechT5] Ready.")

    def _get_speaker_embedding(self) -> torch.Tensor:
        """
        Load speaker embedding from the parquet version of the dataset.
        This is the format HuggingFace converts all datasets to automatically.
        Index 7306 = BDL (US male voice).
        """
        import urllib.request
        import io

        print("[SpeechT5] Loading speaker embedding from parquet...")

        # HuggingFace auto-converts all datasets to parquet — this URL is stable
        url = (
            "https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors"
            "/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"
        )

        with urllib.request.urlopen(url) as resp:
            data = resp.read()

        import pyarrow.parquet as pq
        table = pq.read_table(io.BytesIO(data))

        # Row 7306 = BDL speaker
        xvector = table["xvector"][7306].as_py()
        embedding = torch.tensor(xvector).unsqueeze(0)  # (1, 512)
        print("[SpeechT5] Speaker embedding loaded.")
        return embedding

    def synthesize(self, text: str) -> np.ndarray:
        """
        Synthesize text to a numpy float32 waveform at 16000 Hz.
        """
        inputs = self.processor(text=text, return_tensors="pt")
        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"],
                self.speaker_embedding,
                vocoder=self.vocoder,
            )
        return speech.numpy()  # float32, 16kHz