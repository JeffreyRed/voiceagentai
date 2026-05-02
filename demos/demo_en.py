"""
demos/demo_en.py

Simplest demo: synthesize one English sentence and save to a WAV file.
No API key needed — uses local models only.

Run:
    conda activate voiceagent
    python demos/demo_en.py
"""

import soundfile as sf
from pathlib import Path
from src.tts.speecht5 import SpeechT5TTS

TEXT = "Hello! I am a voice assistant powered by SpeechT5, a pretrained model from Microsoft."
OUTPUT = Path("assets/sample_outputs/en_demo.wav")
OUTPUT.parent.mkdir(parents=True, exist_ok=True)

print("Loading SpeechT5...")
tts = SpeechT5TTS()

print(f"Synthesizing: {TEXT}")
waveform = tts.synthesize(TEXT)

sf.write(str(OUTPUT), waveform, samplerate=16000)
print(f"Saved to {OUTPUT}")
