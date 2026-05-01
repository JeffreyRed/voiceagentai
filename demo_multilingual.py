"""
demos/demo_multilingual.py

Demonstrates the TTS router by synthesizing the same question
in multiple languages and saving WAV files to assets/sample_outputs/.

Usage:
    python demos/demo_multilingual.py
    python demos/demo_multilingual.py --lang es
"""

import argparse
import soundfile as sf
from pathlib import Path
from src.tts.router import TTSRouter

SAMPLES = {
    "en": "Hello! I am your multilingual voice assistant. How can I help you today?",
    "es": "¡Hola! Soy tu asistente de voz multilingüe. ¿En qué puedo ayudarte hoy?",
    "fr": "Bonjour ! Je suis votre assistant vocal multilingue. Comment puis-je vous aider ?",
    "de": "Hallo! Ich bin Ihr mehrsprachiger Sprachassistent. Wie kann ich Ihnen heute helfen?",
    "zh": "你好！我是您的多语言语音助手。今天我能帮您什么？",
}

OUTPUT_DIR = Path("assets/sample_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default=None, help="ISO 639-1 language code, or all if omitted")
    args = parser.parse_args()

    router = TTSRouter()
    langs = [args.lang] if args.lang else list(SAMPLES.keys())

    for lang in langs:
        text = SAMPLES.get(lang, SAMPLES["en"])
        print(f"[{lang}] Synthesizing: {text[:60]}...")
        audio_bytes = router.synthesize(text, lang=lang)
        out_path = OUTPUT_DIR / f"{lang}_sample.wav"
        # audio_bytes is a numpy array (22050 Hz, mono) from the model wrappers
        sf.write(str(out_path), audio_bytes, samplerate=22050)
        print(f"  → Saved: {out_path}")


if __name__ == "__main__":
    main()
