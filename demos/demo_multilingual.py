"""
demos/demo_multilingual.py

Synthesize the same greeting in multiple languages using the TTS router.
Saves one WAV file per language to assets/sample_outputs/.

Run:
    conda activate voiceagent
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
    "de": "Hallo! Ich bin Ihr mehrsprachiger Sprachassistent. Wie kann ich Ihnen helfen?",
    "zh": "你好！我是您的多语言语音助手。今天我能帮您什么？",
    "sw": "Habari! Mimi ni msaidizi wako wa sauti wa lugha nyingi. Ninaweza kukusaidia vipi leo?",
}

OUTPUT_DIR = Path("assets/sample_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default=None, help="ISO 639-1 code, or all languages if omitted")
    args = parser.parse_args()

    router = TTSRouter()
    langs = [args.lang] if args.lang else list(SAMPLES.keys())

    for lang in langs:
        text = SAMPLES.get(lang, SAMPLES["en"])
        print(f"\n[{lang}] {text[:70]}...")
        wav_bytes = router.synthesize(text, lang=lang)

        out_path = OUTPUT_DIR / f"{lang}_sample.wav"
        with open(out_path, "wb") as f:
            f.write(wav_bytes)
        print(f"  → Saved: {out_path}")

    print("\nDone. Open the WAV files in any audio player.")


if __name__ == "__main__":
    main()
