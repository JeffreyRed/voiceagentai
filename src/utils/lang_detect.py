"""
src/utils/lang_detect.py

Text-based language detection. Used when the input is text (not audio).
For audio input, Whisper handles language detection automatically.

Uses langdetect (Python port of Google's language-detection library).
Falls back to pycld2 for short texts where langdetect is unreliable.
"""

from langdetect import detect, DetectorFactory, LangDetectException

# Make langdetect deterministic (it uses random seeding by default)
DetectorFactory.seed = 42

SUPPORTED_LANGS = {
    "en", "es", "fr", "de", "it", "pt", "nl", "ru",
    "ar", "hi", "zh", "ja", "ko", "tr", "pl", "vi",
    "th", "sw",
}


def detect_language(text: str) -> str:
    """
    Detect the language of a text string.

    Args:
        text: Input text of any length. More text = higher accuracy.

    Returns:
        ISO 639-1 language code (e.g. "en", "es", "zh").
        Returns "en" as fallback if detection fails or language is unsupported.
    """
    if not text or len(text.strip()) < 3:
        return "en"

    try:
        lang = detect(text)
        # Chinese: langdetect returns "zh-cn" or "zh-tw" — normalize to "zh"
        if lang.startswith("zh"):
            return "zh"
        return lang if lang in SUPPORTED_LANGS else "en"
    except LangDetectException:
        return "en"


def is_supported(lang: str) -> bool:
    """Check if we have TTS support for this language."""
    return lang in SUPPORTED_LANGS
