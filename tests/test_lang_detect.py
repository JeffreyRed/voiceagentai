"""
tests/test_lang_detect.py

Tests for the text-based language detector.
"""
import pytest
from src.utils.lang_detect import detect_language, is_supported


def test_detect_english():
    assert detect_language("Hello, how are you today?") == "en"


def test_detect_spanish():
    assert detect_language("Hola, ¿cómo estás hoy?") == "es"


def test_detect_french():
    assert detect_language("Bonjour, comment allez-vous?") == "fr"


def test_detect_german():
    assert detect_language("Guten Morgen, wie geht es Ihnen?") == "de"


def test_empty_text_returns_english():
    assert detect_language("") == "en"
    assert detect_language("  ") == "en"


def test_very_short_text_returns_english():
    assert detect_language("ok") == "en"


def test_chinese_normalized():
    result = detect_language("你好，今天怎么样？")
    assert result == "zh"


def test_supported_langs():
    assert is_supported("en") is True
    assert is_supported("es") is True
    assert is_supported("xx") is False
