"""
tests/test_tts_router.py

Unit tests for the TTS router logic (no model downloads — tests routing decisions only).
"""
import pytest
from unittest.mock import MagicMock, patch
from src.tts.router import TTSRouter, SPEECHT5_LANGS, MMS_SUPPORTED


def test_english_routes_to_speecht5():
    router = TTSRouter()
    with patch("src.tts.router.TTSRouter._load_model") as mock_load:
        mock_model = MagicMock()
        mock_model.synthesize.return_value = b"\x00" * 100
        mock_load.return_value = mock_model
        router._cache.clear()
        router._get_model("en")
        mock_load.assert_called_once_with("en")


def test_spanish_routes_to_mms():
    assert "es" in MMS_SUPPORTED
    assert "es" not in SPEECHT5_LANGS


def test_unsupported_lang_falls_back_to_bark():
    router = TTSRouter()
    with patch("src.tts.bark.BarkTTS") as MockBark:
        model = router._load_model("xx")  # unknown language
        # Should instantiate Bark
        assert model is not None


def test_sample_rate_english():
    router = TTSRouter()
    assert router._sample_rate("en") == 16_000


def test_sample_rate_bark_fallback():
    router = TTSRouter()
    assert router._sample_rate("xx") == 24_000


def test_mms_supported_has_major_languages():
    for lang in ["es", "fr", "de", "zh", "ar", "hi", "ja", "ko"]:
        assert lang in MMS_SUPPORTED, f"{lang} should be in MMS_SUPPORTED"
