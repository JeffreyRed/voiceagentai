"""
src/api.py

FastAPI backend — serves the web demo and exposes REST endpoints.

Local development:
    uvicorn src.api:app --reload --port 8000
"""

import io
import os
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from src.tts.router import TTSRouter
from src.utils.lang_detect import detect_language

app = FastAPI(title="VoiceAgent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded singleton
_tts: TTSRouter | None = None


def get_tts() -> TTSRouter:
    global _tts
    if _tts is None:
        _tts = TTSRouter()
    return _tts


@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the frontend demo page."""
    frontend_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "frontend", "index.html"
    )
    with open(frontend_path, "r") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    lang: str = Form(default="en"),
):
    """Convert text to speech. Returns a WAV file."""
    wav_bytes = get_tts().synthesize(text, lang=lang)
    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )


@app.post("/chat")
async def chat(
    text: str = Form(...),
    lang: str = Form(default=""),
):
    """
    Synthesize text and return JSON with base64 audio.
    The frontend plays it directly without saving a file.
    """
    import base64
    detected_lang = lang if lang else detect_language(text)
    wav_bytes = get_tts().synthesize(text, lang=detected_lang)
    audio_b64 = base64.b64encode(wav_bytes).decode()
    return JSONResponse({
        "text": text,
        "language": detected_lang,
        "audio_base64": audio_b64,
    })


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Transcribe an uploaded audio file. Returns text and detected language."""
    import tempfile
    suffix = os.path.splitext(audio.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        from src.asr.whisper_asr import WhisperASR
        asr = WhisperASR(model_size="base")
        result = asr.transcribe(tmp_path)
    finally:
        os.unlink(tmp_path)
    return JSONResponse(result)