"""
src/api.py

FastAPI backend — serves the web demo and exposes REST endpoints.

Endpoints:
    POST /synthesize   — text + lang → WAV audio
    POST /transcribe   — audio file → text + language
    POST /chat         — text + lang → agent response (text + WAV)
    GET  /health       — health check

Local development:
    uvicorn src.api:app --reload --port 8000

Deployment:
    Hugging Face Spaces (recommended for portfolio):
        Upload the repo; HF will detect FastAPI and serve it.
    Render / Railway:
        Add a start command: uvicorn src.api:app --host 0.0.0.0 --port $PORT
"""

import io
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.tts.router import TTSRouter
from src.asr.whisper_asr import WhisperASR
from src.utils.lang_detect import detect_language

app = FastAPI(title="VoiceAgent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-loaded singletons
_tts: TTSRouter | None = None
_asr: WhisperASR | None = None


def get_tts() -> TTSRouter:
    global _tts
    if _tts is None:
        _tts = TTSRouter()
    return _tts


def get_asr() -> WhisperASR:
    global _asr
    if _asr is None:
        _asr = WhisperASR(model_size="base")
    return _asr


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    lang: str = Form(default="en"),
):
    """Convert text to speech in the given language. Returns a WAV file."""
    wav_bytes = get_tts().synthesize(text, lang=lang)
    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=output.wav"},
    )


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Transcribe an uploaded audio file. Returns text and detected language."""
    import tempfile, os
    suffix = os.path.splitext(audio.filename)[-1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        result = get_asr().transcribe(tmp_path)
    finally:
        os.unlink(tmp_path)
    return JSONResponse(result)


@app.post("/chat")
async def chat(
    text: str = Form(...),
    lang: str = Form(default=""),
):
    """
    Run the agent on a text input. Returns JSON with agent text + base64 WAV audio.
    The frontend plays the audio directly from the base64 string.
    """
    import base64
    detected_lang = lang if lang else detect_language(text)

    # Minimal single-turn response (full agent requires LLM API key)
    tts = get_tts()
    wav_bytes = tts.synthesize(text, lang=detected_lang)
    audio_b64 = base64.b64encode(wav_bytes).decode()

    return JSONResponse({
        "text": text,
        "language": detected_lang,
        "audio_base64": audio_b64,
    })


# Serve the frontend static files (HTML demo)
# app.mount("/", StaticFiles(directory="frontend/dist", html=True), name="frontend")
