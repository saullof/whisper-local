from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import tempfile
import shutil
import os

# Variáveis de ambiente (podem ser configuradas na aba Ambiente do EasyPanel, se quiser)
MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # "cpu" ou "cuda"
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")  # int8, int8_float32, float16...

# Carrega o modelo na inicialização
model = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)

app = FastAPI(title="Whisper Local API")


@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str | None = None):
    # Salva o arquivo de áudio temporariamente
    suffix = os.path.splitext(file.filename)[1] or ".mp3"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        segments, info = model.transcribe(tmp_path, language=language)
        text = "".join(seg.text for seg in segments)

        return JSONResponse(
            {
                "text": text.strip(),
                "language": info.language,
                "duration": info.duration,
            }
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
