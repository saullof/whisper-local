from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import tempfile
import shutil
import os

# Config pelas variáveis de ambiente (EasyPanel)
MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # "cpu" ou "cuda"
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# Carrega o modelo na inicialização do container
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
        # Transcrição com VAD para forçar a criação de segmentos
        segments_iter, info = model.transcribe(
            tmp_path,
            language=language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300),
        )

        # ⚠️ IMPORTANTE: transformar em lista para poder usar mais de uma vez
        segments = list(segments_iter)

        # Texto completo
        full_text = "".join(seg.text for seg in segments)

        # Monta os dados de cada segmento com start/end/text
        segment_data = []
        for seg in segments:
            segment_data.append({
                "start": float(seg.start),
                "end":   float(seg.end),
                "text":  seg.text,
            })

        # duration: usa info.duration ou o fim do último segmento
        if getattr(info, "duration", None):
            duration = float(info.duration)
        elif segment_data:
            duration = float(segment_data[-1]["end"])
        else:
            duration = 0.0

        return JSONResponse(
            {
                "text":      full_text.strip(),
                "language":  info.language,
                "duration":  duration,
                "segments":  segment_data,   # 👈 agora vem preenchido
            }
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
