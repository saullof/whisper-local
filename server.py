from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from faster_whisper import WhisperModel
import tempfile
import shutil
import os

MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")  # "cpu" ou "cuda"
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

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
        # 👇 AQUI: pedimos timestamps de palavra
        segments, info = model.transcribe(
            tmp_path,
            language=language,
            word_timestamps=True,
        )

        full_text = "".join(seg.text for seg in segments)

        segment_data = []
        for seg in segments:
            words = []
            # seg.words só existe se word_timestamps=True
            if getattr(seg, "words", None) is not None:
                for w in seg.words:
                    words.append({
                        "word":  w.word,
                        "start": float(w.start),
                        "end":   float(w.end),
                    })

            segment_data.append({
                "start": float(seg.start),
                "end":   float(seg.end),
                "text":  seg.text,
                "words": words,
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
                "text": full_text.strip(),
                "language": info.language,
                "duration": duration,
                "segments": segment_data,   # 👈 AQUI ESTÃO OS TIMECODES
            }
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
