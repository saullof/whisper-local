FROM python:3.11-slim

WORKDIR /app

# Dependências de sistema (ffmpeg ajuda a decodificar vários formatos de áudio)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Instala as libs Python
RUN pip install --no-cache-dir \
    faster-whisper \
    fastapi \
    "uvicorn[standard]" \
    python-multipart

# Copia o server.py pro container
COPY server.py /app/server.py

# Comando padrão ao subir o container
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
