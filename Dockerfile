# RAG Chatbot — HuggingFace Spaces (Docker SDK) için imaj
# Yerelde de çalışır:  docker build -t rag-chatbot . && docker run -p 7860:7860 -e GROQ_API_KEY=... rag-chatbot
FROM python:3.11-slim

# HF Spaces'in önerdiği pratik: root olmayan kullanıcı (UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Önce sadece bağımlılıklar (Docker katman önbelleğinden faydalan)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Embedding modelini imaja göm: ilk istek hızlı olsun, çalışma anında indirme olmasın
ENV HF_HOME=/app/.cache
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/multilingual-e5-small')"

# Uygulama kodu
COPY --chown=user . /app

# Çalışma zamanında yazılabilir klasörler (vektör DB + yüklenen dosyalar)
RUN mkdir -p /app/chroma_db /app/data

# Demo açılışta örnek belgelerle dolu gelsin
ENV PRELOAD_EXAMPLES=true

# HuggingFace Spaces 7860 portunu bekler
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
