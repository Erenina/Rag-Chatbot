"""
FastAPI uygulaması — tüm parçaları birleştiren HTTP katmanı.

Endpoint'ler:
  GET    /health            -> servis ayakta mı + kaç chunk var
  POST   /ingest            -> dosya yükle (PDF/TXT/MD), chunk'la, embed et, sakla
  POST   /chat              -> soru sor, kaynaklı cevap al
  GET    /documents         -> yüklü dosyaları listele
  DELETE /documents         -> tüm dosyaları sil

Sunucu çalışırken otomatik dökümantasyon:  http://localhost:8000/docs
"""

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app import ingestion, embeddings, vectorstore, rag
from app.llm import LLMConfigError, LLMAPIError
from app.models import (
    ChatRequest, ChatResponse, IngestResponse,
    DocumentsResponse, DocumentInfo,
)

app = FastAPI(
    title="RAG Chatbot API",
    description="Belgelerini yükle, üzerine soru sor, kaynak göstererek cevap al.",
    version="1.0.0",
)

# Yüklenen dosyalar için klasörü hazırla
os.makedirs(settings.data_dir, exist_ok=True)

# Frontend (statik tek sayfa). CWD'den bağımsız olsun diye mutlak yol.
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    """Web arayüzünü (sohbet ekranı) servis et."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "total_chunks": vectorstore.count()}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    filename = file.filename or "yuklenen_dosya"

    # 1) Dosyayı diske kaydet
    saved_path = os.path.join(settings.data_dir, filename)
    contents = await file.read()
    with open(saved_path, "wb") as f:
        f.write(contents)

    # 2) Metni çıkar
    try:
        text = ingestion.extract_text(saved_path, filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not text.strip():
        raise HTTPException(status_code=400, detail="Dosyadan metin çıkarılamadı (boş ya da taranmış görsel PDF olabilir).")

    # 3) Chunk'la (her parçaya belge başlığını ekleyerek — contextual chunking)
    chunks = ingestion.chunk_text_with_context(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Metin chunk'lara bölünemedi.")

    # 4) Embedding üret
    vectors = embeddings.embed_passages(chunks)

    # 5) Vektör DB'ye yaz (aynı dosya tekrar yüklenirse önce eskisini temizle)
    vectorstore.delete_by_source(filename)
    ids = [f"{filename}::{i}::{uuid.uuid4().hex[:8]}" for i in range(len(chunks))]
    metadatas = [{"source": filename, "chunk_index": i} for i in range(len(chunks))]
    vectorstore.add_chunks(ids=ids, documents=chunks, embeddings=vectors, metadatas=metadatas)

    return IngestResponse(
        filename=filename,
        chunks_added=len(chunks),
        message=f"'{filename}' başarıyla işlendi ve {len(chunks)} parçaya bölünüp kaydedildi.",
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        result = rag.answer_question(request.question, top_k=request.top_k)
    except LLMConfigError as e:
        # Eksik/yanlış yapılandırma (ör. API key yok) — net mesaj ver
        raise HTTPException(status_code=401, detail=str(e))
    except LLMAPIError as e:
        # LLM çağrısı başarısız (ağ, kota, sunucu vb.)
        raise HTTPException(status_code=502, detail=f"LLM çağrısı başarısız: {e}")
    return ChatResponse(**result)


@app.get("/documents", response_model=DocumentsResponse)
def documents():
    counts = vectorstore.list_documents()
    docs = [DocumentInfo(source=src, chunks=n) for src, n in counts.items()]
    return DocumentsResponse(documents=docs, total_chunks=vectorstore.count())


@app.delete("/documents")
def clear_documents():
    vectorstore.clear()
    return {"message": "Tüm belgeler ve vektörler silindi."}
