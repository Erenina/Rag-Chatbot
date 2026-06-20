"""
Vektör veritabanı sarmalayıcısı (ChromaDB).

ChromaDB; vektörleri, metinleri ve metadata'yı (kaynak dosya adı gibi)
birlikte saklar ve "şu vektöre en yakın N kayıt" sorgusunu hızlıca yapar.
PersistentClient kullanıyoruz, yani veriler diske yazılır ve sunucu
yeniden başladığında kaybolmaz.

Neden ChromaDB? Lokal, kurulumu kolay, metadata + benzerlik aramasını
birlikte verir. Üretimde Qdrant/pgvector/Pinecone'a geçilebilir — arayüz
hemen hemen aynı kalır.
"""

import chromadb

from app.config import settings

# Aynı koleksiyonu tüm uygulama boyunca paylaşıyoruz
_client = chromadb.PersistentClient(path=settings.chroma_dir)
_collection = _client.get_or_create_collection(
    name="documents",
    # Vektörlerimiz normalize edilmiş; kosinüs benzerliği uygun ölçüt
    metadata={"hnsw:space": "cosine"},
)


def add_chunks(
    ids: list[str],
    documents: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
) -> None:
    """Parçaları vektörleriyle birlikte koleksiyona ekle."""
    _collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def delete_by_source(source: str) -> None:
    """Belirli bir dosyaya ait tüm chunk'ları sil (yeniden yüklemeden önce kullanılır)."""
    _collection.delete(where={"source": source})


def query(embedding: list[float], top_k: int) -> list[dict]:
    """
    Verilen soru vektörüne en yakın top_k parçayı döndür.
    Her sonuç: {text, source, chunk_id, score}.
    """
    results = _collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
    )

    # Chroma sonuçları liste-içinde-liste döndürür (her sorgu için bir liste)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    ids = results["ids"][0]

    hits = []
    for text, meta, distance, chunk_id in zip(documents, metadatas, distances, ids):
        hits.append({
            "text": text,
            "source": meta.get("source", "bilinmiyor"),
            "chunk_id": chunk_id,
            # Kosinüs mesafesi 0 (aynı) .. 2 (zıt). Okunabilir bir alaka
            # skoruna çeviriyoruz: 1'e yakın = daha alakalı.
            "score": round(1 - distance, 4),
        })
    return hits


def list_documents() -> dict:
    """Yüklenmiş dosyaları ve her birinin chunk sayısını döndür."""
    data = _collection.get(include=["metadatas"])
    counts: dict[str, int] = {}
    for meta in data["metadatas"]:
        source = meta.get("source", "bilinmiyor")
        counts[source] = counts.get(source, 0) + 1
    return counts


def count() -> int:
    """Koleksiyondaki toplam chunk sayısı."""
    return _collection.count()


def clear() -> None:
    """Tüm koleksiyonu temizle (tüm dosyaları sil)."""
    global _collection
    _client.delete_collection("documents")
    _collection = _client.get_or_create_collection(
        name="documents",
        metadata={"hnsw:space": "cosine"},
    )
