"""
RAG çekirdeği: retrieval (getirme) + generation (üretme).

Akış:
  1) Soruyu embedding'e çevir.
  2) Vektör DB'den en alakalı chunk'ları getir (retrieval).
  3) Bu chunk'ları numaralandırıp Claude'a "bağlam" olarak ver.
  4) Claude, SADECE bu bağlama dayanarak ve [n] ile kaynak göstererek
     cevap üretir (generation).

Kritik fikir: LLM'e "kafandan uydurma, sana verdiğim parçalara dayan"
talimatı vermek. Bu, halüsinasyonu azaltır ve cevabı denetlenebilir kılar.
"""

from app.config import settings
from app import embeddings, vectorstore, llm

SYSTEM_PROMPT = """Sen, kullanıcının yüklediği belgelere dayanarak soru cevaplayan bir asistansın.

Kurallar:
- SADECE sana verilen "Bağlam" bölümündeki bilgilere dayanarak cevap ver.
- Cevabını desteklemek için kullandığın her bilgiden sonra kaynağını [1], [2] gibi köşeli parantezli numaralarla belirt.
- Eğer cevap bağlamda yoksa, kibarca "Bu bilgi yüklenen belgelerde bulunmuyor." de. Tahmin yürütme, uydurma.
- Kullanıcının dilinde (genellikle Türkçe) ve net, öz cevaplar ver."""


def _build_context(hits: list[dict]) -> str:
    """Getirilen chunk'ları, atıf için numaralandırılmış bir bağlam metnine çevir."""
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[{i}] (Kaynak: {hit['source']})\n{hit['text']}")
    return "\n\n".join(blocks)


def answer_question(question: str, top_k: int | None = None) -> dict:
    top_k = top_k or settings.top_k

    # 1 + 2) Soruyu vektöre çevir ve en alakalı parçaları getir
    query_vec = embeddings.embed_query(question)
    hits = vectorstore.query(query_vec, top_k)

    if not hits:
        return {
            "answer": "Henüz hiç belge yüklenmemiş. Önce /ingest ile bir dosya yükleyin.",
            "sources": [],
        }

    # 3) Bağlamı hazırla ve seçili LLM'e gönder (Gemini/Claude — llm.py'de)
    context = _build_context(hits)
    user_message = f"Bağlam:\n{context}\n\nSoru: {question}"

    # 4) Cevabı üret
    answer_text = llm.generate(SYSTEM_PROMPT, user_message)

    # Hangi kaynakların kullanıldığını kullanıcıya da döndür (şeffaflık)
    sources = [
        {
            "index": i,
            "source": hit["source"],
            "chunk_id": hit["chunk_id"],
            "score": hit["score"],
            "preview": hit["text"][:160] + ("..." if len(hit["text"]) > 160 else ""),
        }
        for i, hit in enumerate(hits, start=1)
    ]

    return {"answer": answer_text, "sources": sources}
