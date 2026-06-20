"""
Lokal embedding modeli sarmalayıcısı.

Embedding = metni, anlamını temsil eden bir sayı vektörüne (ör. 384 boyutlu)
çevirmek. Benzer anlamlı metinler, vektör uzayında birbirine yakın olur.
RAG'in "en alakalı parçayı bul" adımı bu yakınlığa dayanır.

Burada 'intfloat/multilingual-e5-small' kullanıyoruz: küçük, hızlı, hem
Türkçe hem İngilizce destekli. Bu model bir önemli detay ister:
  - aranan SORU başına  "query: "  öneki,
  - depolanan METİN (passage) başına  "passage: "  öneki
konmalı. Bu önekler modelin eğitildiği biçim; atlanırsa kalite düşer.
"""

from functools import lru_cache

from app.config import settings


@lru_cache(maxsize=1)
def _get_model():
    """
    Modeli tembel (lazy) ve tek sefer yükle.

    sentence_transformers / torch import'u ağırdır (ilk seferinde 10+ sn).
    Bu import'u modül başına değil, fonksiyon içine koyuyoruz ki sunucu
    anında açılsın; ağır yükleme yalnızca ilk embedding çağrısında olsun.
    lru_cache sayesinde model bir kez yüklenir, sonra bellekten kullanılır.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(settings.embedding_model)


def embed_passages(texts: list[str]) -> list[list[float]]:
    """Depolanacak doküman parçalarını vektöre çevir."""
    model = _get_model()
    prefixed = [f"passage: {t}" for t in texts]
    vectors = model.encode(prefixed, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Kullanıcının sorusunu vektöre çevir."""
    model = _get_model()
    vector = model.encode(f"query: {text}", normalize_embeddings=True)
    return vector.tolist()
