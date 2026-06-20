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

from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    """
    Modeli tembel (lazy) ve tek sefer yükle. lru_cache sayesinde ilk
    çağrıda diskten/internetten yüklenir, sonraki çağrılarda bellekteki
    örnek kullanılır — her istekte yeniden yüklemeyiz.
    """
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
