"""
Belge yükleme (ingestion) yardımcıları: metin çıkarma + chunk'lama.

Neden chunk'lıyoruz?
  1) Embedding modellerinin girdi sınırı var; koca bir PDF tek vektöre sığmaz.
  2) Daha küçük parçalar = daha isabetli arama. Soruya yalnızca ilgili
     paragrafı getiririz, tüm dökümanı değil.
  3) LLM'e yalnızca alakalı parçaları context olarak veririz (maliyet + kalite).

Örtüşme (overlap) neden var?
  Bir cümle parça sınırında ikiye bölünürse anlam kopabilir. Parçaların
  uçlarını biraz örtüştürerek bu bağlam kaybını azaltıyoruz.
"""

from pypdf import PdfReader

from app.config import settings


def extract_text(file_path: str, filename: str) -> str:
    """Dosya türüne göre ham metni çıkar (PDF / TXT / MD)."""
    lower = filename.lower()
    if lower.endswith(".pdf"):
        reader = PdfReader(file_path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages)
    if lower.endswith((".txt", ".md")):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    raise ValueError(f"Desteklenmeyen dosya türü: {filename} (PDF, TXT, MD destekleniyor)")


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """
    Metni, kelime sınırlarına saygı duyarak ~chunk_size karakterlik
    örtüşmeli parçalara böl.

    Basit ama sağlam bir yaklaşım: metni kelimelere ayır, kelimeleri
    chunk_size'ı aşmayacak şekilde paketle, sonraki parçaya 'overlap'
    kadar geri dönerek başla.
    """
    chunk_size = chunk_size or settings.chunk_size
    overlap = overlap or settings.chunk_overlap

    # Fazla boşlukları sadeleştir
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for word in words:
        # +1: kelimeler arası boşluk
        if current_len + len(word) + 1 > chunk_size and current:
            chunks.append(" ".join(current))
            # Örtüşme: son birkaç kelimeyi koruyarak yeni parçaya başla
            overlap_words = _tail_words(current, overlap)
            current = overlap_words
            current_len = sum(len(w) + 1 for w in current)
        current.append(word)
        current_len += len(word) + 1

    if current:
        chunks.append(" ".join(current))

    return chunks


def _tail_words(words: list[str], overlap_chars: int) -> list[str]:
    """Listenin sonundan, toplam ~overlap_chars karaktere kadar kelime al."""
    tail: list[str] = []
    total = 0
    for word in reversed(words):
        if total + len(word) + 1 > overlap_chars:
            break
        tail.insert(0, word)
        total += len(word) + 1
    return tail


def _document_title(text: str) -> str:
    """Belgenin ilk anlamlı satırını 'başlık' olarak al (genelde belge/şirket adı)."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return ""


def chunk_text_with_context(
    text: str,
    chunk_size: int | None = None,
    overlap: int | None = None,
) -> list[str]:
    """
    chunk_text gibi böler, AMA her parçanın başına belgenin BAŞLIĞINI ekler.

    Neden? Chunk'lama, bir parçanın hangi belgeye/şirkete ait olduğunu
    kaybedebilir: ör. bir belgenin 2. parçasında artık şirket adı geçmeyebilir.
    Birden çok benzer belge varken (Acme vs Beta) model parçaları karıştırır ve
    yanlış belgenin bilgisini verir. Başlığı her parçaya iliştirmek bu
    "attribution" (atıf) kaybını önler — basit bir 'contextual chunking'.
    Hem retrieval'ı (başlık embedding'e girer) hem de generation'ı (model hangi
    belge olduğunu görür) iyileştirir.
    """
    title = _document_title(text)
    chunks = chunk_text(text, chunk_size, overlap)
    if not title:
        return chunks
    # İlk parça zaten başlıkla başlar; tekrarı önleyip diğerlerine başlığı ekle.
    return [c if c.startswith(title) else f"{title}\n\n{c}" for c in chunks]
