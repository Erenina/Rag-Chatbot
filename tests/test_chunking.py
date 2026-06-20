"""
chunk_text için birim testleri.

Saf (pure) fonksiyonları test etmek hızlı ve değerlidir: ne model indirir
ne de API çağırır. Chunk'lama RAG'in doğruluğunu doğrudan etkilediği için
buradaki davranışı kilitliyoruz.

Çalıştır:  pytest -q
"""

from app.ingestion import chunk_text


def test_kucuk_metin_tek_parca():
    metin = "kısa bir cümle"
    chunks = chunk_text(metin, chunk_size=800, overlap=120)
    assert chunks == ["kısa bir cümle"]


def test_bos_metin_bos_liste():
    assert chunk_text("   ", chunk_size=800, overlap=120) == []


def test_uzun_metin_birden_fazla_parcaya_bolunur():
    metin = " ".join(["kelime"] * 500)  # ~3000 karakter
    chunks = chunk_text(metin, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    # Her parça (son hariç) sınıra yakın olmalı, taşmamalı
    for c in chunks:
        assert len(c) <= 320  # küçük tolerans


def test_parcalar_ortusur():
    metin = " ".join(f"k{i}" for i in range(200))
    chunks = chunk_text(metin, chunk_size=200, overlap=60)
    # Bir parçanın son kelimeleri, bir sonrakinin başında tekrar etmeli
    first_words = set(chunks[0].split())
    second_words = set(chunks[1].split())
    assert first_words & second_words, "Örtüşme (overlap) bekleniyordu"
