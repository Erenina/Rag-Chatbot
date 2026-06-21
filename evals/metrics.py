"""
Deterministik metrikler — LLM gerektirmez, dolayısıyla hızlı, ücretsiz ve
%100 tekrarlanabilir (aynı girdi -> hep aynı sonuç).

Buradaki tek metrik 'context_recall': retrieval (getirme) adımının kalitesini
ölçer. RAG'de cevap kötüyse iki suçlu olabilir:
  1) retrieval yanlış parçayı getirdi (cevabın hammaddesi hiç gelmedi), ya da
  2) LLM, doğru parça geldiği halde kötü cevap üretti.
Bu metrik (1)'i izole eder: doğru bilgiyi içeren chunk, getirilenler arasında
var mıydı? Varsa retrieval işini yapmıştır; sorun generation'dadır.
"""


def _normalize(text: str) -> str:
    """Karşılaştırma için metni sadeleştir (küçük harf + kırpma)."""
    return text.lower().strip()


def context_recall(retrieved_texts: list[str], expected_keywords: list[str]) -> float | None:
    """
    Beklenen anahtar bilgiler, getirilen chunk'ların içinde geçiyor mu?

    Cevabı içermesi gereken kısa "işaret" kelimeleri (ör. "20 gün") veri
    setinde elle işaretlenmiştir. Bunların kaçının getirilen metinde
    bulunduğunu oranlarız.

    Dönüş:
      0.0 - 1.0 arası oran (1.0 = tüm beklenen bilgiler getirildi)
      None      = bu soru için anahtar kelime yok (ör. "cevap yok" testi) -> N/A
    """
    if not expected_keywords:
        return None

    haystack = _normalize(" ".join(retrieved_texts))
    found = sum(1 for kw in expected_keywords if _normalize(kw) in haystack)
    return round(found / len(expected_keywords), 4)
