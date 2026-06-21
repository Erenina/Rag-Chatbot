"""
LLM-as-a-judge: bir LLM'i, başka bir LLM'in ürettiği cevabı PUANLAMAK için
kullanmak.

Neden gerekli?
  Faithfulness (sadakat) ve correctness (doğruluk) "anlam" gerektiren
  metriklerdir; basit string eşleşmesiyle ölçülemez. Örneğin "20 gün izin"
  ile "yılda yirmi gün tatil hakkı" aynı şeydir ama karakterleri farklıdır.
  Bir yargıç-LLM, insan değerlendirmesine yakın, otomatik ve ölçeklenebilir
  bir puan verir.

İki tasarım kararı (ikisi de CV'de anlatmaya değer):
  1) Tek çağrı, iki metrik: faithfulness + correctness'i AYRI çağrılar yerine
     tek bir istekte alıyoruz. Ücretsiz kotada istek sayısı değerlidir; bu,
     API kullanımını yarıya indirir.
  2) Yargıç ≠ üretici: yargıç modeli ayarlardan (judge_model) gelir ve
     üreticiden farklı olabilir. Aynı modeli hem üretici hem yargıç yapmak
     "self-preference bias" (kendi cevabını kayırma) riski taşır.
"""

import json

from app import llm
from app.config import settings


_JUDGE_SYSTEM = """Sen titiz bir RAG değerlendiricisin. Sana bir SORU, bir yapay zekanın ürettiği CEVAP, cevabın dayanması gereken BAĞLAM ve insan tarafından doğru kabul edilen REFERANS CEVAP verilecek.

İki ayrı metriği 0.0 ile 1.0 arasında puanla:

1) faithfulness (sadakat): CEVAP yalnızca BAĞLAMDAKİ bilgilere mi dayanıyor?
   - Her iddia bağlamla destekleniyorsa 1.0'a yakın.
   - Bağlamda olmayan/uydurulmuş/çelişen bilgi varsa düşür.
   - "Bu bilgi belgede yok" gibi bir ret cevabı hiçbir iddia içermez; sadık say (yüksek puan).

2) correctness (doğruluk): CEVAP, REFERANS CEVAP ile anlamca uyuşuyor mu?
   - Kelimesi kelimesine aynı olması gerekmez; kilit bilgi doğruysa yüksek puan.
   - Kilit bilgi yanlış/eksikse düşür.
   - Referans "bilgi yok" diyor ve cevap da bilginin olmadığını söylüyorsa, bu DOĞRUDUR (yüksek puan).

Yanıtını SADECE şu JSON ile ver, başka hiçbir şey yazma:
{"faithfulness": {"score": <0.0-1.0>, "reason": "<tek cümle>"}, "correctness": {"score": <0.0-1.0>, "reason": "<tek cümle>"}}"""


def _coerce(metric: dict) -> dict:
    """Tek bir metriğin {score, reason}'unu güvenli biçimde normalize et."""
    try:
        score = max(0.0, min(1.0, float(metric["score"])))
        return {"score": round(score, 4), "reason": str(metric.get("reason", "")).strip()}
    except (KeyError, TypeError, ValueError):
        return {"score": None, "reason": "puan okunamadı"}


def _parse(raw: str) -> dict:
    """
    Yargıcın metin yanıtından iki metriği savunmacı biçimde çıkar.

    LLM bazen JSON'u ```json ... ``` çitlerine alır veya başına/sonuna metin
    ekler. Bu yüzden ilk '{' ile son '}' arasını alıp parse etmeyi deneriz;
    başarısız olursa çökmek yerine score=None döneriz.
    """
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            return {
                "faithfulness": _coerce(data.get("faithfulness", {})),
                "correctness": _coerce(data.get("correctness", {})),
            }
        except json.JSONDecodeError:
            pass
    failed = {"score": None, "reason": f"JSON ayrıştırılamadı: {raw[:100]}"}
    return {"faithfulness": dict(failed), "correctness": dict(failed)}


def evaluate(question: str, answer: str, context: str, ground_truth: str) -> dict:
    """
    Cevabı tek bir yargıç çağrısıyla iki açıdan puanla.

    Dönüş:
      {"faithfulness": {"score", "reason"}, "correctness": {"score", "reason"}}
    """
    user_message = (
        f"SORU:\n{question}\n\n"
        f"BAĞLAM:\n{context}\n\n"
        f"CEVAP:\n{answer}\n\n"
        f"REFERANS CEVAP (doğru kabul edilen):\n{ground_truth}"
    )
    raw = llm.generate(
        _JUDGE_SYSTEM,
        user_message,
        model=settings.judge_model,
        provider=settings.judge_provider,
    )
    return _parse(raw)
