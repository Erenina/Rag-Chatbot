"""
RAG değerlendirme (eval) koşucusu.

Ne yapar:
  1) Örnek belgenin yüklü olduğundan emin olur (eval kendi kendine yeter).
  2) evals/dataset.json içindeki her soruyu gerçek RAG hattından geçirir.
  3) Üç metrik hesaplar:
       - context_recall : retrieval doğru parçayı getirdi mi?  (deterministik)
       - faithfulness   : cevap bağlama sadık mı, uydurma var mı?  (LLM yargıç)
       - correctness    : cevap referans cevapla uyuşuyor mu?      (LLM yargıç)
  4) Eşiklerle pass/fail verir, özet tablo basar, evals/report.json'a yazar.

Dayanıklılık:
  - Hız sınırlayıcı (rate limiter) ile ücretsiz kotanın dakikalık limitini aşmaz.
  - Her sorudan SONRA diske yazar; kota/ağ yüzünden yarıda kesilse bile ilerleme
    kaybolmaz. Tekrar çalıştırınca tamamlananları atlayıp kaldığı yerden devam eder.

Çalıştırma (proje kökünden):
    python -m evals.run_eval            # kaldığı yerden devam (önbelleği kullanır)
    python -m evals.run_eval --fresh    # önbelleği yok say, baştan çalıştır
"""

import json
import sys
import time
from pathlib import Path

from app import embeddings, ingestion, vectorstore
from app.config import settings
from app.llm import LLMAPIError, LLMConfigError
from app.rag import answer_question
from evals import judge, metrics

ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = Path(__file__).parent / "dataset.json"
REPORT_PATH = Path(__file__).parent / "report.json"
EXAMPLES_DIR = ROOT / "examples"

# Pass/fail eşikleri — gerçek bir CI kalite kapısı gibi düşün
THRESHOLDS = {"faithfulness": 0.7, "correctness": 0.7, "context_recall": 0.6}

# Ücretsiz katman dakikalık istek (RPM) ile sınırlı. Her LLM çağrısı arasında
# en az bu kadar bekleyerek limiti aşmıyoruz (güvenli pay ile 13s).
MIN_SECONDS_BETWEEN_CALLS = 13.0
_last_call_ts = 0.0

# Geçici hatalarda (kota/hız ya da 503 "model meşgul") ilk denemeden sonra
# en fazla bu kadar kez daha denenir (artan bekleme/backoff ile).
RETRY_ATTEMPTS = 3


def _ensure_examples_ingested() -> None:
    """
    Eval'in tek başına ve tekrarlanabilir çalışması için examples/ klasöründeki
    tüm örnek belgelerin vektör DB'de bulunduğundan emin ol. Zaten yüklü olanlara
    dokunma. (Çeldirici belge buradan otomatik yüklenir.)
    """
    existing = vectorstore.list_documents()
    for path in sorted(EXAMPLES_DIR.glob("*")):
        if path.suffix.lower() not in (".txt", ".md", ".pdf"):
            continue
        if path.name in existing:
            print(f"✓ '{path.name}' zaten yüklü.")
            continue
        print(f"↑ '{path.name}' yükleniyor...")
        text = ingestion.extract_text(str(path), path.name)
        chunks = ingestion.chunk_text_with_context(text)
        vectors = embeddings.embed_passages(chunks)
        ids = [f"{path.name}::{i}" for i in range(len(chunks))]
        metadatas = [{"source": path.name, "chunk_index": i} for i in range(len(chunks))]
        vectorstore.add_chunks(ids=ids, documents=chunks, embeddings=vectors, metadatas=metadatas)
        print(f"  {len(chunks)} parça eklendi.")


# --- Hata sınıflandırma + hız sınırlama ---

def _is_rate_limit(message: str) -> bool:
    """Geçici (yeniden denenebilir) kota/hız hatası mı?"""
    m = message.lower()
    return any(s in m for s in ("429", "resource_exhausted", "quota", "rate limit"))


def _is_auth_error(message: str) -> bool:
    """Kalıcı kimlik doğrulama hatası mı (key eksik/geçersiz)?"""
    m = message.lower()
    return any(s in m for s in ("401", "unauthenticated", "api key", "api_key", "invalid authentication"))


def _is_transient(message: str) -> bool:
    """Geçici sunucu hatası mı (model meşgul/erişilemez)? -> yeniden denenebilir."""
    m = message.lower()
    return any(s in m for s in ("503", "500", "502", "unavailable", "overloaded", "high demand", "deadline", "timeout"))


def _throttle() -> None:
    """Ardışık iki LLM çağrısı arasında en az MIN_SECONDS_BETWEEN_CALLS bırak."""
    global _last_call_ts
    wait = MIN_SECONDS_BETWEEN_CALLS - (time.time() - _last_call_ts)
    if wait > 0:
        time.sleep(wait)
    _last_call_ts = time.time()


def _with_retry(fn, *args):
    """
    Hız limitine uyarak LLM çağrısını çalıştır ve GEÇİCİ hatalarda yeniden dene:
      - kota/hız limiti (429)     -> sabit uzun bekleme
      - sunucu meşgul (503/5xx)   -> artan bekleme (backoff)
    Kalıcı hatalarda (auth/401) boşuna beklemeden hemen yükselt; main() temiz
    bir mesajla yakalar.
    """
    for attempt in range(RETRY_ATTEMPTS + 1):
        _throttle()
        try:
            return fn(*args)
        except LLMAPIError as e:
            msg = str(e)
            if _is_auth_error(msg):
                raise  # kalıcı: tekrar denemenin anlamı yok
            retryable = _is_rate_limit(msg) or _is_transient(msg)
            if not retryable or attempt == RETRY_ATTEMPTS:
                raise  # düzeltilemez ya da deneme hakkı bitti
            if _is_rate_limit(msg):
                wait, reason = 25, "kota/hız limiti"
            else:
                wait, reason = 8 * (attempt + 1), "model meşgul (503)"
            print(f"   ⏳ {reason}; {wait}s bekleyip tekrar ({attempt + 1}/{RETRY_ATTEMPTS})...", flush=True)
            time.sleep(wait)


# --- Tek bir soruyu değerlendirme ---

def evaluate_item(item: dict) -> dict:
    """Tek bir soruyu hattan geçir ve üç metriği hesapla (2 LLM çağrısı)."""
    question = item["question"]

    # --- Retrieval'ı tek başına koştur: getirilen TAM metni görmek için ---
    # (answer_question kendi içinde de retrieval yapar; embedding deterministik
    #  olduğundan sonuç aynıdır. Burada amaç retrieval'ı izole ölçmek.)
    query_vec = embeddings.embed_query(question)
    hits = vectorstore.query(query_vec, settings.top_k)
    retrieved_texts = [h["text"] for h in hits]
    context = "\n\n".join(retrieved_texts)

    # --- Uçtan uca GERÇEK hat: kullanıcının web'de aldığı cevabın aynısı ---
    result = _with_retry(answer_question, question)
    answer = result["answer"]

    # --- Metrikler: 1 deterministik + 1 yargıç çağrısı (iki skor birden) ---
    recall = metrics.context_recall(retrieved_texts, item.get("context_keywords", []))
    scores = _with_retry(judge.evaluate, question, answer, context, item["ground_truth"])
    faith, corr = scores["faithfulness"], scores["correctness"]

    return {
        "id": item["id"],
        "question": question,
        "answer": answer,
        "context_recall": recall,
        "faithfulness": faith["score"],
        "faithfulness_reason": faith["reason"],
        "correctness": corr["score"],
        "correctness_reason": corr["reason"],
    }


# --- Toplama, biçimlendirme, kayıt ---

def _avg(values) -> float | None:
    nums = [v for v in values if isinstance(v, (int, float))]
    return round(sum(nums) / len(nums), 4) if nums else None


def _fmt(v) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else "  —"


def _passed(r: dict) -> bool:
    """Geçer sayılmak için faithfulness VE correctness eşik üstü olmalı."""
    return (
        (r["faithfulness"] or 0) >= THRESHOLDS["faithfulness"]
        and (r["correctness"] or 0) >= THRESHOLDS["correctness"]
    )


def _aggregate(results: list[dict]) -> dict:
    return {
        "context_recall": _avg(r["context_recall"] for r in results),
        "faithfulness": _avg(r["faithfulness"] for r in results),
        "correctness": _avg(r["correctness"] for r in results),
        "pass_rate": round(sum(_passed(r) for r in results) / len(results), 4) if results else None,
    }


def _save_report(results: list[dict]) -> None:
    """
    Sonuçları (kısmi olsa bile) diske yaz — yarıda kesilse de kaybolmasın.
    Boşsa yazma: ilk soruda çöken bir koşu, önceki iyi raporu EZMESİN.
    """
    if not results:
        return
    report = {"thresholds": THRESHOLDS, "aggregate": _aggregate(results), "results": results}
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_cache() -> dict:
    """Önceki koşunun report.json'unu id -> sonuç sözlüğü olarak yükle (yoksa boş)."""
    if not REPORT_PATH.exists():
        return {}
    try:
        data = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        return {r["id"]: r for r in data.get("results", [])}
    except (json.JSONDecodeError, KeyError):
        return {}


def _is_complete(r: dict) -> bool:
    """Bu sonuç başarıyla tamamlanmış mı (yargıç skorları dolu mu)?"""
    return r.get("faithfulness") is not None and r.get("correctness") is not None


# --- Temiz çıkış mesajları ---

def _fail_auth(detail: str) -> None:
    print("\n" + "!" * 74)
    print("❌ LLM kimlik doğrulaması başarısız — API key eksik veya geçersiz.")
    print("   .env'deki ilgili key'i kontrol et:")
    print("     • Gemini:  https://aistudio.google.com/apikey")
    print("     • Groq:    https://console.groq.com   (gsk_... ile başlar)")
    print(f"\n   (teknik detay: {detail[:180]})")
    print("!" * 74 + "\n")
    sys.exit(1)


def _fail_quota(detail: str, done: int, total: int) -> None:
    print("\n" + "!" * 74)
    print(f"⚠️  Ücretsiz GÜNLÜK kota doldu. {done}/{total} soru tamamlandı ve kaydedildi.")
    print("   İlerleme evals/report.json'da — hiçbir şey kaybolmadı.")
    print("\n   Seçenekler:")
    print("   • Kota sıfırlanınca tekrar çalıştır; kalan sorulardan devam eder:")
    print("       python -m evals.run_eval")
    print("   • .env'de daha yüksek limitli bir modele geç (GEMINI_MODEL / JUDGE_MODEL).")
    print(f"\n   (teknik detay: {detail[:140]})")
    print("!" * 74 + "\n")
    sys.exit(2)


def _fail_transient(detail: str, done: int, total: int) -> None:
    print("\n" + "!" * 74)
    print(f"⚠️  Model şu an yoğun/erişilemez (geçici sunucu hatası, 503). "
          f"{done}/{total} tamamlandı ve kaydedildi.")
    print("   Birkaç dakika sonra tekrar çalıştır; kaldığı yerden devam eder:")
    print("       python -m evals.run_eval")
    print(f"\n   (teknik detay: {detail[:140]})")
    print("!" * 74 + "\n")
    sys.exit(3)


def _print_table(results: list[dict]) -> None:
    agg = _aggregate(results)
    print("\n" + "=" * 74)
    print(f"{'soru':<16}{'recall':>9}{'faith':>9}{'correct':>9}   sonuç")
    print("-" * 74)
    for r in results:
        mark = "✅ geçti" if _passed(r) else "❌ kaldı"
        print(
            f"{r['id']:<16}"
            f"{_fmt(r['context_recall']):>9}"
            f"{_fmt(r['faithfulness']):>9}"
            f"{_fmt(r['correctness']):>9}"
            f"   {mark}"
        )
    print("-" * 74)
    print(
        f"{'ORTALAMA':<16}"
        f"{_fmt(agg['context_recall']):>9}"
        f"{_fmt(agg['faithfulness']):>9}"
        f"{_fmt(agg['correctness']):>9}"
        f"   geçme oranı {agg['pass_rate'] * 100:.0f}%"
    )
    print("=" * 74)
    print(f"\n📄 Ayrıntılı rapor (her cevap + gerekçe): {REPORT_PATH}\n")


def main() -> None:
    fresh = "--fresh" in sys.argv
    _ensure_examples_ingested()

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    cache = {} if fresh else _load_cache()

    def _model_for(provider: str) -> str:
        """Bir sağlayıcının o anki varsayılan modelini döndür (etiket için)."""
        return {
            "groq": settings.groq_model,
            "claude": settings.chat_model,
        }.get(provider.lower(), settings.gemini_model)

    gen_label = f"{settings.llm_provider}:{_model_for(settings.llm_provider)}"
    judge_label = f"{settings.judge_provider}:{settings.judge_model or _model_for(settings.judge_provider)}"
    print(
        f"\n🧪 {len(dataset)} soruluk eval seti çalışıyor "
        f"(üretici={gen_label}, yargıç={judge_label})"
    )
    reused = sum(1 for it in dataset if _is_complete(cache.get(it["id"], {})))
    if reused and not fresh:
        print(f"   (önbellek: {reused} tamamlanmış sonuç var, atlanacak)")
    print(f"   (hız limiti ~{MIN_SECONDS_BETWEEN_CALLS:.0f}s/çağrı; birkaç dakika sürebilir)\n")

    results: list[dict] = []
    try:
        for i, item in enumerate(dataset, start=1):
            cached = cache.get(item["id"])
            if cached and _is_complete(cached):
                print(f"[{i}/{len(dataset)}] {item['id']} ... (önbellekten)")
                results.append(cached)
                continue
            print(f"[{i}/{len(dataset)}] {item['id']} ...", flush=True)
            results.append(evaluate_item(item))
            _save_report(results)  # her sorudan sonra kaydet (resilience)
    except LLMConfigError as e:
        _save_report(results)
        _fail_auth(str(e))
    except LLMAPIError as e:
        _save_report(results)
        msg = str(e)
        if _is_auth_error(msg):
            _fail_auth(msg)
        if _is_rate_limit(msg):
            _fail_quota(msg, done=len(results), total=len(dataset))
        if _is_transient(msg):
            _fail_transient(msg, done=len(results), total=len(dataset))
        raise  # gerçekten beklenmedik hata: olduğu gibi göster

    _save_report(results)
    _print_table(results)


if __name__ == "__main__":
    main()
