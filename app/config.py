"""
Uygulama ayarları.

Tüm yapılandırmayı tek bir yerde topluyoruz. pydantic-settings, .env
dosyasındaki değerleri otomatik okur ve tip kontrolü yapar. Böylece
kod içine "magic number" gömmek yerine her şey buradan yönetilir —
bu, mülakatlarda da iyi görünen bir alışkanlıktır.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # .env dosyasını oku, tanımadığı ekstra değişkenleri görmezden gel
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- LLM (provider değiştirilebilir) ---
    llm_provider: str = "groq"                  # "groq" | "gemini" | "claude"
    max_answer_tokens: int = 1024               # cevap için üst sınır

    # Gemini (ücretsiz kota) — https://aistudio.google.com/apikey
    gemini_api_key: str | None = None
    # Not: gemini-2.5-flash ücretsiz katmanda günde sadece ~20 istek veriyor;
    # flash-lite'ın günlük limiti çok daha yüksek, bu yüzden varsayılan o.
    gemini_model: str = "gemini-2.5-flash-lite"

    # Groq (ücretsiz, yüksek limit) — https://console.groq.com
    groq_api_key: str | None = None
    groq_model: str = "llama-3.1-8b-instant"    # üretici: hızlı + yüksek günlük limit

    # --- Eval yargıcı (judge) ---
    # Yargıç, üreticiden FARKLI sağlayıcı/model kullanabilir: hem yükü ikinci bir
    # ücretsiz kotaya dağıtır hem de self-preference bias'ı (kendi cevabını
    # kayırma) azaltır. judge_model=None ise seçilen sağlayıcının kendi
    # varsayılan modeli kullanılır.
    judge_provider: str = "groq"               # "groq" | "gemini" | "claude"
    judge_model: str | None = "llama-3.3-70b-versatile"  # yargıç: daha güçlü model

    # Claude (paralı) — kullanmak istersen LLM_PROVIDER=claude yap
    anthropic_api_key: str | None = None
    chat_model: str = "claude-opus-4-8"

    # --- Embedding ---
    embedding_model: str = "intfloat/multilingual-e5-small"

    # --- Chunking ---
    chunk_size: int = 800                       # her parçanın yaklaşık karakter sayısı
    chunk_overlap: int = 120                    # parçalar arası örtüşme (bağlam kaybını azaltır)

    # --- Retrieval ---
    top_k: int = 4                              # her soruda kaç chunk getirilsin

    # --- Depolama yolları ---
    chroma_dir: str = "chroma_db"               # vektör DB'nin diske yazılacağı klasör
    data_dir: str = "data"                      # yüklenen dosyaların saklanacağı klasör


# Tek bir global ayar nesnesi — her modül bunu import eder
settings = Settings()
