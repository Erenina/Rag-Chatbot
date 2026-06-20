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
    llm_provider: str = "gemini"                # "gemini" | "claude"
    max_answer_tokens: int = 1024               # cevap için üst sınır

    # Gemini (ücretsiz kota) — https://aistudio.google.com/apikey
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

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
