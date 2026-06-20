"""
LLM (cevap üretimi) katmanı — provider-agnostik.

RAG'in "generation" adımını buraya soyutladık. Hangi modeli kullandığımız
(Gemini, Claude, ...) yalnızca burada bilinir; rag.py sadece generate()
çağırır. Bu, sağlayıcıyı değiştirmeyi tek dosyaya indirir ve CV için iyi
bir tasarım örneğidir.

Provider, ayarlardan (LLM_PROVIDER) seçilir.
"""

from app.config import settings


# Üst katmanın (FastAPI) yakalayıp anlamlı HTTP hatasına çevireceği hatalar
class LLMConfigError(Exception):
    """Eksik/yanlış yapılandırma (ör. API key yok)."""


class LLMAPIError(Exception):
    """LLM çağrısı sırasında oluşan hata (ağ, kota, sunucu vb.)."""


def generate(system_prompt: str, user_message: str) -> str:
    """Seçili sağlayıcıyla cevap üret ve düz metin döndür."""
    provider = settings.llm_provider.lower()
    if provider == "gemini":
        return _generate_gemini(system_prompt, user_message)
    if provider == "claude":
        return _generate_claude(system_prompt, user_message)
    raise LLMConfigError(f"Bilinmeyen LLM_PROVIDER: {settings.llm_provider} (gemini | claude)")


def _generate_gemini(system_prompt: str, user_message: str) -> str:
    if not settings.gemini_api_key:
        raise LLMConfigError(
            "GEMINI_API_KEY tanımlı değil. https://aistudio.google.com/apikey "
            "adresinden ücretsiz key al ve .env'e ekle."
        )
    # SDK import'u tembel: yalnızca Gemini kullanılırken yüklensin
    from google import genai
    from google.genai import types
    from google.genai import errors as genai_errors

    client = genai.Client(api_key=settings.gemini_api_key)
    try:
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=settings.max_answer_tokens,
            ),
        )
    except genai_errors.APIError as e:
        raise LLMAPIError(str(e))
    return response.text or ""


def _generate_claude(system_prompt: str, user_message: str) -> str:
    if not settings.anthropic_api_key:
        raise LLMConfigError("ANTHROPIC_API_KEY tanımlı değil (.env'i kontrol et).")
    import anthropic

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    try:
        response = client.messages.create(
            model=settings.chat_model,
            max_tokens=settings.max_answer_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
    except anthropic.APIError as e:
        raise LLMAPIError(str(e))
    return "".join(b.text for b in response.content if b.type == "text")
