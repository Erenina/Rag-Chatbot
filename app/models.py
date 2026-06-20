"""
API istek/cevap şemaları (Pydantic).

FastAPI bu sınıfları kullanarak gelen JSON'u doğrular ve otomatik
dökümantasyon (/docs) üretir. Yani hem validasyon hem de API kontratı
burada tanımlı.
"""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Kullanıcının sorusu")
    top_k: int | None = Field(
        default=None, ge=1, le=20,
        description="Kaç chunk getirileceği (boşsa ayarlardaki varsayılan kullanılır)",
    )


class Source(BaseModel):
    """Cevabın dayandığı bir kaynak parçası."""
    index: int                  # cevapta [1], [2] olarak atıfta bulunulan numara
    source: str                 # dosya adı
    chunk_id: str               # parçanın benzersiz kimliği
    score: float                # alaka skoru (mesafeden türetilmiş, yüksek = daha alakalı)
    preview: str                # parçanın ilk birkaç satırı


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    message: str


class DocumentInfo(BaseModel):
    source: str
    chunks: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int
