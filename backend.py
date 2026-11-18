import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DB_PATH = "./eprints_vector_db"
COLLECTION_NAME = "undip_research_search"
N_RESULTS = 100

# [BARU] Jumlah jurnal teratas yang dikirim ke LLM (angka awal, bisa disetel tim).
SUMMARY_TOP_K = 5

# [BARU] Isi "arahan praktis" saat tidak ada jurnal yang lolos threshold.
GUIDANCE = [
    "Naikkan batas jarak (max_threshold) agar lebih longgar.",
    "Coba kata kunci lain.",
    "Perluas topik pencarian.",
]

# [BARU] Logging dasar untuk observability. Di bawah uvicorn, logger aplikasi
# tidak otomatis aktif, jadi dikonfigurasi di sini.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
log = logging.getLogger("aisen.search")


# Validasi startup (dari backend.py versi tim, dipertahankan)
@asynccontextmanager
async def lifespan(app: FastAPI):
    client = chromadb.PersistentClient(path=DB_PATH)

    try:
        app.state.collection = client.get_collection(name=COLLECTION_NAME)
    except chromadb.errors.NotFoundError as exc:
        raise RuntimeError(
            f"ChromaDB collection '{COLLECTION_NAME}' was not found "
            f"in '{DB_PATH}'. Check the database path and make sure "
            "you've run the data-ingestion step."
        ) from exc

    # [BARU] Import dipindah ke sini agar test yang tidak butuh model
    # tidak perlu memuat torch.
    from sentence_transformers import SentenceTransformer

    app.state.model = SentenceTransformer(MODEL_NAME)
    app.state.summarize = summarize_with_llm  # [BARU] mudah diganti saat test
    yield


app = FastAPI(
    title="Undip Research Semantic Search API",
    description="Semantic search over the harvested Undip eprints repository.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)

class SearchRequest(BaseModel):
    # [BARU] Spasi di awal/akhir dibuang, jadi query "   " ditolak (min_length=1).
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., min_length=1, description="Free-text search query (Indonesian or English).")
    max_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Maximum cosine distance allowed for a result (lower = stricter).",
    )


class SearchResult(BaseModel):
    peringkat: int
    skor: float
    judul: str
    penulis: str
    deskripsi: str
    url: str
    date: str


class SearchResponse(BaseModel):
    query: str
    max_threshold: float
    count: int
    results: list[SearchResult]
    # [BARU] Field tambahan (frontend lama tetap kompatibel):
    #   full       -> ringkasan AI + daftar jurnal
    #   list_only  -> LLM gagal, daftar jurnal saja
    #   no_results -> tidak ada jurnal lolos threshold, lihat `arahan`
    status: Literal["full", "list_only", "no_results"]
    ringkasan: str | None = None  # sitasi berformat [n], n = `peringkat`
    arahan: list[str] = []


# [BARU] ---- LLM -------------------------------------------------------------
class LLMError(Exception):
    pass


def build_prompt(query: str, results: list[SearchResult]) -> str:
    sources = "\n\n".join(f"[{r.peringkat}] {r.judul}\n{r.deskripsi}" for r in results)
    return (
        "Jawab pertanyaan berdasarkan HANYA sumber di bawah ini. "
        "Setiap klaim wajib diberi sitasi berformat [n] sesuai nomor sumber. "
        "Jika sumber tidak cukup, katakan demikian.\n\n"
        f"Pertanyaan: {query}\n\nSumber:\n{sources}"
    )


def summarize_with_llm(query: str, results: list[SearchResult]) -> str:
    """TODO: sambungkan ke LLM yang dipilih tim (pakai build_prompt).
    Fungsi ini HARUS raise exception bila gagal atau timeout; API lalu
    otomatis memakai jalur fallback 'list_only'."""
    raise LLMError("LLM belum dikonfigurasi")


@app.get("/health")
def health() -> dict:
    # [BARU] Untuk pengecekan hidup/mati dan jumlah dokumen ter-index.
    return {"status": "ok", "documents": app.state.collection.count()}


@app.post("/search", response_model=SearchResponse)
async def search(payload: SearchRequest) -> SearchResponse:
    started = time.perf_counter()
    model = app.state.model
    collection = app.state.collection

    query_vector = await run_in_threadpool(model.encode, payload.query)
    query_vector = query_vector.tolist()

    raw = await run_in_threadpool(
        collection.query,
        query_embeddings=[query_vector],
        n_results=N_RESULTS,
        include=["metadatas", "distances"],
    )

    results: list[SearchResult] = []
    rank = 0
    for metadata, distance in zip(raw["metadatas"][0], raw["distances"][0]):
        if distance > payload.max_threshold:
            continue
        rank += 1
        results.append(
            SearchResult(
                peringkat=rank,
                skor=distance,
                judul=metadata["title"],
                penulis=metadata["creators"],
                deskripsi=metadata["description"],
                url=metadata["url"],
                date=metadata["date"],
            )
        )

    # [BARU] Cabang flowchart setelah threshold
    ringkasan = None
    arahan: list[str] = []
    if not results:
        status = "no_results"
        arahan = GUIDANCE
    else:
        try:
            ringkasan = await run_in_threadpool(
                app.state.summarize, payload.query, results[:SUMMARY_TOP_K]
            )
            status = "full"
        except Exception:
            log.exception("LLM gagal, fallback ke daftar jurnal saja")
            ringkasan = None
            status = "list_only"

    log.info(
        "search status=%s count=%d max_threshold=%.2f duration_ms=%.0f",
        status,
        len(results),
        payload.max_threshold,
        (time.perf_counter() - started) * 1000,
    )

    return SearchResponse(
        query=payload.query,
        max_threshold=payload.max_threshold,
        count=len(results),
        results=results,
        status=status,
        ringkasan=ringkasan,
        arahan=arahan,
    )