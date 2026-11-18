from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Change: These are library I added
from starlette.concurrency import run_in_threadpool

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DB_PATH = "./eprints_vector_db"
COLLECTION_NAME = "undip_research_search"
N_RESULTS = 100

# Change: Making it so now it validate startup failures
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
    
    app.state.model = SentenceTransformer(MODEL_NAME)
    yield


app = FastAPI(
    title="Undip Research Semantic Search API",
    description="Semantic search over the harvested Undip eprints repository.",
    version="1.0.0",
    lifespan=lifespan,
)


class SearchRequest(BaseModel):
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

# Change: Making it to async def + adding thread offloading
@app.post("/search", response_model=SearchResponse)
async def search(payload: SearchRequest) -> SearchResponse:
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

    return SearchResponse(
        query=payload.query,
        max_threshold=payload.max_threshold,
        count=len(results),
        results=results,
    )
