from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.debug.auto_fix import AutoFixEngine
from app.debug.diagnoser import Diagnoser
from app.debug.suggester import Suggester
from app.models.schemas import QueryRequest
from app.rag.loader import load_and_chunk


app = FastAPI(title=settings.app_name)
CORS_ORIGINS = settings.frontend_origins or ["*"]
ALLOW_CREDENTIALS = bool(settings.frontend_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_FILE = Path("frontend/index.html")
diagnoser = Diagnoser()
suggester = Suggester()
auto_fix_engine = AutoFixEngine()
embedding_model: Any | None = None
debug_evaluator: Any | None = None
vector_store: Any | None = None
hybrid_retriever: Any | None = None
llm_service: Any | None = None
query_cache: dict[tuple[str, int], dict[str, Any]] = {}


def _get_embedding_model():
    global embedding_model

    if embedding_model is None:
        from app.rag.embeddings import EmbeddingModel

        embedding_model = EmbeddingModel()

    return embedding_model


def _get_debug_evaluator():
    global debug_evaluator

    if debug_evaluator is None:
        from app.debug.evaluator import DebugEvaluator

        debug_evaluator = DebugEvaluator(_get_embedding_model())

    return debug_evaluator


def _build_vector_store(file_path: str = "data/docs.txt"):
    global hybrid_retriever
    global vector_store

    from app.rag.retrieval import BM25Retriever, HybridRetriever
    from app.rag.vector_store import VectorStore

    chunks = load_and_chunk(file_path)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = _get_embedding_model().encode(texts)

    if not embeddings:
        raise ValueError("No embeddings were generated from the provided documents.")

    vector_store = VectorStore(dim=len(embeddings[0]))
    vector_store.add(embeddings, chunks)
    bm25_retriever = BM25Retriever(chunks)
    hybrid_retriever = HybridRetriever(vector_store, bm25_retriever, embedding_model)
    return vector_store


def _get_llm_service():
    global llm_service

    if llm_service is None:
        from app.services.llm import LLMService

        llm_service = LLMService()

    return llm_service


def run_pipeline(query: str, top_k: int = 3) -> dict[str, Any]:
    print("RUN PIPELINE:", query, "top_k:", top_k)
    cache_key = (query, top_k)
    if cache_key in query_cache:
        return deepcopy(query_cache[cache_key])

    retrieval_results = hybrid_retriever.search(query, top_k=top_k)
    top_results = retrieval_results[:top_k]
    context_chunks = [result["chunk"]["text"] for result in top_results if result["chunk"]["text"].strip()]

    if not context_chunks:
        debug = _get_debug_evaluator().evaluate(query, [], "I don't know")
        result = {
            "query": query,
            "answer": "I don't know",
            "sources": [],
            "debug": debug,
            "diagnosis": diagnoser.diagnose(debug),
            "suggestions": suggester.suggest(debug),
            "top_k": len(top_results),
            "fix_applied": [],
        }
        query_cache[cache_key] = deepcopy(result)
        return result

    try:
        service = _get_llm_service()
        answer = service.generate(query, context_chunks)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    debug = _get_debug_evaluator().evaluate(query, top_results, answer)

    result = {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "source": result["chunk"]["metadata"]["source"],
                "chunk_index": result["chunk"]["metadata"]["chunk_index"],
                "score": result["score"],
                "retrieval_source": result["source"],
            }
            for result in top_results
        ],
        "debug": debug,
        "diagnosis": diagnoser.diagnose(debug),
        "suggestions": suggester.suggest(debug),
        "top_k": len(top_results),
        "fix_applied": [],
    }
    query_cache[cache_key] = deepcopy(result)
    return result


def _run_auto_fix_pipeline(query: str) -> dict[str, Any]:
    before_top_k = 3
    before = run_pipeline(query, top_k=before_top_k)
    before_failure_type = before["debug"]["failure_type"]
    before_confidence = float(before["debug"]["confidence"])

    # skip auto-fix for successful or already-strong queries to avoid over-correction
    if before_failure_type == "success" or before_confidence >= 0.7 or before_confidence >= 0.5:
        before["fix_applied"] = ["No fix needed (query was already valid)"]
        return {
            "before": before,
            "after": before,
        }

    fix = auto_fix_engine.apply_fix(query, before_failure_type, original_top_k=before_top_k)
    fixed_query = fix["query"]
    print("Auto-fix applied:", fixed_query, fix["top_k"])
    after = run_pipeline(fixed_query, top_k=fix["top_k"])
    after["fix_applied"] = fix.get("fix_applied", [])
    return {
        "before": before,
        "after": after,
    }


@app.api_route("/", methods=["GET", "HEAD"])
async def read_root() -> dict[str, str]:
    return {"message": "RAG Debugger backend is running."}


@app.api_route("/health", methods=["GET", "HEAD"])
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ui")
async def serve_ui() -> FileResponse:
    return FileResponse(FRONTEND_FILE)


@app.get("/load")
async def load_documents() -> dict[str, int]:
    chunks = load_and_chunk("data/docs.txt")
    return {"chunks": len(chunks)}


@app.get("/embed")
async def embed_documents() -> dict[str, int]:
    store = _build_vector_store("data/docs.txt")
    return {
        "vectors_stored": store.index.ntotal,
        "embedding_dimension": store.dim,
    }


@app.get("/search")
async def search_documents(q: str, top_k: int = 5) -> dict[str, Any]:
    global hybrid_retriever
    global vector_store

    if vector_store is None or hybrid_retriever is None:
        try:
            vector_store = _build_vector_store("data/docs.txt")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    results = hybrid_retriever.search(q, top_k=top_k)

    return {
        "query": q,
        "results": [
            {
                "text": result["chunk"]["text"],
                "score": result["score"],
                "source": result["source"],
            }
            for result in results
        ],
    }


@app.post("/query")
async def query_documents(payload: QueryRequest) -> dict[str, Any]:
    global hybrid_retriever
    global vector_store

    if not payload.query:
        return JSONResponse(status_code=400, content={"error": "Query cannot be empty"})

    if len(payload.query) > 500:
        return JSONResponse(status_code=400, content={"error": "Query too long"})

    if vector_store is None or hybrid_retriever is None:
        try:
            vector_store = _build_vector_store("data/docs.txt")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not payload.auto_fix:
        return run_pipeline(payload.query, top_k=3)

    return _run_auto_fix_pipeline(payload.query)
