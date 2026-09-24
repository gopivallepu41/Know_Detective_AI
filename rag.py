import os
import re
import shutil
import time
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import EMBEDDING_MODEL, GOOGLE_API_KEY, KNOWLEDGE_DIR, VECTOR_DIR


# ============================================================
# RETRY SETTINGS (for embedding rate limits)
# ============================================================

MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 5


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        "429" in message
        or "RESOURCE_EXHAUSTED" in message
        or "quota" in message.lower()
    )


def _extract_retry_delay(exc: Exception, fallback: float) -> float:
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", str(exc))
    if match:
        try:
            return float(match.group(1)) + 1
        except ValueError:
            pass
    return fallback


def _with_retry(func, *args, **kwargs):
    """Call func(*args, **kwargs), retrying with backoff on rate-limit errors."""

    backoff = INITIAL_BACKOFF_SECONDS
    last_error: Exception = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            if not _is_rate_limit_error(exc) or attempt == MAX_RETRIES:
                raise
            wait_time = _extract_retry_delay(exc, backoff)
            print(
                f"[KnowDetective] Embedding rate limit hit (attempt "
                f"{attempt}/{MAX_RETRIES}). Retrying in {wait_time:.0f}s..."
            )
            time.sleep(wait_time)
            backoff *= 2

    raise last_error


class RetryingEmbeddings(GoogleGenerativeAIEmbeddings):
    """
    Same as GoogleGenerativeAIEmbeddings, but automatically retries
    with backoff when the API returns a rate-limit / quota error.
    """

    def embed_documents(self, texts, *args, **kwargs):
        return _with_retry(
            super().embed_documents, texts, *args, **kwargs
        )

    def embed_query(self, text, *args, **kwargs):
        return _with_retry(
            super().embed_query, text, *args, **kwargs
        )


def load_source_documents() -> List[Document]:
    docs: List[Document] = []
    if not os.path.isdir(KNOWLEDGE_DIR):
        return docs

    for filename in sorted(os.listdir(KNOWLEDGE_DIR)):
        path = os.path.join(KNOWLEDGE_DIR, filename)
        if not os.path.isfile(path):
            continue
        if not filename.lower().endswith((".md", ".txt")):
            continue
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={"source": filename, "path": path},
                )
            )
    return docs


def build_vectorstore(rebuild: bool = False) -> Chroma:
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not configured.")

    embeddings = RetryingEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    if rebuild and os.path.isdir(VECTOR_DIR):
        shutil.rmtree(VECTOR_DIR, ignore_errors=True)

    source_docs = load_source_documents()
    if not source_docs:
        raise RuntimeError(
            "No knowledge files found. Add .md or .txt files under data/knowledge/."
        )

    # Rebuild when the local store doesn't exist. For a demo deployment, the store
    # is generated from repository knowledge files at runtime.
    if not os.path.isdir(VECTOR_DIR) or not os.listdir(VECTOR_DIR):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        chunks = splitter.split_documents(source_docs)
        return Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=VECTOR_DIR,
            collection_name="knowdetective",
        )

    return Chroma(
        persist_directory=VECTOR_DIR,
        embedding_function=embeddings,
        collection_name="knowdetective",
    )


def get_retriever(rebuild: bool = False):
    vectorstore = build_vectorstore(rebuild=rebuild)
    return vectorstore.as_retriever(search_kwargs={"k": 6})
