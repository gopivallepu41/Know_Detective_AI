import os
import shutil
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import EMBEDDING_MODEL, GOOGLE_API_KEY, KNOWLEDGE_DIR, VECTOR_DIR


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

    embeddings = GoogleGenerativeAIEmbeddings(
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
