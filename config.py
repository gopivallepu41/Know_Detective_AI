import os
from typing import Any

try:
    import streamlit as st
except Exception:
    st = None


def get_secret(name: str, default: Any = None) -> Any:
    """Read Streamlit secrets first, then environment variables."""
    if st is not None:
        try:
            value = st.secrets.get(name)
            if value:
                return value
        except Exception:
            pass
    return os.getenv(name, default)


GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY") or get_secret("GEMINI_API_KEY")
GOOGLE_MODEL = get_secret("GOOGLE_MODEL") or "gemini-3.6-flash"
EMBEDDING_MODEL = get_secret("EMBEDDING_MODEL", "gemini-embedding-2")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "data", "knowledge")
VECTOR_DIR = os.path.join(BASE_DIR, "vectorstore")
