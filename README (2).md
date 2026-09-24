# 🧠 KnowDetective AI

**KnowDetective AI** is an agentic knowledge-investigation tool. Instead of just answering a question from your documents, it actively investigates your knowledge base to uncover:

- ⚠️ **Conflicting information** — different documents giving different answers
- ❓ **Missing information** — gaps where documentation doesn't say enough
- 🕒 **Stale / outdated content** — information that may need review
- 🔍 **Unsupported claims** — statements without evidence behind them

It's built as a multi-step **Agentic RAG** workflow using LangGraph, with a guided Streamlit front end.

---

## How it works

1. **Ask a question** about your knowledge base (e.g. *"Are there conflicting instructions in our deployment documents?"*).
2. **Planner** — an LLM breaks the question into up to 4 focused retrieval queries.
3. **Retriever** — each query is run against a Chroma vector store built from your `.md` / `.txt` knowledge files.
4. **Analyzer** — the LLM reviews only the retrieved evidence and identifies contradictions, gaps, or unverifiable claims.
5. **Verifier** — checks whether the evidence is strong enough, or whether another retrieval pass is needed.
6. **Final report** — a structured "Knowledge Health Report" with Findings, Evidence, Unresolved Questions, and Suggested Actions.

The whole flow is orchestrated as a graph (`graph.py`) so the investigation can loop back for more evidence before finalizing.

---

## Project structure

```
.
├── app.py          # Streamlit UI — 5-step guided flow (welcome → question → results)
├── graph.py        # LangGraph workflow: planner → retriever → analyzer → verifier → final
├── rag.py          # Document loading, chunking, embeddings, Chroma vector store, retriever
├── schemas.py       # InvestigationState definition shared across graph nodes
├── config.py        # Configuration (API keys, model names, directories)
└── data/knowledge/  # Your knowledge base (.md / .txt files) — KNOWLEDGE_DIR
```

---

## Requirements

- Python 3.10+
- A Google Generative AI API key (used for both the chat model and embeddings)

Core dependencies:

```
streamlit
langgraph
langchain-core
langchain-chroma
langchain-google-genai
langchain-text-splitters
```

Install with:

```bash
pip install streamlit langgraph langchain-core langchain-chroma langchain-google-genai langchain-text-splitters
```

---

## Configuration

Set the following in your environment (or in `config.py`):

| Variable          | Description                                    |
|-------------------|-------------------------------------------------|
| `GOOGLE_API_KEY`  | API key for Google Generative AI (Gemini)       |
| `GOOGLE_MODEL`    | Chat model name used for planning/analysis/report |
| `EMBEDDING_MODEL` | Embedding model used to build the vector store  |
| `KNOWLEDGE_DIR`   | Folder containing your `.md` / `.txt` knowledge files |
| `VECTOR_DIR`      | Folder where the Chroma vector store is persisted |

Example:

```bash
export GOOGLE_API_KEY="your-key-here"
```

---

## Adding knowledge

Drop `.md` or `.txt` files into your `KNOWLEDGE_DIR`. Each file becomes a source document, chunked and embedded into the vector store. The vector store is built automatically the first time it's needed; delete `VECTOR_DIR` (or call `get_retriever(rebuild=True)`) to force a rebuild after adding or changing files.

---

## Running the app

```bash
streamlit run app.py
```

The app walks the user through:

1. **Welcome** — intro to KnowDetective AI
2. **What it does** — the four types of knowledge problems it looks for
3. **Ask a question** — free text or example prompts
4. **Investigating** — live progress: documents loaded, passages matched, evidence gathered
5. **Results** — the full Knowledge Health Report plus the evidence used

---

## Notes

- The investigation currently retrieves the top `k=6` most relevant chunks per query, across up to 4 planner queries (plus up to 2 extra queries if the verifier decides more evidence is needed).
- Reports are generated strictly from retrieved evidence — the analyzer and final-report prompts are instructed not to invent findings.
- All user-supplied text (question, evidence) is HTML-escaped before rendering, since the UI renders custom HTML blocks in Streamlit.
