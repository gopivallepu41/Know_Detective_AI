import re
import time
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from config import GOOGLE_API_KEY, GOOGLE_MODEL
from schemas import InvestigationState


# ============================================================
# RETRY SETTINGS
# ============================================================

MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 5


# ============================================================
# PROMPTS
# ============================================================

PLANNER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are an autonomous enterprise knowledge investigator.

Your job is NOT to answer the user directly.

Your job is to create focused retrieval queries that can uncover:

1. contradictions
2. stale information
3. unverified claims
4. missing explanations
5. missing relationships between documents

Return exactly 4 focused retrieval queries.

Return one query per line.
Do not use numbering or bullets.
""",
    ),
    (
        "human",
        "Investigation Mission:\n{mission}",
    ),
])


ANALYZER_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are the evidence analyst for KnowDetective AI.

Analyze ONLY the supplied evidence.

Do not invent facts.

Identify concrete knowledge problems such as:

- contradictions
- stale information
- unverified claims
- missing explanations
- missing relationships

For every important finding:

1. Explain the finding.
2. Mention the supporting evidence.
3. Cite the source filename.

If the evidence is insufficient, clearly say so.
""",
    ),
    (
        "human",
        """
Investigation Mission:
{mission}

Evidence:
{evidence}
""",
    ),
])


VERIFY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are the verification agent in an Agentic RAG workflow.

Determine whether the current evidence is sufficient to support the investigation.

Return exactly one of:

ENOUGH

or

NEED_MORE
QUERY: <query>
QUERY: <query>

Use NEED_MORE only when another targeted retrieval could materially
improve confidence in the findings.
""",
    ),
    (
        "human",
        """
Investigation Mission:
{mission}

Current Findings:
{findings}

Evidence:
{evidence}
""",
    ),
])


FINAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are the final reporting agent for KnowDetective AI.

Create a concise but useful knowledge-health report.

Use ONLY the supplied evidence.

Never manufacture findings.

Use exactly these sections:

## Knowledge Health Report

### Findings

### Evidence

### Unresolved Questions

### Suggested Actions

For each finding, include the relevant source filename(s).
""",
    ),
    (
        "human",
        """
Investigation Mission:
{mission}

Findings:
{findings}

Evidence:
{evidence}
""",
    ),
])


# ============================================================
# LLM
# ============================================================

def _llm():
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY is not configured.")

    return ChatGoogleGenerativeAI(
        model=GOOGLE_MODEL,
        google_api_key=GOOGLE_API_KEY,
        temperature=0,
    )


# ============================================================
# HELPERS
# ============================================================

def _content_to_text(content: Any) -> str:
    """
    Convert Gemini/LangChain response content into plain text.
    """

    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:

            if isinstance(item, str):
                parts.append(item)

            elif isinstance(item, dict):

                if "text" in item:
                    parts.append(str(item["text"]))

                elif "content" in item:
                    parts.append(str(item["content"]))

            else:
                parts.append(str(item))

        return "\n".join(parts)

    return str(content)


def _is_rate_limit_error(exc: Exception) -> bool:
    """
    Best-effort check for a Google API rate-limit / quota error,
    without depending on a specific exception class.
    """

    message = str(exc)

    return (
        "429" in message
        or "RESOURCE_EXHAUSTED" in message
        or "quota" in message.lower()
    )


def _extract_retry_delay(exc: Exception, fallback: float) -> float:
    """
    Try to pull the server-suggested retry delay (e.g. "retryDelay": "22s")
    out of the error message; fall back to our own backoff value otherwise.
    """

    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+(?:\.\d+)?)s", str(exc))

    if match:
        try:
            return float(match.group(1)) + 1  # small safety buffer
        except ValueError:
            pass

    return fallback


def _invoke_prompt(prompt, **kwargs) -> str:

    chain = prompt | _llm()

    backoff = INITIAL_BACKOFF_SECONDS

    last_error: Exception = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = chain.invoke(kwargs)
            return _content_to_text(response.content)

        except Exception as exc:

            last_error = exc

            if not _is_rate_limit_error(exc) or attempt == MAX_RETRIES:
                raise

            wait_time = _extract_retry_delay(exc, backoff)

            print(
                f"[KnowDetective] Rate limit hit (attempt {attempt}/"
                f"{MAX_RETRIES}). Retrying in {wait_time:.0f}s..."
            )

            time.sleep(wait_time)

            backoff *= 2  # exponential backoff for the next attempt

    # Should not be reached, but keep a safe fallback.
    raise last_error


def _get_state_value(
    state: InvestigationState,
    key: str,
    default=None,
):

    if isinstance(state, dict):
        return state.get(key, default)

    return getattr(state, key, default)


# ============================================================
# PLANNER NODE
# ============================================================

def planner_node(state: InvestigationState):

    mission = _get_state_value(
        state,
        "mission",
        "",
    )

    response = _invoke_prompt(
        PLANNER_PROMPT,
        mission=mission,
    )

    queries = []

    for line in response.splitlines():

        query = line.strip()

        query = re.sub(
            r"^\s*[-*•]\s*",
            "",
            query,
        )

        query = re.sub(
            r"^\s*\d+[\.\)]\s*",
            "",
            query,
        )

        if query:
            queries.append(query)

    return {
        "queries": queries[:4]
    }


# ============================================================
# RETRIEVER NODE
# ============================================================

def retriever_node(state: InvestigationState, retriever):

    queries = _get_state_value(
        state,
        "queries",
        [],
    )

    if not queries:
        return {
            "evidence": []
        }

    all_documents = []

    for query in queries:

        try:

            documents = retriever.invoke(query)

        except Exception as exc:

            print(
                f"Retriever error for query '{query}': {exc}"
            )

            continue

        for document in documents:

            source = document.metadata.get(
                "source",
                "unknown",
            )

            evidence_item = (
                f"Source: {source}\n"
                f"Content:\n{document.page_content}"
            )

            all_documents.append(
                evidence_item
            )

    # Remove duplicate evidence
    unique_evidence = []

    seen = set()

    for item in all_documents:

        if item not in seen:

            seen.add(item)

            unique_evidence.append(item)

    return {
        "evidence": unique_evidence
    }


# ============================================================
# ANALYZER NODE
# ============================================================

def analyzer_node(state: InvestigationState):

    mission = _get_state_value(
        state,
        "mission",
        "",
    )

    evidence = _get_state_value(
        state,
        "evidence",
        [],
    )

    if isinstance(evidence, list):

        evidence_text = "\n\n".join(
            str(item)
            for item in evidence
        )

    else:

        evidence_text = str(evidence)

    findings = _invoke_prompt(
        ANALYZER_PROMPT,
        mission=mission,
        evidence=evidence_text,
    )

    return {
        "findings": findings
    }


# ============================================================
# VERIFIER NODE
# ============================================================

def verifier_node(state: InvestigationState):

    mission = _get_state_value(
        state,
        "mission",
        "",
    )

    findings = _get_state_value(
        state,
        "findings",
        "",
    )

    evidence = _get_state_value(
        state,
        "evidence",
        [],
    )

    if isinstance(evidence, list):

        evidence_text = "\n\n".join(
            str(item)
            for item in evidence
        )

    else:

        evidence_text = str(evidence)

    verification = _invoke_prompt(
        VERIFY_PROMPT,
        mission=mission,
        findings=findings,
        evidence=evidence_text,
    )

    verification_upper = verification.upper()

    if "NEED_MORE" in verification_upper:

        status = "NEED_MORE"

    else:

        status = "ENOUGH"

    additional_queries = []

    for line in verification.splitlines():

        if line.strip().upper().startswith("QUERY:"):

            query = line.split(
                ":",
                1,
            )[1].strip()

            if query:
                additional_queries.append(query)

    return {
        "verification": verification,
        "verification_status": status,
        "additional_queries": additional_queries[:2],
    }


# ============================================================
# PREPARE ADDITIONAL RETRIEVAL
# ============================================================

def additional_retrieval_planner_node(
    state: InvestigationState,
):

    additional_queries = _get_state_value(
        state,
        "additional_queries",
        [],
    )

    return {
        "queries": additional_queries
    }


# ============================================================
# FINAL REPORT NODE
# ============================================================

def final_node(state: InvestigationState):

    mission = _get_state_value(
        state,
        "mission",
        "",
    )

    findings = _get_state_value(
        state,
        "findings",
        "",
    )

    evidence = _get_state_value(
        state,
        "evidence",
        [],
    )

    if isinstance(evidence, list):

        evidence_text = "\n\n".join(
            str(item)
            for item in evidence
        )

    else:

        evidence_text = str(evidence)

    report = _invoke_prompt(
        FINAL_PROMPT,
        mission=mission,
        findings=findings,
        evidence=evidence_text,
    )

    return {
        "final_report": report
    }


# ============================================================
# ROUTING
# ============================================================

def route_after_verification(
    state: InvestigationState,
):

    status = _get_state_value(
        state,
        "verification_status",
        "ENOUGH",
    )

    if status == "NEED_MORE":

        return "additional_retrieval"

    return "final"


# ============================================================
# BUILD GRAPH
# ============================================================

def build_graph(retriever):

    workflow = StateGraph(
        InvestigationState
    )

    workflow.add_node(
        "planner",
        planner_node,
    )

    workflow.add_node(
        "retriever",
        lambda state: retriever_node(
            state,
            retriever,
        ),
    )

    workflow.add_node(
        "analyzer",
        analyzer_node,
    )

    workflow.add_node(
        "verifier",
        verifier_node,
    )

    workflow.add_node(
        "additional_retrieval_planner",
        additional_retrieval_planner_node,
    )

    workflow.add_node(
        "additional_retrieval",
        lambda state: retriever_node(
            state,
            retriever,
        ),
    )

    workflow.add_node(
        "final",
        final_node,
    )

    # Initial workflow
    workflow.add_edge(
        START,
        "planner",
    )

    workflow.add_edge(
        "planner",
        "retriever",
    )

    workflow.add_edge(
        "retriever",
        "analyzer",
    )

    workflow.add_edge(
        "analyzer",
        "verifier",
    )

    # Verification routing
    workflow.add_conditional_edges(
        "verifier",
        route_after_verification,
        {
            "additional_retrieval":
                "additional_retrieval_planner",

            "final":
                "final",
        },
    )

    # Additional retrieval loop
    workflow.add_edge(
        "additional_retrieval_planner",
        "additional_retrieval",
    )

    workflow.add_edge(
        "additional_retrieval",
        "analyzer",
    )

    # Finish
    workflow.add_edge(
        "final",
        END,
    )

    return workflow.compile()
