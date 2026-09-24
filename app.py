import os
import html
import textwrap
from datetime import datetime
import streamlit as st

from config import KNOWLEDGE_DIR
from rag import get_retriever
from graph import build_graph


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="KnowDetective AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = 1

if "question" not in st.session_state:
    st.session_state.question = ""

if "result" not in st.session_state:
    st.session_state.result = None


# ============================================================
# HELPERS
# ============================================================

def render_html(content: str):
    """Render an HTML block, stripping leading indentation first.

    Streamlit's markdown renderer treats 4+ spaces of leading
    indentation as a code block, which caused raw tags like <h1>
    to show up as plain text instead of being rendered. Wrapping
    every HTML call in this helper (which runs textwrap.dedent)
    fixes that.
    """
    st.markdown(textwrap.dedent(content), unsafe_allow_html=True)


def get_report(result):
    if not isinstance(result, dict):
        return None

    return (
        result.get("final_report")
        or result.get("report")
        or result.get("answer")
        or result.get("response")
    )


def get_evidence(result):
    if not isinstance(result, dict):
        return []

    evidence = result.get("evidence", [])

    if evidence is None:
        return []

    if isinstance(evidence, list):
        return evidence

    return [evidence]


def evidence_details(item, index):
    if isinstance(item, dict):

        metadata = item.get("metadata", {})

        if not isinstance(metadata, dict):
            metadata = {}

        source = (
            item.get("source")
            or metadata.get("source")
            or item.get("name")
            or f"Evidence {index}"
        )

        content = (
            item.get("content")
            or item.get("page_content")
            or item.get("text")
            or ""
        )

    else:
        source = f"Evidence {index}"
        content = str(item)

    return str(source), str(content)


def get_greeting():
    hour = datetime.now().hour

    if hour < 12:
        return "Good morning 👋"
    elif hour < 18:
        return "Good afternoon 👋"
    else:
        return "Good evening 👋"


def next_page():
    st.session_state.page += 1


def previous_page():
    st.session_state.page -= 1


# ============================================================
# CUSTOM CSS
# ============================================================

render_html(
    """
    <style>

    /* ---------- GENERAL ---------- */

    .stApp {
        background: #f7f8fc;
    }

    .block-container {
        max-width: 1050px;
        padding-top: 35px;
        padding-bottom: 50px;
    }

    /* ---------- HIDE STREAMLIT UI ---------- */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    /* ---------- HERO ---------- */

    .hero {
        text-align: center;
        padding: 55px 20px 30px 20px;
    }

    .welcome-text {
        font-size: 15px;
        color: #667085;
        font-weight: 600;
        letter-spacing: 0.3px;
        margin-bottom: 8px;
    }

    .logo {
        font-size: 58px;
        margin-bottom: 12px;
    }

    .hero h1 {
        font-size: 46px;
        margin: 0;
        font-weight: 750;
        letter-spacing: -1px;
    }

    .hero p {
        font-size: 19px;
        color: #667085;
        max-width: 700px;
        margin: 18px auto;
        line-height: 1.6;
    }

    /* ---------- PAGE LABEL ---------- */

    .page-label {
        text-align: center;
        color: #667085;
        font-size: 14px;
        margin-bottom: 25px;
    }

    /* ---------- CARDS ---------- */

    .card {
        background: white;
        border: 1px solid #e7e9ef;
        border-radius: 18px;
        padding: 26px;
        margin: 12px 0;
        box-shadow: 0 4px 18px rgba(16, 24, 40, 0.04);
    }

    .card h3 {
        margin-top: 0;
        font-size: 21px;
    }

    .card p {
        color: #667085;
        line-height: 1.55;
        margin-bottom: 0;
    }

    /* ---------- FEATURE CARDS ---------- */

    .feature {
        background: white;
        border: 1px solid #e7e9ef;
        border-radius: 18px;
        padding: 25px;
        height: 175px;
        box-shadow: 0 4px 18px rgba(16, 24, 40, 0.04);
    }

    .feature-icon {
        font-size: 30px;
        margin-bottom: 10px;
    }

    .feature-title {
        font-weight: 700;
        font-size: 18px;
        margin-bottom: 8px;
    }

    .feature-text {
        color: #667085;
        font-size: 14px;
        line-height: 1.5;
    }

    /* ---------- RESULT ---------- */

    .result-box {
        background: white;
        border: 1px solid #e7e9ef;
        border-radius: 20px;
        padding: 32px;
        margin-top: 20px;
        box-shadow: 0 6px 25px rgba(16, 24, 40, 0.05);
    }

    .result-title {
        font-size: 25px;
        font-weight: 750;
        margin-bottom: 18px;
    }

    .evidence {
        background: #fafbff;
        border: 1px solid #eaecf0;
        border-radius: 14px;
        padding: 20px;
        margin: 12px 0;
    }

    .source {
        font-weight: 700;
        font-size: 15px;
        margin-bottom: 8px;
    }

    .source-text {
        color: #667085;
        line-height: 1.55;
        font-size: 14px;
    }

    /* ---------- PRIMARY BUTTON (match theme) ---------- */

    div.stButton > button[kind="primary"] {
        background: #5b5ce8;
        border: none;
        border-radius: 12px;
        font-weight: 650;
    }

    div.stButton > button[kind="primary"]:hover {
        background: #4a4bd6;
    }

    /* ---------- STEP STATUS (page 4) ---------- */

    .step-status {
        display: flex;
        align-items: center;
        gap: 10px;
        background: #f5f6fe;
        border: 1px solid #e3e4fb;
        border-radius: 14px;
        padding: 16px 20px;
        margin: 14px 0 0 0;
        font-weight: 600;
        color: #3f3fb0;
    }

    .step-status.done {
        background: #f0fbf5;
        border: 1px solid #d6f2e3;
        color: #17915a;
    }

    .step-detail {
        background: #fafbff;
        border: 1px solid #eaecf0;
        border-radius: 14px;
        padding: 16px 20px;
        margin: 10px 0 0 0;
        color: #475467;
        font-size: 14px;
        line-height: 1.6;
    }

    /* ---------- PROGRESS ---------- */

    .progress-box {
        background: white;
        border-radius: 22px;
        padding: 45px;
        text-align: center;
        border: 1px solid #e7e9ef;
        box-shadow: 0 5px 25px rgba(16, 24, 40, 0.05);
        margin-top: 30px;
    }

    .progress-icon {
        font-size: 55px;
        margin-bottom: 15px;
    }

    .progress-title {
        font-size: 27px;
        font-weight: 750;
    }

    .progress-text {
        color: #667085;
        margin-top: 10px;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #98a2b3;
        font-size: 13px;
        margin-top: 55px;
    }

    </style>
    """
)


# ============================================================
# PAGE 1 — WELCOME
# ============================================================

if st.session_state.page == 1:

    render_html(
        f"""
        <div class="hero">
            <p class="welcome-text">{get_greeting()}</p>
            <div class="logo">🧠</div>
            <h1>KnowDetective AI</h1>
            <p>
                Discover what your organization's knowledge really says.
                Find differences, missing information, outdated content,
                and unsupported claims.
            </p>
        </div>
        """
    )

    render_html(
        """
        <div class="card">
            <h3>🔎 Investigate your knowledge</h3>
            <p>
                Ask a question about your documents and let KnowDetective
                search, compare, and verify the information before giving
                you an answer.
            </p>
        </div>
        """
    )

    st.write("")

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button(
            "Get Started  →",
            use_container_width=True,
            type="primary",
        ):
            next_page()
            st.rerun()


# ============================================================
# PAGE 2 — WHAT IT DOES
# ============================================================

elif st.session_state.page == 2:

    render_html('<div class="page-label">STEP 2 OF 5</div>')

    render_html(
        """
        <div class="hero" style="padding-top:10px;">
            <h1 style="font-size:36px;">
                What would you like to discover?
            </h1>
            <p>
                KnowDetective looks beyond a simple answer.
                It examines the information available in your documents.
            </p>
        </div>
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        render_html(
            """
            <div class="feature">
                <div class="feature-icon">⚠️</div>
                <div class="feature-title">
                    Conflicting information
                </div>
                <div class="feature-text">
                    Find when different documents give different
                    answers about the same topic.
                </div>
            </div>
            """
        )

        render_html(
            """
            <div class="feature">
                <div class="feature-icon">❓</div>
                <div class="feature-title">
                    Missing information
                </div>
                <div class="feature-text">
                    Identify important areas where your documentation
                    does not provide enough information.
                </div>
            </div>
            """
        )

    with col2:
        render_html(
            """
            <div class="feature">
                <div class="feature-icon">🕒</div>
                <div class="feature-title">
                    Information that may need review
                </div>
                <div class="feature-text">
                    Spot content that may be old or inconsistent
                    with other available information.
                </div>
            </div>
            """
        )

        render_html(
            """
            <div class="feature">
                <div class="feature-icon">🔎</div>
                <div class="feature-title">
                    Unsupported information
                </div>
                <div class="feature-text">
                    Check whether important statements can be
                    supported by the available documents.
                </div>
            </div>
            """
        )

    st.write("")
    st.write("")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("← Back", use_container_width=True):
            previous_page()
            st.rerun()

    with col3:
        if st.button(
            "Continue  →",
            use_container_width=True,
            type="primary",
        ):
            next_page()
            st.rerun()


# ============================================================
# PAGE 3 — QUESTION
# ============================================================

elif st.session_state.page == 3:

    render_html('<div class="page-label">STEP 3 OF 5</div>')

    render_html(
        """
        <div class="hero" style="padding-top:10px;">
            <h1 style="font-size:36px;">
                Ask KnowDetective
            </h1>
            <p>
                Ask a question about the information in your knowledge base.
            </p>
        </div>
        """
    )

    render_html('<div class="card"><h3>💬 Your question</h3></div>')

    question = st.text_area(
        "Question",
        value=st.session_state.question,
        height=150,
        placeholder=(
            "Example:\n"
            "What is the correct process for deploying the application?"
        ),
        label_visibility="collapsed",
    )

    st.write("")

    render_html(
        """
        <p style="color:#667085; font-size:14px;">
            Try asking:
        </p>
        """
    )

    examples = [
        "Are there conflicting instructions in our deployment documents?",
        "What information is missing from the troubleshooting guide?",
        "Which document describes the current deployment process?",
    ]

    for example in examples:
        if st.button(
            f"💡 {example}",
            use_container_width=True,
        ):
            st.session_state.question = example
            st.rerun()

    st.write("")
    st.write("")

    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        if st.button("← Back", use_container_width=True):
            previous_page()
            st.rerun()

    with col3:
        if st.button(
            "Investigate  🔎",
            use_container_width=True,
            type="primary",
        ):

            if not question.strip():
                st.warning("Please enter a question first.")

            else:
                st.session_state.question = question.strip()
                next_page()
                st.rerun()


# ============================================================
# PAGE 4 — INVESTIGATION
# ============================================================

elif st.session_state.page == 4:

    render_html('<div class="page-label">STEP 4 OF 5</div>')

    render_html(
        """
        <div class="progress-box">
            <div class="progress-icon">🔎</div>
            <div class="progress-title">
                Investigating your question
            </div>
            <div class="progress-text">
                KnowDetective is looking through your knowledge
                and checking the available information.
            </div>
        </div>
        """
    )

    st.write("")

    progress = st.progress(0)

    status = st.empty()

    detail_box = st.empty()

    try:

        # -------- STEP 1: load knowledge base --------

        status.markdown(
            '<div class="step-status">🧠 Loading your knowledge base...</div>',
            unsafe_allow_html=True,
        )

        doc_count = 0

        if os.path.isdir(KNOWLEDGE_DIR):
            doc_count = sum(
                1
                for f in os.listdir(KNOWLEDGE_DIR)
                if os.path.isfile(os.path.join(KNOWLEDGE_DIR, f))
            )

        detail_box.markdown(
            f'<div class="step-detail">📁 Found <b>{doc_count}</b> '
            f'document(s) in your knowledge base.</div>',
            unsafe_allow_html=True,
        )

        progress.progress(20)

        retriever = get_retriever()

        # -------- STEP 2: retrieve real matching docs --------

        status.markdown(
            '<div class="step-status">📚 Searching for relevant documents...</div>',
            unsafe_allow_html=True,
        )

        retrieved_docs = []

        try:
            retrieved_docs = retriever.invoke(st.session_state.question)
        except AttributeError:
            retrieved_docs = retriever.get_relevant_documents(
                st.session_state.question
            )

        source_names = []

        for doc in retrieved_docs:
            metadata = getattr(doc, "metadata", {}) or {}
            name = metadata.get("source") or metadata.get("name")
            if name and name not in source_names:
                source_names.append(name)

        if source_names:
            listed = "<br>".join(f"&bull; {html.escape(n)}" for n in source_names)
            detail_box.markdown(
                f'<div class="step-detail">🔎 Matched <b>{len(retrieved_docs)}</b> '
                f'passage(s) from <b>{len(source_names)}</b> document(s):<br>{listed}</div>',
                unsafe_allow_html=True,
            )
        else:
            detail_box.markdown(
                f'<div class="step-detail">🔎 Matched <b>{len(retrieved_docs)}</b> '
                f'passage(s).</div>',
                unsafe_allow_html=True,
            )

        progress.progress(50)

        graph = build_graph(retriever)

        # -------- STEP 3: run the investigation graph --------

        status.markdown(
            '<div class="step-status">🔍 Comparing and verifying the information...</div>',
            unsafe_allow_html=True,
        )
        progress.progress(70)

        result = graph.invoke(
            {
                "mission": st.session_state.question
            }
        )

        evidence_count = len(get_evidence(result))

        status.markdown(
            '<div class="step-status">✅ Checking the available evidence...</div>',
            unsafe_allow_html=True,
        )

        detail_box.markdown(
            f'<div class="step-detail">📋 Investigation used '
            f'<b>{evidence_count}</b> piece(s) of evidence.</div>',
            unsafe_allow_html=True,
        )

        progress.progress(90)

        st.session_state.result = result

        progress.progress(100)

        status.markdown(
            '<div class="step-status done">🎉 Investigation complete.</div>',
            unsafe_allow_html=True,
        )

        st.write("")

        if st.button(
            "View Investigation Results  →",
            use_container_width=True,
            type="primary",
        ):
            next_page()
            st.rerun()

    except Exception as e:

        progress.empty()
        status.empty()
        detail_box.empty()

        st.error(
            "We couldn't complete the investigation right now."
        )

        st.info(
            "Please check that your knowledge documents and AI "
            "connection are available, then try again."
        )

        with st.expander("Technical details"):
            st.code(str(e))

        if st.button("← Return to Question"):
            previous_page()
            st.rerun()


# ============================================================
# PAGE 5 — RESULTS
# ============================================================

elif st.session_state.page == 5:

    render_html('<div class="page-label">STEP 5 OF 5</div>')

    render_html(
        """
        <div class="hero" style="padding-top:10px;">
            <h1 style="font-size:36px;">
                Your Investigation
            </h1>
            <p>
                Here's what KnowDetective found in your knowledge.
            </p>
        </div>
        """
    )

    result = st.session_state.result

    if not result:

        st.info("No investigation result is available.")

        if st.button("Start Investigation"):
            st.session_state.page = 3
            st.rerun()

    else:

        report = get_report(result)
        evidence = get_evidence(result)

        # ----------------------------------------------------
        # QUESTION
        # ----------------------------------------------------

        render_html(
            f"""
            <div class="card">
                <h3>💬 Your question</h3>
                <p>
                    {html.escape(st.session_state.question)}
                </p>
            </div>
            """
        )

        # ----------------------------------------------------
        # ANSWER
        # ----------------------------------------------------

        if report:

            render_html(
                """
                <div class="result-box">
                    <div class="result-title">
                        📋 What we found
                    </div>
                """
            )

            st.markdown(report)

            render_html("</div>")

        # ----------------------------------------------------
        # EVIDENCE
        # ----------------------------------------------------

        if evidence:

            render_html(
                """
                <div style="margin-top:35px;">
                    <h2 style="font-size:25px;">
                        📚 Information used
                    </h2>
                    <p style="color:#667085;">
                        These documents were used to investigate your question.
                    </p>
                </div>
                """
            )

            for index, item in enumerate(evidence, start=1):

                source, content = evidence_details(item, index)

                safe_source = html.escape(source)
                safe_content = html.escape(content)

                render_html(
                    f"""
                    <div class="evidence">
                        <div class="source">
                            📄 {safe_source}
                        </div>
                        <div class="source-text">
                            {safe_content}
                        </div>
                    </div>
                    """
                )

        # ----------------------------------------------------
        # ACTIONS
        # ----------------------------------------------------

        st.write("")
        st.write("")

        col1, col2 = st.columns(2)

        with col1:
            if st.button(
                "← Ask another question",
                use_container_width=True,
            ):
                st.session_state.question = ""
                st.session_state.result = None
                st.session_state.page = 3
                st.rerun()

        with col2:
            if st.button(
                "🏠 Start over",
                use_container_width=True,
            ):
                st.session_state.question = ""
                st.session_state.result = None
                st.session_state.page = 1
                st.rerun()


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <div class="footer">
        KnowDetective AI · Enterprise Knowledge Investigation
    </div>
    """
)