"""
app.py  ·  PDF Chat with Citations
Run: streamlit run app.py
"""

import streamlit as st
from openai import OpenAI

from modules.pdf_processor import load_pdf, get_pdf_hash, MAX_PAGES
from modules.embeddings import (
    chunk_pages,
    build_and_save,
    load_vectorstore,
    is_cached,
)
from modules.retriever import retrieve_relevant_chunks
from modules.qa_chain import answer_question
from modules.utils import validate_api_key, make_client, friendly_error


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF Chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS tweaks ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .source-box {
        background: #f0f4ff;
        border-left: 4px solid #4f8ef7;
        padding: 8px 12px;
        border-radius: 4px;
        margin: 4px 0;
        font-size: 0.85rem;
    }
    .llm-box {
        background: #fff8e1;
        border-left: 4px solid #f5a623;
        padding: 8px 12px;
        border-radius: 4px;
        margin: 4px 0;
        font-size: 0.85rem;
    }
    .stChatMessage { padding: 8px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state initialisation ──────────────────────────────────────────────
def _init_state():
    defaults = {
        "chat_history": [],
        "vectorstore": None,
        "pdf_hash": None,
        "pdf_name": None,
        "api_key_valid": False,
        "client": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📄 PDF Chat")
    st.caption("Powered by GPT-4o-mini + FAISS")
    st.divider()

    # ── API Key ──
    st.subheader("🔑 OpenAI API Key")
    api_key_input = st.text_input(
        "Paste your key",
        type="password",
        placeholder="sk-...",
        help="Your key is never stored permanently.",
    )

    if api_key_input:
        if api_key_input != getattr(st.session_state, "_last_key", ""):
            with st.spinner("Validating key…"):
                valid, err = validate_api_key(api_key_input)
            st.session_state.api_key_valid = valid
            st.session_state._last_key = api_key_input
            if valid:
                st.session_state.client = make_client(api_key_input)
            else:
                st.session_state.client = None

        if st.session_state.api_key_valid:
            st.success("✅ Key validated")
        else:
            st.error(f"❌ {err if api_key_input else 'Enter your key'}")

    st.divider()

    # ── PDF Upload ──
    st.subheader(f"📁 Upload PDF  *(max {MAX_PAGES} pages)*")
    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help=f"Maximum {MAX_PAGES} pages. File is processed once and cached locally.",
    )

    # ── Process PDF ──
    if uploaded_file and st.session_state.api_key_valid:
        pdf_bytes = uploaded_file.read()
        pdf_hash = get_pdf_hash(pdf_bytes)

        if pdf_hash != st.session_state.pdf_hash:
            # New (or first) PDF — reset chat
            st.session_state.chat_history = []
            st.session_state.vectorstore = None
            st.session_state.pdf_hash = None

            with st.spinner("Processing PDF…"):
                try:
                    if is_cached(pdf_hash):
                        vs = load_vectorstore(pdf_hash, api_key_input)
                        st.success("⚡ Loaded from cache — no re-embedding!")
                    else:
                        pages = load_pdf(pdf_bytes)
                        st.info(f"📄 {len(pages)} page(s) extracted")
                        chunks = chunk_pages(pages)
                        st.info(f"🔢 {len(chunks)} chunks created")
                        with st.spinner("Embedding… (one-time only)"):
                            vs = build_and_save(chunks, pdf_hash, api_key_input)
                        st.success("✅ Embedded and cached!")

                    st.session_state.vectorstore = vs
                    st.session_state.pdf_hash = pdf_hash
                    st.session_state.pdf_name = uploaded_file.name

                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(friendly_error(e))

    # ── Current status ──
    st.divider()
    if st.session_state.pdf_name:
        st.caption(f"**Active doc:** {st.session_state.pdf_name}")
        n = len(st.session_state.chat_history)
        st.caption(f"**Turns:** {n}")
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    # ── Model info ──
    st.divider()
    with st.expander("ℹ️ Model & Config"):
        st.markdown(
            """
| Setting | Value |
|---|---|
| **Embedding** | `text-embedding-3-small` |
| **Embedding dims** | 1 536 |
| **LLM** | `gpt-4o-mini` |
| **Chunk size** | 500 tokens |
| **Chunk overlap** | 50 tokens |
| **Retrieval** | MMR, k=4 |
| **Vector DB** | FAISS (local) |
| **Max pages** | 10 |
"""
        )


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("💬 Chat with your PDF")

# Guard: prerequisites not met
if not st.session_state.api_key_valid:
    st.info("👈 Enter your OpenAI API key in the sidebar to get started.")
    st.stop()

if not uploaded_file:
    st.info("👈 Upload a PDF (≤ 10 pages) in the sidebar.")
    st.stop()

if st.session_state.vectorstore is None:
    st.warning("PDF is still being processed…")
    st.stop()

st.caption(f"Chatting with **{st.session_state.pdf_name}**")
st.divider()


# ── Render chat history ───────────────────────────────────────────────────────
def _render_sources(sources: list):
    for src in sources:
        if src["type"] == "LLM":
            st.markdown(
                f'<div class="llm-box">🤖 <b>Source: LLM</b> — {src["note"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="source-box">'
                f'📖 <b>Source {src["source_num"]}</b> · Page {src["page"]}<br>'
                f'<i>{src["snippet"]}</i>'
                f'</div>',
                unsafe_allow_html=True,
            )


for turn in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        st.write(turn["answer"])
        if turn.get("sources"):
            with st.expander("📚 Sources / Citations"):
                _render_sources(turn["sources"])


# ── Chat input ────────────────────────────────────────────────────────────────
query = st.chat_input("Ask anything about your PDF…")

if query:
    query = query.strip()
    if not query:
        st.stop()

    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("_Thinking…_")

        try:
            client: OpenAI = st.session_state.client

            # 1. Retrieve relevant chunks (includes query rewrite + relevance gate)
            docs, rewritten_q = retrieve_relevant_chunks(
                st.session_state.vectorstore,
                query,
                client,
            )

            # 2. Generate grounded answer
            result = answer_question(
                query,
                docs,
                st.session_state.chat_history,
                client,
            )

            thinking_placeholder.empty()
            st.write(result["answer"])

            # Badge
            if result["used_pdf"]:
                st.caption(f"🔍 Query searched as: *\"{rewritten_q}\"*  |  📄 {len(docs)} chunk(s) used")
            else:
                st.caption("🤖 Answered from general knowledge — PDF had no relevant content.")

            if result["sources"]:
                with st.expander("📚 Sources / Citations"):
                    _render_sources(result["sources"])

            # Persist turn
            st.session_state.chat_history.append(
                {
                    "user": query,
                    "answer": result["answer"],
                    "sources": result["sources"],
                }
            )

        except Exception as e:
            thinking_placeholder.empty()
            st.error(friendly_error(e))
