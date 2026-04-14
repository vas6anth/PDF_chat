"""
qa_chain.py
Builds the RAG prompt, calls gpt-4o-mini, and returns a structured result
containing the answer text and citation metadata.

LLM: gpt-4o-mini (cheapest OpenAI chat model with good quality)
"""

from openai import OpenAI


LLM_MODEL = "gpt-4o-mini"
MAX_HISTORY_TURNS = 6        # last N turns fed to the model for context


# ── Prompt builders ───────────────────────────────────────────────────────────
def _build_context_block(docs: list) -> str:
    parts = []
    for i, doc in enumerate(docs, start=1):
        page = doc.metadata.get("page_num", "?")
        parts.append(f"[Source {i} | Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _history_messages(chat_history: list) -> list[dict]:
    """Convert stored chat turns into OpenAI message format (last N turns)."""
    msgs = []
    for turn in chat_history[-MAX_HISTORY_TURNS:]:
        msgs.append({"role": "user", "content": turn["user"]})
        msgs.append({"role": "assistant", "content": turn["answer"]})
    return msgs


# ── Public API ────────────────────────────────────────────────────────────────
def answer_question(
    query: str,
    docs: list,
    chat_history: list,
    client: OpenAI,
) -> dict:
    """
    Generate a grounded answer with citations.

    Args:
        query       : original user question
        docs        : relevant LangChain Document objects (may be empty)
        chat_history: list of previous {"user", "answer", "sources"} dicts
        client      : OpenAI client

    Returns:
        {
          "answer"  : str,
          "sources" : list[dict],   # see schema below
          "used_pdf": bool,
        }

    Source schema (PDF hit):
        {"type":"PDF", "source_num":int, "page":int, "snippet":str}
    Source schema (LLM fallback):
        {"type":"LLM", "note":str}
    """
    has_pdf_context = bool(docs)

    if has_pdf_context:
        context_block = _build_context_block(docs)

        system_prompt = (
            "You are a precise document assistant. "
            "Answer the user's question using ONLY the context passages provided below. "
            "Cite every claim with its source tag, e.g. [Source 1 | Page 3]. "
            "If the context is insufficient for a complete answer, say so honestly "
            "and note which parts you could not find. "
            "Do NOT invent information."
        )

        user_content = (
            f"DOCUMENT CONTEXT:\n{context_block}\n\n"
            f"QUESTION: {query}\n\n"
            "Provide a thorough answer with inline citations using [Source N | Page P] format."
        )

    else:
        system_prompt = (
            "You are a knowledgeable assistant. "
            "The uploaded PDF does not contain information relevant to this question. "
            "Answer from your general knowledge and clearly state that "
            "this answer is NOT from the document."
        )
        user_content = query

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_history_messages(chat_history))
    messages.append({"role": "user", "content": user_content})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=1200,
        temperature=0.2,
    )

    answer_text = response.choices[0].message.content.strip()

    # Build sources list
    if has_pdf_context:
        sources = [
            {
                "type": "PDF",
                "source_num": i + 1,
                "page": doc.metadata.get("page_num", "?"),
                "snippet": doc.page_content[:180].replace("\n", " ").strip() + "…",
            }
            for i, doc in enumerate(docs)
        ]
    else:
        sources = [
            {
                "type": "LLM",
                "note": (
                    "This answer comes from the model's general knowledge. "
                    "The uploaded PDF did not contain relevant information."
                ),
            }
        ]

    return {
        "answer": answer_text,
        "sources": sources,
        "used_pdf": has_pdf_context,
    }
