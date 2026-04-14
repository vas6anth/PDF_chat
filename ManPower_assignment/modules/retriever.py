"""
retriever.py
Query rewriting → similarity retrieval (k=4) → per-chunk relevance validation.

Strategy: MMR (Maximal Marginal Relevance) is used so returned chunks are both
relevant *and* diverse — avoids returning 4 nearly identical passages.
Relevance validation then drops any chunk that doesn't genuinely help answer
the query, signalling to qa_chain.py that the answer comes from the LLM.
"""

from openai import OpenAI
from langchain_community.vectorstores import FAISS

TOP_K = 4
RELEVANCE_THRESHOLD = 0.0   # cosine similarity floor (MMR already handles quality)


# ── Query rewriting ───────────────────────────────────────────────────────────
def rewrite_query(query: str, client: OpenAI) -> str:
    """
    Condense / clarify the user's question into a short retrieval-optimised query.
    Preserves the intent of follow-up questions that reference prior context.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a search-query optimizer. "
                    "Rewrite the user query into a short, specific, keyword-rich sentence "
                    "suitable for semantic search over a document. "
                    "Keep it under 20 words. Return ONLY the rewritten query, nothing else."
                ),
            },
            {"role": "user", "content": query},
        ],
        max_tokens=60,
        temperature=0,
    )
    rewritten = response.choices[0].message.content.strip()
    # Fallback: if model returns empty or very long text, use original
    return rewritten if rewritten and len(rewritten) < 200 else query


# ── Chunk relevance check ─────────────────────────────────────────────────────
def _is_relevant(chunk_text: str, query: str, client: OpenAI) -> bool:
    """
    Binary relevance gate — returns True if the chunk contains useful
    information for answering the query.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict relevance judge. "
                    "Reply with exactly 'yes' if the CHUNK contains information "
                    "that helps answer the QUERY, otherwise reply 'no'."
                ),
            },
            {
                "role": "user",
                "content": f"QUERY: {query}\n\nCHUNK:\n{chunk_text}",
            },
        ],
        max_tokens=5,
        temperature=0,
    )
    verdict = response.choices[0].message.content.strip().lower()
    return verdict.startswith("y")


# ── Public API ────────────────────────────────────────────────────────────────
def retrieve_relevant_chunks(
    vectorstore: FAISS,
    original_query: str,
    client: OpenAI,
    k: int = TOP_K,
) -> tuple[list, str]:
    """
    Full retrieval pipeline:
      1. Rewrite query for better embedding alignment.
      2. MMR search — k results, diverse.
      3. Per-chunk relevance validation.

    Returns:
        (relevant_docs, rewritten_query)
        relevant_docs is empty if nothing in the PDF answers the question.
    """
    rewritten = rewrite_query(original_query, client)

    # MMR fetch — fetch_k=k*3 candidates, return k diverse ones
    candidates = vectorstore.max_marginal_relevance_search(
        rewritten, k=k, fetch_k=k * 3
    )

    relevant = [
        doc for doc in candidates
        if _is_relevant(doc.page_content, rewritten, client)
    ]

    return relevant, rewritten
