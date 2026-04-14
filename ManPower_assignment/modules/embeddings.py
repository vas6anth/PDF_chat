"""
embeddings.py
Handles text chunking, OpenAI embeddings, and FAISS vector store
persistence with hash-based caching so a PDF is never re-embedded twice.

Model  : text-embedding-3-small  (1 536 dims, cheapest OpenAI embedding)
Chunking: RecursiveCharacterTextSplitter, 500 tokens / 50 token overlap
"""

from pathlib import Path

import tiktoken
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


# ── Constants ────────────────────────────────────────────────────────────────
STORAGE_DIR = Path("storage")
EMBEDDING_MODEL = "text-embedding-3-small"   # 1 536 dims, ~$0.00002 / 1k tokens
CHUNK_SIZE = 500        # tokens
CHUNK_OVERLAP = 50      # tokens


# ── Helpers ───────────────────────────────────────────────────────────────────
def _token_counter(text: str) -> int:
    """Count tokens with the cl100k_base encoder (used by all OpenAI models)."""
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def _get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=_token_counter,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _embeddings(api_key: str) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=api_key)


# ── Public API ────────────────────────────────────────────────────────────────
def chunk_pages(pages: list[dict]) -> list[dict]:
    """
    Split page texts into overlapping token-bounded chunks.

    Args:
        pages: [{"page_num": int, "text": str}, ...]

    Returns:
        [{"text": str, "page_num": int}, ...]
    """
    splitter = _get_splitter()
    chunks: list[dict] = []

    for page in pages:
        splits = splitter.split_text(page["text"])
        for split in splits:
            if split.strip():
                chunks.append({"text": split, "page_num": page["page_num"]})

    return chunks


def is_cached(pdf_hash: str) -> bool:
    """Return True if a FAISS index already exists for this PDF hash."""
    index_path = STORAGE_DIR / pdf_hash / "index.faiss"
    return index_path.exists()


def build_and_save(chunks: list[dict], pdf_hash: str, api_key: str) -> FAISS:
    """
    Embed chunks and persist the FAISS index to disk.

    Args:
        chunks  : output of chunk_pages()
        pdf_hash: MD5 hash used as the storage directory name
        api_key : OpenAI API key

    Returns:
        Loaded FAISS vector store
    """
    texts = [c["text"] for c in chunks]
    metadatas = [{"page_num": c["page_num"]} for c in chunks]

    vs = FAISS.from_texts(texts, _embeddings(api_key), metadatas=metadatas)

    save_dir = STORAGE_DIR / pdf_hash
    save_dir.mkdir(parents=True, exist_ok=True)
    vs.save_local(str(save_dir))

    return vs


def load_vectorstore(pdf_hash: str, api_key: str) -> FAISS:
    """Load a previously-saved FAISS index from disk."""
    save_dir = STORAGE_DIR / pdf_hash
    return FAISS.load_local(
        str(save_dir),
        _embeddings(api_key),
        allow_dangerous_deserialization=True,
    )
