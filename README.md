# PDF Chat with Citations

A Streamlit app that lets you upload any PDF (≤ 10 pages) and chat with it using RAG — with page-level citations on every answer.

---

## Quick Start

```bash
git clone https://github.com/vas6anth/PDF_chat
cd ManPower_assignment
create venv
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501**, paste your OpenAI API key, upload a PDF, and start chatting.

---

## Environment Variables

No `.env` file required — the API key is entered directly in the Streamlit sidebar and kept only in session memory.

---

## Architecture

```
app.py
└─ modules/
   ├─ pdf_processor.py   PDF loading, page-limit check, MD5 hashing
   ├─ embeddings.py      Chunking, OpenAI embeddings, FAISS store/load
   ├─ retriever.py       Query rewrite, MMR retrieval, relevance gating
   ├─ qa_chain.py        RAG prompt, GPT-4o-mini call, citation assembly
   └─ utils.py           API-key validation, client factory, error formatting
storage/
└─ <md5-hash>/           One folder per unique PDF (FAISS index.faiss + index.pkl)
```

---

## Models Used

| Component | Model | Notes |
|-----------|-------|-------|
| **Embeddings** | `text-embedding-3-small` | 1 536 dims · ~$0.00002 / 1k tokens |
| **LLM** | `gpt-4o-mini` | Cheapest OpenAI chat model with strong reasoning |

---

## Chunking Strategy

- **Splitter**: `RecursiveCharacterTextSplitter` (LangChain)  
- **Chunk size**: 500 tokens  
- **Overlap**: 50 tokens  
- **Length function**: `tiktoken` cl100k_base encoder  
- **Separators**: `["\n\n", "\n", ". ", " ", ""]` — tries to preserve paragraph → sentence → word boundaries  

*Why 500/50?* Large enough to contain a coherent passage; small enough for precise retrieval. 50-token overlap prevents losing context at boundaries.

---

## Retrieval Strategy

1. **Query Rewrite** (`gpt-4o-mini`, T=0): condenses the user question into a short, keyword-rich retrieval query, handling vague follow-ups correctly.  
2. **MMR Search** (Maximal Marginal Relevance, k=4, fetch_k=12): returns 4 relevant *and diverse* chunks — avoids 4 near-duplicate passages.  
3. **Relevance Gating** (`gpt-4o-mini`, T=0): each chunk is binary-judged (yes/no). Chunks that don't help answer the question are dropped before the LLM sees them.  
4. **LLM Fallback**: if zero chunks pass the relevance gate, the question is answered from general knowledge and the source label shows **LLM** instead of **PDF**.

---

## Conversation History

- Last **6 turns** are prepended as alternating `user` / `assistant` messages in every API call.  
- History is stored in `st.session_state` — it lives for the duration of the browser session.  
- Uploading a new PDF automatically resets the history.

---

## Caching

- Each PDF is hashed with MD5.  
- Before embedding, the app checks `storage/<hash>/index.faiss`.  
- If the file exists, the index is loaded from disk — **no API calls, instant load**.  
- This means the same PDF is never embedded twice, even across app restarts.

---

## Known Limitations

| Limitation | Detail |
|------------|--------|
| **10-page cap** | Scoped to the assignment spec; remove the check in `pdf_processor.py` for larger docs |
| **Scanned PDFs** | `pdfplumber` can't OCR images — text-based PDFs only |
| **Single file** | One PDF active at a time; uploading a second resets the session |
| **No streaming** | Full response appears at once (easy to add with `stream=True`) |
| **Session-only history** | Chat history is lost on page refresh |
| **FAISS on disk** | Not suitable for multi-user production; swap to Chroma/Qdrant for that |

---

## Eval (sample Q&A — manual judgment)

**please check the "sample_test_cases" folder for test cases and recordings**
