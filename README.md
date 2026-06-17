# ALDAR – Data Science Assignment

## Building a Minimal Retrieval-Augmented Generation (RAG) Pipeline

### Overview

**Objective:** This project designs and builds a minimal Retrieval-Augmented Generation (RAG) pipeline using Python.

The pipeline operates on a dataset of ~20 PDF documents (10-Q quarterly filings) manually downloaded from the investor relations websites of major technology companies — **AAPL, AMZN, INTC, MSFT, NVDA**.

For the purpose of this assignment:
- The provided documents are used directly for implementation.
- In a real system, such documents would instead be ingested automatically from external sources or enterprise document repositories.
- The pipeline is structured the way a production-ready GenAI solution would be — modular stages for extraction, embedding, storage, retrieval, and generation — rather than a single end-to-end script.

---

### Pipeline Architecture

```
PDF Documents (data/)
        │
        ▼
 ┌──────────────────┐
 │   pdf_main.py     │  Extraction: text + tables, in reading order,
 │                    │  tagged with ticker / year / quarter metadata
 └──────────────────┘
        │  JSON per document
        ▼
      output/
        │
        ▼
 ┌──────────────────┐
 │   embedder.py     │  BGE-M3 embedding: dense (semantic) +
 │                    │  sparse (lexical) vectors per chunk
 └──────────────────┘
        │
        ▼
 ┌──────────────────────────┐
 │ my-test-embedding1.ipynb │  Stores embeddings in Milvus Lite
 │                           │  (hybrid dense + sparse collections,
 │                           │   separate text / table chunks)
 └──────────────────────────┘
        │
        ▼
 ┌──────────────────────────┐
 │ vector-search-gemini.ipynb│  Retrieval: hybrid vector search →
 │                           │  context assembly → Gemini LLM →
 │                           │  grounded financial answer
 └──────────────────────────┘
```

---

### Project Structure

```
.
├── pdf_main.py                  # PDF → structured JSON extraction
├── embedder.py                  # BGE-M3 dense + sparse embedding utilities
├── my-test-embedding1.ipynb     # Builds the Milvus Lite vector store
├── vector-search-gemini.ipynb   # Hybrid retrieval + Gemini-based RAG answering
├── data/                        # Input 10-Q PDFs (gitignored, not included)
├── output/                      # One JSON file per processed PDF
└── .gitignore
```

---

### `pdf_main.py` — Description

A PDF ingestion script that uses `pdfplumber` to extract text and tables from 10-Q filings in correct reading order, converting tables to markdown and filtering headers/footers. It parses ticker/year/quarter from filenames, then saves each document as a structured JSON file (text + table elements with metadata) into `output/`.

---

### Pipeline Stages

1. **Extraction (`pdf_main.py`)**
   Reads every PDF in `data/`, detects tables vs. paragraphs per page, converts tables to markdown, filters page headers/footers, and reconstructs top-to-bottom reading order. Filename metadata (ticker, year, quarter) is parsed and attached to every extracted element. Output: one JSON file per document in `output/`, e.g. `output/2022_Q3_AAPL.json`.

2. **Embedding (`embedder.py`)**
   Uses `BAAI/bge-m3` (via `FlagEmbedding`) to embed each text/table chunk into:
   - a **dense** 1024-dim vector for semantic similarity, and
   - a **sparse** lexical-weight vector for exact term matching.

   Documents and queries are embedded differently (`encode_corpus` vs. `encode_queries`), matching BGE-M3's internal design.

3. **Vector Storage (`my-test-embedding1.ipynb`)**
   Loads the JSON outputs, embeds them, and stores dense + sparse vectors in **Milvus Lite**, with separate collections for text chunks and table chunks, each carrying ticker/year/quarter metadata for filtering.

4. **Retrieval + Generation (`vector-search-gemini.ipynb`)**
   Embeds the user's query, performs a hybrid (dense + sparse) similarity search against Milvus Lite, assembles the top-matching text and table chunks into a grounded context, and prompts **Gemini** to answer as a financial analyst strictly from the retrieved context.

---

### Setup

```bash
pip install pdfplumber FlagEmbedding torch pymilvus[milvus-lite] milvus-lite google-generativeai
```

> Note: `BAAI/bge-m3` downloads ~2.2 GB on first run and is cached afterward. CPU works for the assignment's scale; GPU (`use_fp16=True`) speeds up embedding.

Place source PDFs in a `data/` folder at the project root before running extraction:

```
data/
├── 2022 Q3 AAPL.pdf
├── 2023 Q1 MSFT.pdf
└── ...
```

**Security note:** the Gemini API key should be supplied via an environment variable (e.g. `GEMINI_API_KEY`) rather than hardcoded in a notebook — avoid committing real keys to version control.

### Usage

```bash
# 1. Extract all PDFs in data/ into structured JSON
python pdf_main.py

# 2. (Optional) run the embedder's demo to sanity-check dense/sparse output shapes
python embedder.py
```

Then open `my-test-embedding1.ipynb` to build the vector store, and `vector-search-gemini.ipynb` to run RAG queries, e.g.:

```python
answer, prompt = rag_query("What was Apple's revenue in 2022?")
```

### Output Format

Each processed PDF produces a JSON file shaped as:

```json
{
  "metadata": {
    "ticker": "AAPL",
    "year": "2022",
    "quarter": "Q3",
    "period": "Q3_2022",
    "source_file": "2022 Q3 AAPL.pdf",
    "total_pages": 28,
    "total_elements": 224,
    "text_count": 194,
    "table_count": 30
  },
  "elements": [
    {
      "ticker": "AAPL",
      "year": "2022",
      "quarter": "Q3",
      "period": "Q3_2022",
      "type": "text",
      "page": 1,
      "position": 1,
      "content": "UNITED STATES SECURITIES AND EXCHANGE COMMISSION ..."
    }
  ]
}
```

### Tech Stack

| Component        | Tool / Library                  |
|-------------------|----------------------------------|
| PDF parsing       | `pdfplumber`                    |
| Embedding model    | `BAAI/bge-m3` (`FlagEmbedding`) |
| Vector store       | Milvus Lite (`pymilvus`)        |
| LLM (generation)   | Google Gemini                   |
