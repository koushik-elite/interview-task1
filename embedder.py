"""
embedder.py  —  BGE-M3 embedding for corpus and queries.

BGE-M3 is a single model that produces THREE vector types simultaneously:
  dense_vecs      → 1024-dim float vector  (semantic similarity)
  lexical_weights → sparse dict {token_id: weight}  (exact term matching)
  colbert_vecs    → multi-vector  (token-level interaction)  ← not used in POC

For our hybrid RAG pipeline we use:
  corpus  → encode_corpus()  → dense + sparse
  query   → encode_queries() → dense + sparse
  (BGE-M3 uses different internal logic for queries vs documents)

Install:
  pip install FlagEmbedding
  pip install torch   (CPU version is fine for POC)
"""

import json
from pathlib import Path
from FlagEmbedding import BGEM3FlagModel


# ── CONFIG ────────────────────────────────────────────────────────────────────

MODEL_NAME  = "BAAI/bge-m3"   # downloads ~2.2GB on first run, cached after
BATCH_SIZE  = 4                # keep low for CPU / Kaggle free tier
MAX_LENGTH  = 512              # 10-Q chunks are rarely longer than this
USE_FP16    = False            # True only if you have a GPU


# ── SINGLETON MODEL LOADER ────────────────────────────────────────────────────
# Load once, reuse everywhere — avoids re-downloading on every call

_model = None

def get_model() -> BGEM3FlagModel:
    """
    Return the BGE-M3 model, loading it only on the first call.
    Subsequent calls return the cached instance.
    """
    global _model
    if _model is None:
        print(f"Loading BGE-M3 model: {MODEL_NAME} ...")
        _model = BGEM3FlagModel(
            MODEL_NAME,
            use_fp16=USE_FP16,   # fp16 = faster on GPU, keep False on CPU
        )
        print("Model loaded.\n")
    return _model


# ── SPARSE VECTOR CONVERTER ───────────────────────────────────────────────────

def lexical_weights_to_sparse(lexical_weights: dict) -> dict:
    """
    Convert BGE-M3 lexical_weights output to Milvus Lite sparse format.

    BGE-M3 returns:  { "token_id_str": weight_float, ... }
    Milvus needs:    { "indices": [int, ...], "values": [float, ...] }
    """
    indices = [int(k)   for k in lexical_weights.keys()]
    values  = [float(v) for v in lexical_weights.values()]
    return {"indices": indices, "values": values}


# ── CORPUS EMBEDDER ───────────────────────────────────────────────────────────

def embed_corpus(texts: list[str]) -> list[dict]:
    """
    Embed a list of document chunks (corpus side).

    Uses encode_corpus() — BGE-M3 treats documents differently from queries
    internally (no instruction prefix added for documents).

    Returns list of:
      {
        "dense":  [float, ...]        # 1024-dim vector
        "sparse": {"indices": [...],  # token ids
                   "values":  [...]}  # term weights
      }
    """
    model = get_model()

    print(f"  Embedding {len(texts)} corpus chunks "
          f"(batch_size={BATCH_SIZE}) ...")

    output = model.encode_corpus(
        texts,
        batch_size          = BATCH_SIZE,
        max_length          = MAX_LENGTH,
        return_dense        = True,
        return_sparse       = True,
        return_colbert_vecs = False,   # not needed for POC
    )

    dense_vecs      = output["dense_vecs"]       # np.ndarray (N, 1024)
    lexical_weights = output["lexical_weights"]  # list of dicts

    results = []
    for i in range(len(texts)):
        results.append({
            "dense":  dense_vecs[i].tolist(),
            "sparse": lexical_weights_to_sparse(lexical_weights[i]),
        })

    print(f"  Done. Dense dim={len(results[0]['dense'])}  "
          f"Sparse tokens={len(results[0]['sparse']['indices'])}\n")

    return results


# ── QUERY EMBEDDER ────────────────────────────────────────────────────────────

def embed_query(query: str) -> dict:
    """
    Embed a single user query (query side).

    Uses encode_queries() — BGE-M3 adds an internal instruction prefix
    for queries to improve retrieval quality.

    Returns:
      {
        "dense":  [float, ...]        # 1024-dim vector
        "sparse": {"indices": [...],
                   "values":  [...]}
      }
    """
    model = get_model()

    output = model.encode_queries(
        [query],                       # always pass as list
        batch_size          = 1,
        max_length          = 512,
        return_dense        = True,
        return_sparse       = True,
        return_colbert_vecs = False,
    )

    dense_vecs      = output["dense_vecs"]
    lexical_weights = output["lexical_weights"]

    return {
        "dense":  dense_vecs[0].tolist(),
        "sparse": lexical_weights_to_sparse(lexical_weights[0]),
    }


# ── DEMO MAIN ─────────────────────────────────────────────────────────────────

def main():
    """
    Demo: embed a small corpus of 10-Q-style sentences + one query.
    Shows the shape of dense and sparse outputs.
    No real PDF needed — uses hardcoded sample sentences.
    """

    # ── Sample corpus (simulating chunked 10-Q content) ───────────────────────
    corpus = [
        # text chunks
        "Apple's total net sales for the third quarter of 2022 were $82.96 billion, "
        "compared to $81.43 billion in the same quarter of 2021.",

        "iPhone net sales increased to $40.67 billion in Q3 2022 from $39.57 billion "
        "in Q3 2021, driven by strong demand across all geographies.",

        "Services revenue reached $19.6 billion in Q3 2022, representing a new "
        "quarterly record and growing 12% year-over-year.",

        # table chunk (markdown table as a string — same pipeline, same encoder)
        "| Product     | Q3 2022 ($M) | Q3 2021 ($M) |\n"
        "|-------------|-------------|-------------|\n"
        "| iPhone      | 40,665      | 39,570       |\n"
        "| Mac         | 7,382       | 9,178        |\n"
        "| iPad        | 7,224       | 7,368        |\n"
        "| Wearables   | 8,084       | 8,775        |\n"
        "| Services    | 19,604      | 17,486       |\n"
        "| Total       | 82,959      | 82,377       |",

        "Risk factors include macroeconomic conditions, foreign exchange fluctuations, "
        "and supply chain constraints that could adversely affect Apple's results.",
    ]

    # ── Sample queries ─────────────────────────────────────────────────────────
    queries = [
        "What was Apple's iPhone revenue in Q3 2022?",
        "What are the main risk factors for AAPL?",
    ]

    print("=" * 60)
    print("  BGE-M3 Embedding Demo")
    print("=" * 60)

    # ── Embed corpus ──────────────────────────────────────────────────────────
    print("\n[ CORPUS ]\n")
    corpus_embeddings = embed_corpus(corpus)

    # Show shape for each chunk
    for i, (text, emb) in enumerate(zip(corpus, corpus_embeddings)):
        preview  = text[:60].replace("\n", " ") + "..."
        n_sparse = len(emb["sparse"]["indices"])
        print(f"  Chunk {i+1}: dense=({len(emb['dense'])},)  "
              f"sparse={n_sparse} tokens  |  {preview}")

    # ── Embed queries ─────────────────────────────────────────────────────────
    print("\n[ QUERIES ]\n")
    query_embeddings = []
    for q in queries:
        print(f"  Query: \"{q}\"")
        qemb = embed_query(q)
        query_embeddings.append(qemb)
        print(f"    dense=({len(qemb['dense'])},)  "
              f"sparse={len(qemb['sparse']['indices'])} tokens\n")

    # ── Show raw sparse weights for first query (insight into BGE-M3 sparse) ──
    print("[ SPARSE WEIGHT SAMPLE — Query 1 ]\n")
    sparse = query_embeddings[0]["sparse"]
    # Sort by weight descending to see top tokens
    pairs = sorted(
        zip(sparse["indices"], sparse["values"]),
        key=lambda x: x[1], reverse=True
    )
    print(f"  Top 10 sparse token weights:")
    for token_id, weight in pairs[:10]:
        print(f"    token_id={token_id:6d}  weight={weight:.4f}")

    # ── Save sample output to JSON (so you can inspect the shape) ─────────────
    sample_output = {
        "corpus_embeddings": [
            {
                "text_preview": t[:80],
                "dense_dim":    len(e["dense"]),
                "dense_sample": e["dense"][:5],   # first 5 dims only
                "sparse_token_count": len(e["sparse"]["indices"]),
                "sparse_top3": dict(
                    sorted(
                        zip(e["sparse"]["indices"], e["sparse"]["values"]),
                        key=lambda x: x[1], reverse=True
                    )[:3]
                ),
            }
            for t, e in zip(corpus, corpus_embeddings)
        ],
        "query_embeddings": [
            {
                "query":        q,
                "dense_dim":    len(e["dense"]),
                "dense_sample": e["dense"][:5],
                "sparse_token_count": len(e["sparse"]["indices"]),
            }
            for q, e in zip(queries, query_embeddings)
        ],
    }

    out_path = "embeddings_sample.json"
    with open(out_path, "w") as f:
        json.dump(sample_output, f, indent=2)

    print(f"\n  Sample saved to: {out_path}")
    print("\n" + "=" * 60)
    print("  Next step: store these in Milvus Lite")
    print("=" * 60)


if __name__ == "__main__":
    main()