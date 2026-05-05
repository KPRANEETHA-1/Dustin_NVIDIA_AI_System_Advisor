"""Build sentence embeddings and a FAISS index from chunks.json."""

import json
import os
import time
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# PRIMARY MODEL: NV-Embed-v2
#   This is NVIDIA's own embedding model, trained on NVIDIA docs
#   and ML engineering text. It scores #1 on MTEB retrieval benchmark.
#   Model card: https://huggingface.co/nvidia/NV-Embed-v2
#
# FALLBACK MODEL: all-MiniLM-L6-v2
#   If NV-Embed-v2 fails to download (slow internet, no HuggingFace access),
#   this fallback is smaller (384-dim instead of 768) but still solid.
#   We detect the fallback and adjust VECTOR_DIM automatically.

PRIMARY_MODEL   = "nvidia/NV-Embed-v2"
FALLBACK_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
CHUNKS_PATH     = os.path.join(SCRIPT_DIR, "chunks.json")
INDEX_PATH      = os.path.join(SCRIPT_DIR, "faiss_index.index")
STORE_PATH      = os.path.join(SCRIPT_DIR, "chunk_store.json")

BATCH_SIZE = 32



def load_model() -> tuple[SentenceTransformer, int, str]:
    """Load the preferred embedding model, falling back to MiniLM if needed."""
    print("\n" + "=" * 60)
    print("  STEP 1 â€” Loading embedding model")
    print("=" * 60)

    for model_name in [PRIMARY_MODEL, FALLBACK_MODEL]:
        try:
            print(f"\n  Trying: {model_name}")
            print(f"  (First run downloads the model â€” may take a minute)")

            # SentenceTransformer() downloads the model from HuggingFace
            # on first use, then caches it locally (~/.cache/huggingface/).
            # Subsequent runs load from cache instantly.
            #
            # trust_remote_code=True is needed for NV-Embed-v2 because it
            # uses custom pooling code that lives in the model's repository.
            # Without this flag, the library refuses to run non-standard code.
            model = SentenceTransformer(model_name, trust_remote_code=True)

            # encode() returns a 2D numpy array of shape (num_sentences, dim).
            # We encode 1 sentence, so shape is (1, dim). We grab dim with [1].
            dummy_vec = model.encode(["test"], show_progress_bar=False)
            vector_dim = dummy_vec.shape[1]

            print(f"  âœ… Loaded: {model_name}")
            print(f"  Vector dimension: {vector_dim}")
            return model, vector_dim, model_name

        except Exception as e:
            print(f"  âš ï¸  Failed: {e}")
            if model_name == FALLBACK_MODEL:
                raise RuntimeError("Both models failed to load. Check internet connection.")
            print(f"  Falling back to: {FALLBACK_MODEL}\n")

    raise RuntimeError("Unreachable")



def embed_chunks(
    chunks: list[dict],
    model: SentenceTransformer,
    model_name: str
) -> np.ndarray:
    """Convert chunk text into normalized embedding vectors."""
    print("\n" + "=" * 60)
    print("  STEP 2 â€” Embedding all chunks")
    print("=" * 60)

    texts = [chunk["content"] for chunk in chunks]

    print(f"\n  Chunks to embed : {len(texts)}")
    print(f"  Model           : {model_name}")
    print(f"  Batch size      : {BATCH_SIZE}")
    print(f"\n  Embedding in progress...")

    start_time = time.time()

    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    elapsed = time.time() - start_time

    print(f"\n  âœ… Embedding complete")
    print(f"  Time taken      : {elapsed:.2f}s")
    print(f"  Output shape    : {vectors.shape}  â† ({len(chunks)} chunks Ã— {vectors.shape[1]} dimensions)")
    print(f"  Memory usage    : ~{vectors.nbytes / 1024:.1f} KB")

    # np.linalg.norm computes the length of each vector.
    # All normalized vectors should have length â‰ˆ 1.0.
    # We check the first 5 to confirm normalization worked.
    norms = np.linalg.norm(vectors[:5], axis=1)
    print(f"  Vector norms    : {np.round(norms, 4)}  â† should all be â‰ˆ 1.0")

    return vectors



def build_faiss_index(vectors: np.ndarray, vector_dim: int) -> faiss.Index:
    """Build an exact cosine-similarity FAISS index."""
    print("\n" + "=" * 60)
    print("  STEP 3 â€” Building FAISS index")
    print("=" * 60)

    print(f"\n  Index type  : IndexFlatIP (exact cosine similarity)")
    print(f"  Vector dim  : {vector_dim}")
    print(f"  Num vectors : {len(vectors)}")

    # faiss.IndexFlatIP(vector_dim) creates an empty flat inner product index.
    # It needs vector_dim so it knows the shape of vectors it will receive.
    index = faiss.IndexFlatIP(vector_dim)

    # By default numpy uses float64 (64-bit floats = doubles).
    # FAISS only accepts float32 (32-bit floats).
    # .astype(np.float32) converts the array without changing its values.
    # If you skip this you get a cryptic C++ error inside FAISS.
    vectors_f32 = vectors.astype(np.float32)

    # index.add() takes the 2D array (57, 768) and stores all vectors.
    # After this call, index.ntotal == 57.
    # The position of each vector in the index (0, 1, 2, ...) matches
    # its position in our chunks list â€” chunk_store[i] â†” index vector i.
    index.add(vectors_f32)

    print(f"\n  âœ… Index built")
    print(f"  Vectors stored : {index.ntotal}")
    print(f"  Index is trained: {index.is_trained}  â† Flat indexes need no training")

    return index



def save_artifacts(
    index: faiss.Index,
    chunks: list[dict],
    vectors: np.ndarray,
    model_name: str,
    vector_dim: int
) -> None:
    """Save the FAISS index and chunk metadata used by the retriever."""
    print("\n" + "=" * 60)
    print("  STEP 4 â€” Saving to disk")
    print("=" * 60)

    faiss.write_index(index, INDEX_PATH)
    index_size_kb = os.path.getsize(INDEX_PATH) / 1024
    print(f"\n  âœ… FAISS index   â†’ faiss_index.index  ({index_size_kb:.1f} KB)")

    # We store the chunks alongside metadata so the retriever knows exactly
    # which model was used (for future re-embedding checks) and the dimensions.
    chunk_store = {
        "metadata": {
            "model_name":    model_name,
            "vector_dim":    vector_dim,
            "num_chunks":    len(chunks),
            "index_type":    "IndexFlatIP",
            "normalized":    True,
            "similarity":    "cosine",
        },
        "chunks": chunks,
    }

    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(chunk_store, f, indent=2, ensure_ascii=False)

    store_size_kb = os.path.getsize(STORE_PATH) / 1024
    print(f"  âœ… Chunk store   â†’ chunk_store.json   ({store_size_kb:.1f} KB)")

    print(f"\n  Saved metadata:")
    for k, v in chunk_store["metadata"].items():
        print(f"    {k:<15} : {v}")



def verify_retrieval(
    index: faiss.Index,
    chunks: list[dict],
    model: SentenceTransformer,
    model_name: str
) -> None:
    """Run sample queries to sanity-check the saved index."""
    print("\n" + "=" * 60)
    print("  STEP 5 â€” Verifying retrieval")
    print("=" * 60)

    test_queries = [
        "How does Triton handle dynamic batching?",
        "What is TensorRT quantization?",
        "How do I fine-tune a model with NeMo?",
    ]

    for query in test_queries:
        print(f"\n  Query: \"{query}\"")

        # This is critical: both chunks and queries MUST be embedded by the
        # same model. Mixing models produces meaningless distance scores.
        query_vec = model.encode(
            [query],                        # must be a list, even for one query
            normalize_embeddings=True,      # same normalization as chunks
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)               # float32 required by FAISS

        # query_vec shape: (1, 768) â€” one query, 768 dimensions
        # k=3 means return the 3 most similar chunks
        distances, indices = index.search(query_vec, k=3)

        # distances[0] and indices[0] because we searched 1 query (batch size 1)
        for rank, (dist, idx) in enumerate(zip(distances[0], indices[0]), 1):
            chunk = chunks[idx]
            print(f"  #{rank}  score={dist:.4f}  [{chunk['chunk_id']}]  \"{chunk['section']}\"")
            print(f"       {chunk['content'][:100].replace(chr(10), ' ')}...")



def main():
    print("=" * 60)
    print("  NVIDIA AI Advisor â€” Embedder")
    print("  chunks.json â†’ vectors â†’ FAISS index")
    print("=" * 60)

    print(f"\n  Loading chunks from: {CHUNKS_PATH}")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  Chunks loaded: {len(chunks)}")
    print(f"  Sources: {sorted(set(c['source'] for c in chunks))}")

    model, vector_dim, model_name = load_model()

    vectors = embed_chunks(chunks, model, model_name)

    index = build_faiss_index(vectors, vector_dim)

    save_artifacts(index, chunks, vectors, model_name, vector_dim)

    verify_retrieval(index, chunks, model, model_name)

    print("\n" + "=" * 60)
    print("  ALL DONE â€” Your retrieval layer is ready")
    print("=" * 60)
    print(f"""
  Files created:
    faiss_index.index  â† load with faiss.read_index()
    chunk_store.json   â† load with json.load()

  What's next:
    rag_pipeline.py    â† query loop: embed query â†’ FAISS search
                          â†’ fetch chunks â†’ send to LLM â†’ get answer

  The full RAG flow:
    User question
        â†“  embed with {model_name.split('/')[-1]}
    Query vector (1 Ã— {vector_dim})
        â†“  FAISS IndexFlatIP search
    Top-3 chunk indices + scores
        â†“  look up chunk_store["chunks"][i]
    Retrieved text chunks
        â†“  inject into LLM prompt
    Final answer with source citations
    """)


if __name__ == "__main__":
    main()






