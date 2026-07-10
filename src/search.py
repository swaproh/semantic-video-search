import os
import json
import numpy as np
import faiss

def normalize_path(p: str) -> str:
    if p is None:
        return None
    return p.replace("\\", "/")

def load_chunks(chunks_path):
    chunks_path = normalize_path(chunks_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        return json.load(f)

def semantic_search(query_embedding, index, top_k=5):
    query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)
    distances, indices = index.search(query_embedding, top_k)
    return distances[0], indices[0]

def search_video(query_embedding, chunks_path, faiss_index_path):
    chunks_path = normalize_path(chunks_path)
    faiss_index_path = normalize_path(faiss_index_path)

    chunks = load_chunks(chunks_path)
    index = faiss.read_index(faiss_index_path)

    distances, top_indices = semantic_search(query_embedding, index, top_k=5)

    results = []
    for rank, idx in enumerate(top_indices):
        if idx < len(chunks):
            chunk = chunks[int(idx)]  # ensure Python int

            results.append({
                "chunk_id": int(idx),
                "text": chunk.get("text", ""),
                "start": float(chunk.get("start", 0.0)),
                "end": float(chunk.get("end", 0.0)),
                "score": float(distances[rank])
            })

    return results
