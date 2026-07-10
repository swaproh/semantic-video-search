import os
import json
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


model = SentenceTransformer("all-MiniLM-L6-v2")

def normalize_path(p: str) -> str:
    if p is None:
        return None
    return p.replace("\\", "/")


def embed_chunks(chunks_path, output_dir="data/embeddings"):
    chunks_path = normalize_path(chunks_path)
    output_dir = normalize_path(output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Load chunks
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = []
    for c in chunks:
        if isinstance(c, dict) and "text" in c:
            texts.append(c["text"])
        else:
            texts.append(str(c))

    # Compute embeddings (NumPy array)
    embeddings = model.encode(texts, convert_to_tensor=False)

    # Convert to Python list
    embeddings_list = embeddings.tolist()

    # Build output path
    base = os.path.basename(chunks_path).split('.')[0]
    out_path = f"{output_dir}/{base}_embeddings.json"
    out_path = normalize_path(out_path)

    # Save JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(embeddings_list, f, indent=2, ensure_ascii=False)

    print(f"Embeddings saved to: {out_path}")
    return out_path
