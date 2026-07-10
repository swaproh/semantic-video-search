import os
import json
import numpy as np


def normalize_path(p: str) -> str:
    if p is None:
        return None
    return p.replace("\\", "/")


def chunk_transcript(transcript_path, output_dir="data/chunks"):
    # Normalize paths
    transcript_path = normalize_path(transcript_path)
    output_dir = normalize_path(output_dir)

    # Ensure chunk directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load transcript JSON
    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data.get("segments", [])

    # Build output file path
    base = os.path.basename(transcript_path).split('.')[0]
    out_path = f"{output_dir}/{base}_chunks.json"
    out_path = normalize_path(out_path)

    # Save chunks
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=2, ensure_ascii=False)

    print(f"Chunks saved to: {out_path}")
    return out_path
