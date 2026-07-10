import whisper
import json
import os
import numpy as np


model = whisper.load_model("small")

def normalize_path(p: str) -> str:
    if p is None:
        return None
    return p.replace("\\", "/")


def transcribe_video(video_path, output_dir="data/transcripts"):
    # Normalize inputs
    video_path = normalize_path(video_path)
    output_dir = normalize_path(output_dir)

    # Ensure transcript directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Run Whisper
    result = model.transcribe(video_path, verbose=True)

    # Build transcript output path
    base = os.path.basename(video_path).split('.')[0]
    out_path = f"{output_dir}/{base}.json"
    out_path = normalize_path(out_path)

    # Save transcript JSON
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Transcript saved to: {out_path}")
    return out_path