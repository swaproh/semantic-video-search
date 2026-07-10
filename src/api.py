import os
import json
from typing import List
import numpy as np
import faiss

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response

from sentence_transformers import SentenceTransformer

from src.whisper_transcribe import transcribe_video
from src.chunker import chunk_transcript
from src.embedder import embed_chunks
from src.search import search_video

# Paths
RAW_VIDEO_DIR = "data/raw_video"
TRANSCRIPTS_DIR = "data/transcripts"
CHUNKS_DIR = "data/chunks"
EMBEDDINGS_DIR = "data/embeddings"
FAISS_DIR = "models/faiss_index"
REGISTRY_PATH = "data/registry.json"

os.makedirs("data", exist_ok=True)
os.makedirs(RAW_VIDEO_DIR, exist_ok=True)
os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs("data/subtitles", exist_ok=True)

def normalize_path(p: str) -> str:
    if p is None:
        return None
    return p.replace("\\", "/")

# Registry helpers
def load_registry() -> dict:
    if not os.path.exists(REGISTRY_PATH):
        with open(REGISTRY_PATH, "w") as f:
            json.dump({}, f)
    with open(REGISTRY_PATH, "r") as f:
        return json.load(f)

def save_registry(registry: dict) -> None:
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=4)

# Global embedding model (avoid reloading on every search)
model = SentenceTransformer("all-MiniLM-L6-v2")

# FastAPI app
app = FastAPI(title="Semantic Video Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health
@app.get("/health")
def health():
    return {"status": "ok"}

# Upload video
@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    filename = file.filename
    base = os.path.splitext(filename)[0]

    video_path = os.path.join(RAW_VIDEO_DIR, filename)
    video_path = normalize_path(video_path)

    with open(video_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return {
        "message": "Video uploaded",
        "video_name": base,
        "video_path": video_path,
    }

# Transcribe
@app.post("/transcribe")
def transcribe(video_path: str):
    video_path = normalize_path(video_path)
    transcript_path = transcribe_video(video_path)
    return {
        "message": "Transcription complete",
        "transcript_path": transcript_path
    }

@app.post("/build")
def build_pipeline(transcript_path: str):
    transcript_path = normalize_path(transcript_path)
    base = os.path.splitext(os.path.basename(transcript_path))[0]

    registry = load_registry()

    # If already built and files exist, reuse them
    if base in registry:
        entry = registry[base]
        if (
            os.path.exists(entry["faiss"]) and
            os.path.exists(entry["meta"]) and
            os.path.exists(entry["video_path"])
        ):
            return {
                "message": "Pipeline already built",
                "video_name": base,
                "video_path": entry["video_path"],
                "meta_path": entry["meta"],
                "faiss_index_path": entry["faiss"],
            }

    # 1. Load Whisper transcript (segments with timestamps)
    with open(transcript_path, "r") as f:
        whisper_data = json.load(f)

    segments = whisper_data["segments"]

    # 2. Build embeddings + metadata
    embeddings = []
    metadata = []

    for seg in segments:
        text = seg["text"]
        start = seg["start"]

        emb = model.encode(text).astype("float32")
        embeddings.append(emb)

        metadata.append({
            "text": text,
            "start": start
        })

    embeddings = np.array(embeddings)

    # 3. Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    os.makedirs("faiss_index", exist_ok=True)

    faiss_path = f"faiss_index/{base}.index"
    meta_path = f"faiss_index/{base}_meta.json"

    faiss.write_index(index, faiss_path)

    with open(meta_path, "w") as f:
        json.dump(metadata, f)

    # 4. Find matching raw video
    possible_video = None
    for fname in os.listdir(RAW_VIDEO_DIR):
        if os.path.splitext(fname)[0] == base:
            possible_video = os.path.join(RAW_VIDEO_DIR, fname)
            break

    video_path = normalize_path(possible_video) if possible_video else os.path.join(RAW_VIDEO_DIR, f"{base}.mp4")

    registry[base] = {
        "video_path": video_path,
        "faiss": faiss_path,
        "meta": meta_path
    }
    save_registry(registry)

    return {
        "message": "Pipeline built",
        "video_name": base,
        "video_path": video_path,
        "meta_path": meta_path,
        "faiss_index_path": faiss_path,
    }

# List videos
@app.get("/videos")
def list_videos():
    registry = load_registry()
    return {"videos": list(registry.keys())}

@app.post("/search")
def search_topic(payload: dict):
    video_name = payload["video_name"]
    query = payload["query"]

    index_path = f"faiss_index/{video_name}.index"
    meta_path = f"faiss_index/{video_name}_meta.json"

    if not os.path.exists(index_path):
        return {"error": "Index not found. Build pipeline first."}

    index = faiss.read_index(index_path)

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    query_emb = model.encode(query).astype("float32")

    k = 5
    distances, indices = index.search(np.array([query_emb]), k)

    results = []
    for idx in indices[0]:
        item = metadata[idx]
        results.append({
            "text": item["text"],
            "start": item["start"]
        })

    return {
        "video_name": video_name,
        "results": results
    }



# Serve raw video file
@app.get("/video/{filename}")
def get_video(filename: str):
    path = os.path.join(RAW_VIDEO_DIR, filename)
    path = normalize_path(path)
    if not os.path.exists(path):
        return {"error": "Video not found"}
    return FileResponse(path, media_type="video/mp4")

# Summary (simple text summary from transcript)
@app.get("/summary/{video_name}")
def generate_summary(video_name: str):
    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{video_name}.json")

    if not os.path.exists(transcript_path):
        return {"summary": "Transcript not found."}

    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support multiple transcript formats
    if "chunks" in data:
        full_text = " ".join([c["text"] for c in data["chunks"]])
    elif "segments" in data:
        full_text = " ".join([s["text"] for s in data["segments"]])
    elif "text" in data:
        full_text = data["text"]
    else:
        return {"summary": "Transcript format not recognized."}

    summary = full_text[:700] + "..."
    return {"summary": summary}

# Generate subtitles (VTT) from transcript
@app.post("/generate_cc")
def generate_cc(video_name: str):
    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{video_name}.json")
    vtt_path = os.path.join("data/subtitles", f"{video_name}.vtt")

    if not os.path.exists(transcript_path):
        return {"error": "Transcript not found"}

    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support Whisper format
    if "segments" in data:
        segments = data["segments"]
    # Support chunk format
    elif "chunks" in data:
        segments = data["chunks"]
    else:
        return {"error": "Transcript format not supported"}

    def to_vtt_time(sec):
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h:02}:{m:02}:{s:06.3f}"

    with open(vtt_path, "w", encoding="utf-8") as vtt:
        vtt.write("WEBVTT\n\n")
        for seg in segments:
            start = to_vtt_time(seg["start"])
            end = to_vtt_time(seg["end"])
            text = seg["text"].strip()
            vtt.write(f"{start} --> {end}\n{text}\n\n")

    return {"vtt_path": vtt_path}

# Serve subtitles (VTT) with CORS
@app.get("/subtitles/{filename}")
def get_subtitles(filename: str):
    path = os.path.join("data/subtitles", filename)

    if not os.path.exists(path):
        return {"error": "Subtitle not found"}

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    return Response(
        content,
        media_type="text/vtt",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
            "Access-Control-Allow-Methods": "*",
        }
    )

# Delete video + all related artifacts
@app.delete("/delete_video")
def delete_video(video_name: str):
    registry = load_registry()

    if video_name not in registry:
        return {"error": "Video not found"}

    # Delete video file
    video_path = registry[video_name]["video_path"]
    if video_path and os.path.exists(video_path):
        os.remove(video_path)

    # Delete chunks
    chunks_path = registry[video_name]["chunks"]
    if chunks_path and os.path.exists(chunks_path):
        os.remove(chunks_path)

    # Delete FAISS index
    faiss_path = registry[video_name]["faiss"]
    if faiss_path and os.path.exists(faiss_path):
        os.remove(faiss_path)

    # Delete transcript
    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{video_name}.json")
    if os.path.exists(transcript_path):
        os.remove(transcript_path)

    # Delete subtitles
    vtt_path = os.path.join("data/subtitles", f"{video_name}.vtt")
    if os.path.exists(vtt_path):
        os.remove(vtt_path)

    # Remove from registry
    del registry[video_name]
    save_registry(registry)

    return {"message": "Video deleted"}


@app.post("/search_all")
def search_all(payload: dict):
    query = payload["query"]

    registry = load_registry()
    query_emb = model.encode(query).astype("float32")

    all_results = []

    for video_name, entry in registry.items():
        faiss_path = entry["faiss"]
        meta_path = entry["meta"]

        if not os.path.exists(faiss_path) or not os.path.exists(meta_path):
            continue

        index = faiss.read_index(faiss_path)

        with open(meta_path, "r") as f:
            metadata = json.load(f)

        distances, indices = index.search(np.array([query_emb]), 3)

        for dist, idx in zip(distances[0], indices[0]):
            item = metadata[idx]

            all_results.append({
                "video_name": video_name,
                "text": item["text"],
                "start": item["start"],
                "distance": float(dist)
            })

    # Sort by best match (lowest distance)
    all_results.sort(key=lambda x: x["distance"])

    return {"results": all_results}
