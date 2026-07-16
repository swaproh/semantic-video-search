# semantic-video-search

Overview:
The Semantic Video Search System is an AI‑powered application that allows users to upload any video and instantly search inside it using natural language.
The system automatically transcribes the video, builds semantic embeddings, and enables fast, meaningful search inside video content.
Users can click a search result to jump directly to the exact timestamp in the video.
It also generates an overview and topic‑wise summary using LLMs.

Features:
Upload any video

Automatic transcription using Whisper

Segment extraction with timestamps

Semantic embeddings using MiniLM

Fast similarity search using FAISS

Jump‑to‑timestamp video navigation

LLM‑powered overview

LLM‑powered topic‑wise summary

Clean Streamlit UI

FastAPI backend



Architecture:
Pipeline
Video Upload

Whisper Transcription

Segment Extraction

MiniLM Embeddings

FAISS Index Creation

Semantic Search

Jump to Timestamp

LLM Overview & Summaries



Tech Stack:
Backend
FastAPI

Whisper (OpenAI)

SentenceTransformer (MiniLM)

FAISS

OpenAI GPT‑4o‑mini

Frontend
Streamlit

HTML5 video player

JavaScript timestamp control

Storage
Local filesystem

JSON registry

FAISS index files