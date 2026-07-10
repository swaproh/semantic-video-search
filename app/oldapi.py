from openai import OpenAI

#Overview and Summary 
client = OpenAI()

@app.get("/summary_topics/{video_name}")
def generate_topic_summary(video_name: str):
    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{video_name}.json")

    if not os.path.exists(transcript_path):
        return {"summary": "Transcript not found."}

    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract full transcript text
    if "chunks" in data:
        full_text = " ".join([c["text"] for c in data["chunks"]])
    elif "segments" in data:
        full_text = " ".join([s["text"] for s in data["segments"]])
    else:
        full_text = data.get("text", "")

    # LLM Topic-wise summary
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Generate a topic-wise summary. "
                    "Break the transcript into clear sections. "
                    "Each section must have a topic title and 3–5 bullet points."
                )
            },
            {"role": "user", "content": full_text}
        ]
    )

    summary_text = response.choices[0].message.content
    return {"summary": summary_text}


@app.get("/overview/{video_name}")
def generate_overview(video_name: str):
    transcript_path = os.path.join(TRANSCRIPTS_DIR, f"{video_name}.json")

    if not os.path.exists(transcript_path):
        return {"overview": "Transcript not found."}

    with open(transcript_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "chunks" in data:
        full_text = " ".join([c["text"] for c in data["chunks"]])
    elif "segments" in data:
        full_text = " ".join([s["text"] for s in data["segments"]])
    else:
        full_text = data.get("text", "")

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Give a short overview of the transcript."},
            {"role": "user", "content": full_text}
        ]
    )

    overview_text = response.choices[0].message.content
    return {"overview": overview_text}