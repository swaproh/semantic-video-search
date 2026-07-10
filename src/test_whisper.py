import whisper

model = whisper.load_model("small")
result = model.transcribe("data/raw_video/sample.mp4")

print(result["text"])
