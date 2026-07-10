import streamlit as st
import requests
import os

API = "http://localhost:8000"

st.set_page_config(layout="wide")
st.title("Semantic Video Search")
# ---------------------------------------------------
# Load existing videos
# ---------------------------------------------------
try:
    existing_videos = requests.get(f"{API}/videos").json().get("videos", [])
except Exception:
    existing_videos = []

# ---------------------------------------------------
# Upload or select video
# ---------------------------------------------------
st.subheader("Upload or Select Video")

col_upload, col_select = st.columns([0.5, 0.5])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload a video",
        type=["mp4", "mov", "mkv", "mpeg4"]
    )

with col_select:
    selected_video = st.selectbox(
        "Select from existing videos",
        [""] + existing_videos
    )

# ---------------------------------------------------
# Handle upload or selection
# ---------------------------------------------------
if uploaded_file:
    video_name = os.path.splitext(uploaded_file.name)[0]
    st.session_state["video_name"] = video_name

    save_path = f"data/raw_video/{uploaded_file.name}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"Video uploaded: {uploaded_file.name}")

    # Transcribe
    with st.spinner("Transcribing video..."):
        resp = requests.post(f"{API}/transcribe", params={"video_path": save_path})
        transcript_path = resp.json()["transcript_path"]

    # Build search pipeline
    with st.spinner("Building search index..."):
        requests.post(f"{API}/build", params={"transcript_path": transcript_path})

    # Generate subtitles
    with st.spinner("Generating subtitles..."):
        requests.post(f"{API}/generate_cc", params={"video_name": video_name})

    st.session_state["video_path"] = f"http://localhost:8000/video/{video_name}.mp4"
    st.session_state["vtt_path"] = f"http://localhost:8000/subtitles/{video_name}.vtt"

elif selected_video:
    video_name = selected_video
    st.session_state["video_name"] = video_name

    st.session_state["video_path"] = f"http://localhost:8000/video/{video_name}.mp4"
    st.session_state["vtt_path"] = f"http://localhost:8000/subtitles/{video_name}.vtt"

    st.success(f"Loaded existing video: {video_name}")

jump_time = st.session_state.get("jump_to", 0)

# ---------------------------------------------------
# Video player
# ---------------------------------------------------
if "video_path" in st.session_state:
    st.subheader("Video Player")

    # Delete video button
    if st.button("🗑️ Delete Video"):
        with st.spinner("Deleting video..."):
            requests.delete(
                f"{API}/delete_video",
                params={"video_name": st.session_state["video_name"]}
            )
        st.success("Video deleted.")
        st.session_state.clear()
        st.rerun()

    # Native Streamlit video player
    #st.video(st.session_state["video_path"])

    
    start_time = jump_time   # seconds
    st.write(f"Starting video at {start_time}s ...")
    html_code = f"""
    <video id="myvideo" width="600" controls>
        <source src="{st.session_state["video_path"]}" type="video/mp4">
    </video>

    <script>
        const vid = document.getElementById('myvideo');
        vid.onloadeddata = function() {{
            vid.currentTime = {start_time};
            vid.play();
        }};
    </script>
    """

    st.components.v1.html(html_code, height=400)



    def format_timestamp(seconds: float):
        seconds = int(seconds)
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60

        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        else:
            return f"{m:02d}:{s:02d}"

  # Tools Section
# ---------------------------------------------------
st.subheader("Tools")

# Initialize session state for summaries
if "overview" not in st.session_state:
    st.session_state["overview"] = None

if "topic_summary" not in st.session_state:
    st.session_state["topic_summary"] = None

col_left, col_right = st.columns([0.5, 0.5])

# ---------------- OVERVIEW ----------------
with col_left:
    st.markdown("### 📘 Overview")

    if st.button("Generate Overview"):
        with st.spinner("Generating overview..."):
            resp = requests.get(
                f"{API}/overview/{st.session_state['video_name']}"
            ).json()

        st.session_state["overview"] = resp.get("overview", "No overview available.")

    # Display Overview
    if st.session_state["overview"]:
        st.markdown("**Overview:**")
        st.write(st.session_state["overview"])
        
        # Download Overview
        st.download_button(
            label="⬇️ Download Overview",
            data=st.session_state["overview"],
            file_name=f"{st.session_state['video_name']}_overview.txt",
            mime="text/plain"
        )

    st.markdown("---")

    # ---------------- TOPIC-WISE SUMMARY ----------------
    st.markdown("### 🧩 Topic-Wise Summary")

    if st.button("Generate Topic-Wise Summary"):
        with st.spinner("Generating topic‑wise summary..."):
            resp = requests.get(
                f"{API}/summary_topics/{st.session_state['video_name']}"
            ).json()

        st.session_state["topic_summary"] = resp.get("summary", "No topic‑wise summary available.")

    # Display Topic Summary
    if st.session_state["topic_summary"]:
        st.markdown("**Topic-Wise Summary:**")
        st.write(st.session_state["topic_summary"])

        # Download Topic Summary
        st.download_button(
            label="⬇️ Download Topic-Wise Summary",
            data=st.session_state["topic_summary"],
            file_name=f"{st.session_state['video_name']}_topic_summary.txt",
            mime="text/plain"
        )


# ---------------- SEARCH ----------------
with col_right:
    st.markdown("### 🔍 Search inside video")
    query = st.text_input("Type a topic to search")

    if st.button("Search"):
        if query.strip():
            with st.spinner("Searching..."):
                resp = requests.post(
                    f"{API}/search",
                    json={
                        "video_name": st.session_state["video_name"],
                        "query": query
                    }
                ).json()

            st.session_state["search_results"] = resp.get("results", [])
        else:
            st.warning("Please enter a search query.")

    # Show results
    if "search_results" in st.session_state:
        results = st.session_state["search_results"]
        video_name = st.session_state["video_name"]

        if results:
            st.markdown(f"### Results for **{video_name}** (click to jump)")

            for i, r in enumerate(results):
                text = r["text"]
                start_time = r["start"]
                ts = format_timestamp(start_time)

                label = f"{text}\n🕒 **{ts}**"

                if st.button(label, key=f"result_{i}"):
                    st.session_state["jump_to"] = start_time
        else:
            st.info("No matching segment found.")

    # Perform jump
    if "jump_to" in st.session_state:
        jump_time = st.session_state["jump_to"]

