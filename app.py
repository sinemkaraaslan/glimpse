import streamlit as st
import face_recognition
import cv2
import tempfile
import os
import numpy as np
import pandas as pd
import altair as alt
from PIL import Image

# Page config
st.set_page_config(
    page_title="Glimpse",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .stApp {
            background-color: #151520;
        }

        [data-testid="stSidebar"] {
            background-color: #1E1E2C;
            border-right: 1px solid #FF80AB;
        }

        h1, h2, h3 {
            color: #FF80AB !important;
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
        }

        .stMarkdown, p, label, div, span {
            color: #E0E0E0 !important;
        }

        .stButton>button {
            color: #FF80AB !important;
            border: 1px solid #FF80AB;
            border-radius: 12px;
            background: transparent;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background-color: #FF80AB;
            color: #151520 !important;
            font-weight: bold;
        }

        [data-testid="stFileUploader"] {
            background-color: #1E1E2C;
            border: 1px dashed #FF80AB;
            border-radius: 10px;
            padding: 10px;
        }

        .stSuccess {
            background-color: #1E1E2C;
            color: #FF80AB;
            border: none;
            border-left: 4px solid #FF80AB;
        }
        
        .stInfo {
            background-color: #1E1E2C;
            color: #FF80AB;
            border: none;
            border-left: 4px solid #00E5FF;
        }

        .stWarning {
            background-color: #1E1E2C;
            color: #FF80AB;
            border: none;
            border-left: 4px solid #FFD740;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        
    </style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### ⚙️ Settings")
uploaded_photo = st.sidebar.file_uploader("Reference Face", type=['jpg', 'png', 'jpeg', 'webp'])
uploaded_video = st.sidebar.file_uploader("Video to Analyze", type=['mp4', 'mov', 'avi'])
st.sidebar.divider()
tolerance = st.sidebar.slider("Tolerance", 0.4, 0.8, 0.6, 0.05)

# Header
st.title("Glimpse 🌸")
st.markdown("**Status:** `Active` | **Mode:** `Deep Biometric Analysis`")
st.divider()

if uploaded_photo and uploaded_video:
    # Load reference face
    try:
        pil_image = Image.open(uploaded_photo).convert('RGB')
        ref_array = np.array(pil_image)
        ref_encoding = face_recognition.face_encodings(ref_array)[0]
        st.sidebar.success("Target Locked 🔒")
        st.sidebar.image(pil_image, caption="Target", use_container_width=True)
    except:
        st.error("Error: No face found in reference photo.")
        st.stop()

    # Save video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    scene_buffer = []
    scene_timestamps = []
    scene_count = 0
    all_face_distances = []
    scene_best_score = 0.0

    results_area = st.container()
    progress_bar = st.progress(0)
    status_text = st.empty()
    frame_idx = 0
    
    # Main loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        
        # Skip frames for performance
        if frame_idx % 15 != 0:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locs = face_recognition.face_locations(rgb_frame)
        face_encs = face_recognition.face_encodings(rgb_frame, face_locs)

        match_found = False
        current_frame_confidence = 0

        for face_enc, face_loc in zip(face_encs, face_locs):
            face_dist = face_recognition.face_distance([ref_encoding], face_enc)[0]
            all_face_distances.append(face_dist)
            
            if face_dist <= tolerance:
                match_found = True
                
                # Boost score using sqrt for better visual feedback
                raw_similarity = 1.0 - face_dist
                boosted_score = (raw_similarity ** 0.5) * 100
                boosted_score = min(max(boosted_score, 0), 99)
                
                if boosted_score > current_frame_confidence:
                    current_frame_confidence = boosted_score

                # Draw rectangle
                y1, x2, y2, x1 = face_loc
                cv2.rectangle(rgb_frame, (x1, y1), (x2, y2), (171, 128, 255), 4)

        current_time = frame_idx / fps

        # Scene grouping
        if match_found:
            scene_buffer.append(rgb_frame)
            scene_timestamps.append(current_time)
            if current_frame_confidence > scene_best_score:
                scene_best_score = current_frame_confidence
        else:
            # End of scene, display results
            if len(scene_timestamps) > 0:
                start_t = scene_timestamps[0]
                end_t = scene_timestamps[-1]
                scene_count += 1
                
                with results_area:
                    with st.container():
                        c1, c2 = st.columns([1, 3])
                        with c1:
                            mid_i = len(scene_buffer) // 2
                            st.image(scene_buffer[mid_i], use_container_width=True)
                        with c2:
                            st.subheader(f"Match #{scene_count}")
                            st.info(f"⏱️ **Time:** {start_t:.1f}s - {end_t:.1f}s  |  🌟 **Confidence:** {scene_best_score:.1f}%")
                        st.divider()
                
                # Reset for next scene
                scene_buffer = []
                scene_timestamps = []
                scene_best_score = 0.0

        status_text.markdown(f"<span style='color:#FF80AB'>Analyzing... {int((frame_idx / total_frames) * 100)}%</span>", unsafe_allow_html=True)
        progress_bar.progress(min(int((frame_idx / total_frames) * 100), 100))

    # Handle remaining scene at video end
    if len(scene_timestamps) > 0:
        scene_count += 1
        with results_area:
            with st.container():
                c1, c2 = st.columns([1, 3])
                with c1:
                    mid_i = len(scene_buffer) // 2
                    st.image(scene_buffer[mid_i], use_container_width=True)
                with c2:
                    st.subheader(f"Match #{scene_count}")
                    st.info(f"⏱️ **Time:** {scene_timestamps[0]:.1f}s - {scene_timestamps[-1]:.1f}s  |  🌟 **Confidence:** {scene_best_score:.1f}%")

    progress_bar.empty()
    status_text.empty()
    
    # Sensitivity chart
    if len(all_face_distances) > 0:
        st.divider()
        st.subheader("📊 Sensitivity Curve")
        
        thresholds = np.arange(0.3, 0.9, 0.05)
        match_counts = [sum(1 for d in all_face_distances if d <= t) for t in thresholds]
            
        chart_data = pd.DataFrame({'Threshold': thresholds, 'Detections': match_counts})
        
        chart = alt.Chart(chart_data).mark_line(point=True, color='#FF80AB').encode(
            x=alt.X('Threshold', title='Tolerance Threshold'),
            y=alt.Y('Detections', title='Face Detections'),
            tooltip=['Threshold', 'Detections']
        ).properties(height=300, background='#1E1E2C').configure_axis(
            labelColor='#E0E0E0', titleColor='#FF80AB'
        ).configure_title(color='#FF80AB')
        
        st.altair_chart(chart, use_container_width=True)
        st.caption("Shows how many faces would be detected at different tolerance levels.")

    if scene_count > 0:
        st.success(f"Done. Found {scene_count} match(es).")
        st.balloons()
    else:
        st.warning("No matches found.")

    os.remove(video_path)

else:
    st.info("Upload files from the sidebar to start...")