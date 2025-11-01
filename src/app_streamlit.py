import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import sqlite3
import pandas as pd
from datetime import datetime
import time
import base64
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

# ==============================
# 🔧 CONFIG
# ==============================
st.set_page_config(
    page_title="VisionGuard - Helmet Detection",
    layout="wide",
    page_icon="🪖",
    initial_sidebar_state="expanded"
)

# ==============================
# 🎨 BACKGROUND IMAGE
# ==============================
def set_bg(image_path):
    if not os.path.exists(image_path):
        return
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(f"""
        <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{encoded}");
                background-size: cover;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
        </style>
    """, unsafe_allow_html=True)

set_bg("assets/bg.jpg")

# Dark theme polish
st.markdown("""
<style>
    h1, h2, h3, h4, h5, h6, p, label, span {
        color: #e2e8f0 !important;
    }
    .stSidebar {
        background-color: rgba(17, 24, 39, 0.9) !important;
    }
    .stButton>button {
        background-color: #2563eb !important;
        color: white !important;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ==============================
# ✅ LOAD MODEL (cached)
# ==============================
@st.cache_resource
def load_model():
    model_path = "models/visionguard_exp2/weights/best.pt"
    return YOLO(model_path)

model = load_model()

# ==============================
# ✅ SQLITE DB
# ==============================
DB_PATH = "visionguard.db"
conn = sqlite3.connect(DB_PATH, check_same_thread=False)
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    image_name TEXT,
    label TEXT,
    confidence REAL
)
""")
conn.commit()

# ==============================
# ✅ SIDEBAR
# ==============================
st.sidebar.title("🔍 Navigate")
page = st.sidebar.radio("Go to", ["Upload Detection", "Webcam Detection", "Dashboard", "About"])


# ================================================================
# ✅ ✅ UPLOAD DETECTION
# ================================================================
if page == "Upload Detection":
    st.title("🪖 Helmet Detection - Upload Image/Video")

    uploaded_file = st.file_uploader(
        "Choose an image or video",
        type=["jpg", "jpeg", "png", "mp4", "mpeg4"]
    )

    if uploaded_file:
        st.info("🔍 Running detection...")

        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        try:
            results = model.predict(temp_path, conf=0.5)
            for r in results:
                annotated = r.plot()
                st.image(annotated, caption="Detection Result", use_container_width=True)

                # Save detections
                for i, box in enumerate(r.boxes):
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    label = model.names[cls_id]

                    # Insert into DB
                    cursor.execute(
                        "INSERT INTO detections (timestamp, image_name, label, confidence) VALUES (?, ?, ?, ?)",
                        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), uploaded_file.name, label, conf)
                    )
                    conn.commit()

                    # Alerts
                    if "Without Helmet" in label:
                        st.error(f"🚨 No Helmet – {conf:.2f}")
                    else:
                        st.success(f"✅ Helmet Worn – {conf:.2f}")

        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            os.remove(temp_path)


# ================================================================
# ✅ ✅ WEBCAM DETECTION WITH STREAMLIT-WEBRTC
# ================================================================
elif page == "Webcam Detection":
    st.title("📹 Real-time Helmet Detection")

    st.info("✅ Works on Render ✅ Works on HTTPS ✅ Mobile/Browser supported")

    class VideoProcessor(VideoProcessorBase):

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # YOLO inference
            results = model.predict(img, conf=0.45, verbose=False)
            annotated = results[0].plot()

            # Log detections
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls)
                    conf = float(box.conf)
                    label = model.names[cls_id]
                    cursor.execute(
                        "INSERT INTO detections (timestamp, image_name, label, confidence) VALUES (?, ?, ?, ?)",
                        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "webcam", label, conf)
                    )
                    conn.commit()

            return av.VideoFrame.from_ndarray(annotated, format="bgr24")

    webrtc_streamer(
        key="helmet-detect",
        video_processor_factory=VideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )


# ================================================================
# ✅ ✅ DASHBOARD
# ================================================================
elif page == "Dashboard":
    st.title("📈 Detection Dashboard")

    df = pd.read_sql_query("SELECT * FROM detections ORDER BY timestamp DESC", conn)

    if df.empty:
        st.info("No detections yet.")
    else:
        st.dataframe(df, use_container_width=True)

        total = len(df)
        helmet = len(df[df.label.str.contains("With Helmet", case=False)])
        no_helmet = len(df[df.label.str.contains("Without Helmet", case=False)])

        c1, c2, c3 = st.columns(3)
        c1.metric("Total Detections", total)
        c2.metric("Helmet Worn", helmet)
        c3.metric("Violations", no_helmet)

        st.subheader("📊 Violation Distribution")
        st.bar_chart(df["label"].value_counts())


# ================================================================
# ✅ ✅ ABOUT
# ================================================================
elif page == "About":
    st.title("🪖 VisionGuard")
    st.subheader("AI Helmet Detection System")

    st.markdown("""
    VisionGuard is an AI-powered safety system that uses **YOLOv8**
    to detect helmet usage in real-time from webcam or uploaded footage.

    **Features:**
    - ✅ Live webcam helmet detection  
    - ✅ Upload images/videos  
    - ✅ Violation alerts  
    - ✅ Dashboard with history  
    - ✅ Render deployment compatible  

    **Developer:** Sanvi Kulkarni  
    """)

