import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
import os
import sqlite3
import pandas as pd
from datetime import datetime
import time
import base64  # <-- added for local sound embedding

# ==============================
# 🧠 CONFIGURATION
# ==============================
st.set_page_config(
    page_title="VisionGuard - Helmet Detection",
    layout="wide",
    page_icon="🪖",
    initial_sidebar_state="expanded"
)

# Custom dark theme CSS
st.markdown("""
    <style>
        body {
            background-color: #0e1117;
            color: #f5f6fa;
        }
        .stApp {
            background: linear-gradient(135deg, #0e1117, #1e293b);
            color: white;
        }
        .css-18e3th9, .css-1d391kg {
            background-color: transparent !important;
        }
        .stSidebar {
            background-color: #111827 !important;
        }
        h1, h2, h3, h4, h5, h6, p {
            color: #e2e8f0 !important;
        }
        .sidebar .sidebar-content {
            background-color: #111827;
        }
        .stButton>button {
            background-color: #2563eb !important;
            color: white !important;
            border-radius: 8px;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================
# 📦 LOAD MODEL
# ==============================
model_path = "models/visionguard_exp2/weights/best.pt"
model = YOLO(model_path)

# ==============================
# 🗃️ DATABASE SETUP
# ==============================
DB_PATH = "visionguard.db"
conn = sqlite3.connect(DB_PATH)
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
# 🧭 SIDEBAR NAVIGATION
# ==============================
st.sidebar.title("🔍 Navigate")
page = st.sidebar.radio("Go to", ["Upload Detection", "Webcam Detection", "Dashboard", "About"])

# ==============================
# 📸 UPLOAD DETECTION PAGE
# ==============================
if page == "Upload Detection":
    st.title("🪖 VisionGuard")
    st.subheader("Smart Helmet Detection & Violation Alerts")

    uploaded_file = st.file_uploader(
        "Choose an image or video",
        type=["jpg", "jpeg", "png", "mp4", "mpeg4"]
    )

    if uploaded_file is not None:
        st.info("🔍 Running detection... please wait")
        
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_path = tmp_file.name

        try:
            results = model.predict(temp_path, conf=0.5)
            for r in results:
                result_img = r.plot()
                st.image(result_img, caption="Detection Result", use_container_width=True)

                # Detection Details
                st.subheader("📊 Detection Details")
                boxes = r.boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    conf = float(boxes.conf[i])
                    label = model.names[cls_id]
                    st.write(f"**{label}** – Confidence: `{conf:.2f}`")

                    # Save to DB
                    cursor.execute("INSERT INTO detections (timestamp, image_name, label, confidence) VALUES (?, ?, ?, ?)",
                                   (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), uploaded_file.name, label, conf))
                    conn.commit()

                    # 🚨 Alert for violation
                    if "Without Helmet" in label:
                        st.error("🚨 Violation Detected: No Helmet!")

                        # Try playing local beep.mp3
                        try:
                            beep_path = "src/beep.mp3"
                            if os.path.exists(beep_path):
                                with open(beep_path, "rb") as audio_file:
                                    audio_bytes = audio_file.read()
                                st.markdown(
                                    f"""
                                    <audio autoplay>
                                        <source src="data:audio/mp3;base64,{base64.b64encode(audio_bytes).decode()}" type="audio/mp3">
                                    </audio>
                                    """,
                                    unsafe_allow_html=True
                                )
                            else:
                                # fallback online beep
                                st.markdown(
                                    '<audio autoplay><source src="https://actions.google.com/sounds/v1/alarms/beep_short.ogg" type="audio/ogg"></audio>',
                                    unsafe_allow_html=True
                                )
                        except Exception as sound_err:
                            st.warning(f"⚠️ Sound could not play: {sound_err}")

                    else:
                        st.success("✅ Helmet Worn. Safe to ride!")
        except Exception as e:
            st.error(f"⚠️ Error during detection: {e}")
        finally:
            os.remove(temp_path)

# ==============================
# 🎥 WEBCAM DETECTION PAGE
# ==============================
elif page == "Webcam Detection":
    st.title("📹 Real-time Helmet Detection via Webcam")
    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Unable to access webcam.")
            break

        results = model.predict(frame)
        annotated = results[0].plot()
        FRAME_WINDOW.image(annotated, channels="BGR")

        # Save detections periodically
        for r in results:
            boxes = r.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                label = model.names[cls_id]
                cursor.execute("INSERT INTO detections (timestamp, image_name, label, confidence) VALUES (?, ?, ?, ?)",
                               (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "webcam_frame", label, conf))
                conn.commit()

        time.sleep(0.05)
    else:
        camera.release()

# ==============================
# 📊 DASHBOARD PAGE
# ==============================
elif page == "Dashboard":
    st.title("📈 Detection Dashboard")

    df = pd.read_sql_query("SELECT * FROM detections ORDER BY timestamp DESC", conn)

    if df.empty:
        st.info("No detection data yet.")
    else:
        st.dataframe(df)

        total = len(df)
        helmet = len(df[df['label'].str.contains("With Helmet", case=False)])
        no_helmet = len(df[df['label'].str.contains("Without Helmet", case=False)])

        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Detections", total)
        col2.metric("Helmet Worn", helmet)
        col3.metric("Violations", no_helmet)

        st.bar_chart(df['label'].value_counts())

# ==============================
# ℹ️ ABOUT PAGE
# ==============================
elif page == "About":
    st.title("🪖 VisionGuard")
    st.subheader("Smart Helmet Detection & Violation Alerts")

    st.markdown("""
    **VisionGuard** is an AI-powered computer vision system that detects helmet usage in real-time using YOLOv8 models.

    ### 🚀 Features:
    - Real-time image & webcam detection  
    - Detection dashboard with live logging  
    - Instant alerts for *No-Helmet* violations  
    - SQLite-based data tracking  
    - Sleek dark mode interface

    ---
    👩‍💻 **Developed by:** *Sanvi Kulkarni*  
    🧠 **Powered by:** *Ultralytics YOLOv8 + Streamlit*
    """)
