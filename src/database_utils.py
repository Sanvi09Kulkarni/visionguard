import sqlite3
from datetime import datetime

# Create database connection
def get_connection():
    conn = sqlite3.connect("visionguard.db", check_same_thread=False)
    return conn

# Create detections table (if not exists)
def create_table():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            image_path TEXT,
            label TEXT,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

# Insert detection record
def insert_detection(image_path, label, confidence):
    conn = get_connection()
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("""
        INSERT INTO detections (timestamp, image_path, label, confidence)
        VALUES (?, ?, ?, ?)
    """, (timestamp, image_path, label, confidence))
    conn.commit()
    conn.close()

# Fetch all detections
def fetch_all_detections():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM detections ORDER BY id DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows
