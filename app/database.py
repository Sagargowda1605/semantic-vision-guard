import sqlite3
import json
from datetime import datetime
import os

# 1. Define the path for the database file
DB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vision_guard.db")

# 2. Function to initialize the database (Run this when FastAPI starts)
def init_db():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            filename TEXT,
            yolo_class TEXT,
            yolo_conf REAL,
            clip_score REAL,
            status TEXT,
            bbox TEXT
        )
    ''')
    conn.commit()
    conn.close()

# 3. Function to log a detection (Call this after CLIP makes its decision)
def log_detection(filename, yolo_class, yolo_conf, clip_score, status, bbox):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Convert bbox list to a JSON string so SQLite can store it easily
    bbox_str = json.dumps(bbox) 
    
    cursor.execute('''
        INSERT INTO detections (filename, yolo_class, yolo_conf, clip_score, status, bbox)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (filename, yolo_class, yolo_conf, clip_score, status, bbox_str))
    
    conn.commit()
    conn.close()