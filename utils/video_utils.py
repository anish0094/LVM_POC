# utils/video_utils.py

import cv2
import os
import pandas as pd
from datetime import datetime
from models.yolo_detector import detect_objects

LOG_FILE = "logs/event_log.csv"

def log_detections(detections):
    if not detections:
        return

    log_entries = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for det in detections:
        log_entries.append({
            "timestamp": timestamp,
            "label": det["label"],
            "confidence": det["confidence"],
            "x1": det["bbox"][0],
            "y1": det["bbox"][1],
            "x2": det["bbox"][2],
            "y2": det["bbox"][3]
        })

    df_new = pd.DataFrame(log_entries)

    if not os.path.exists(LOG_FILE):
        df_new.to_csv(LOG_FILE, index=False)
    else:
        df_existing = pd.read_csv(LOG_FILE)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_csv(LOG_FILE, index=False)

def play_video_stream(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video.")
        return None, []

    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame.")
        cap.release()
        return None, []

    annotated_frame, detections = detect_objects(frame)

    # Log detected objects
    log_detections(detections)

    cap.release()
    return annotated_frame, detections
