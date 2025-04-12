import streamlit as st
from models.yolo_detector import detect_objects
from nlp.query_handler import handle_query
from utils.video_utils import play_video_stream
import pandas as pd

st.set_page_config(page_title="LVM Proof of Concept", layout="wide")

st.title("🧠 LVM – Real-Time Visual Intelligence")

video_file = "data/sample_video.mp4"

# Sidebar Options
query = st.text_input("Ask a question:", "How many people are in frame?")

# Video + Detection
st.header("Live Video Feed")
frame, detections = play_video_stream(video_file)
if frame is not None:
    st.image(frame, channels="BGR")

# Display detections
st.subheader("Detections")
for det in detections:
    st.write(det)

# NLP Response
st.subheader("Query Answer")
response = handle_query(query, detections)
st.success(response)

# Event Log
st.subheader("Event Log")
log_df = pd.read_csv("logs/event_log.csv")
st.dataframe(log_df)
