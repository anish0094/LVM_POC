# nlp/query_handler.py

import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def handle_query(query, detections):
    """
    Handles queries using GPT by giving it the detection list and query.
    """
    if not detections:
        return "No objects detected in the current frame."

    # Convert detections to a human-readable summary
    detected_summary = []
    for d in detections:
        detected_summary.append(
            f"{d['label']} (conf: {d['confidence']}) at position {d['bbox']}"
        )

    detection_text = "\n".join(detected_summary)

    prompt = f"""
You are a visual analytics assistant. You are given object detections from a CCTV video frame.
The user is asking a question based on these detections.

Detections:
{detection_text}

Question: {query}
Answer:
    """

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=100,
        )

        return response.choices[0].message["content"].strip()

    except Exception as e:
        return f"Error querying LLM: {e}"
