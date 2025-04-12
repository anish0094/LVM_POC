# models/yolo_detector.py

from ultralytics import YOLO
import cv2

# Load YOLO model once
model = YOLO("yolov8n.pt")  # You can switch to yolov8s.pt or custom model if needed

def detect_objects(frame):
    """
    Perform object detection on a given frame using YOLOv8.

    Args:
        frame (numpy.ndarray): Input video frame (BGR format)

    Returns:
        tuple: (annotated_frame, detections)
    """
    results = model.predict(source=frame, conf=0.4, stream=False)

    detections = []
    annotated_frame = frame.copy()

    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            # Only process 'person' class
            if label == 'person':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                detections.append({
                    "label": label,
                    "confidence": round(conf, 2),
                    "bbox": [x1, y1, x2, y2]
                })

    return annotated_frame, detections
