# yolo_service.py
import os
from ultralytics import YOLO

# YOLO 모델 로딩

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "yolov8_model", "best.pt")
model = YOLO(model_path)

def detect_building(image_path):
    results = model(image_path, conf=0.4, iou=0.5)[0]
    boxes = results.boxes
    if boxes is None or len(boxes.cls) == 0:
        return None
    return results.names[int(boxes.cls[0])]
