# yolo_service.py
import os
from ultralytics import YOLO

# YOLO 모델 로딩
model_path = os.path.abspath("python/yolov8_model/best.pt")
model = YOLO(model_path)

def detect_building(image_path):
    results = model(image_path, conf=0.4, iou=0.5)[0]
    boxes = results.boxes
    if boxes is None or len(boxes.cls) == 0:
        return None
    return results.names[int(boxes.cls[0])]
