from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import os
from PIL import Image, ImageEnhance

app = Flask(__name__)

# YOLOv8 모델 로드 (훈련된 모델 가중치 파일 경로)
model_path = os.path.join('yolov8_model', 'best.pt')  # 상대 경로 사용
if not os.path.exists(model_path):
    raise FileNotFoundError(f"YOLO 모델 파일이 존재하지 않습니다: {model_path}")

model = YOLO(model_path)  # 모델 로드


@app.route('/')
def home():
    return render_template('index.html')  # 이미지를 업로드하는 HTML 폼을 렌더링


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    # 이미지 파일 받기
    image = request.files['image']
    image_path = os.path.join('static', image.filename)

    # 전처리: 이미지 대비 조정을 통해 모델 감지 성능 향상
    try:
        img = Image.open(image)
        enhancer = ImageEnhance.Contrast(img)
        enhanced_img = enhancer.enhance(1.5)  # 명암비 조정
        enhanced_img.save(image_path)
    except Exception as e:
        print(f"Image preprocessing failed: {e}")
        return jsonify({"error": "Image preprocessing failed"}), 500

    # YOLO 모델 예측
    results = model(image_path, conf=0.1, iou=0.2)  # 신뢰도 & IoU 조정
    results = results[0]  # 첫 번째 이미지의 결과로 한정
    boxes = results.boxes  # 예측된 bounding boxes

    # 디버깅: RAW 예측 데이터 확인
    print("Raw Results:", results)
    print("Boxes:", boxes)
    if boxes is not None:
        print("Classes detected (cls):", boxes.cls.tolist() if boxes.cls is not None and len(boxes.cls) > 0 else [])
        print("Confidences:", boxes.conf.tolist() if boxes.conf is not None and len(boxes.conf) > 0 else [])
    else:
        print("No detections found.")

    # 예측 결과 처리
    prediction_text = ""
    if boxes is not None and boxes.cls is not None and len(boxes.cls) > 0:  # 감지된 객체가 있는 경우
        for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            class_name = results.names[int(cls_id)]  # 클래스 이름 가져오기
            prediction_text += f"Class: {class_name}, Confidence: {conf:.2f}\n"
    else:
        prediction_text = "No objects detected."

    return jsonify({"prediction": prediction_text})


if __name__ == '__main__':
    app.run(debug=True)
