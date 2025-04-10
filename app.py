from flask import Flask, request, jsonify
import os
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# YOLO 모델 로드 (모델 경로가 로컬 또는 특정 경로에 있다고 가정)
model_path = os.path.join('yolov8_model', 'bestsec.pt')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"YOLO 모델이 존재하지 않습니다: {model_path}")

model = YOLO(model_path)


@app.route('/api/predict', methods=['POST'])
def predict():
    # 1. 이미지 파일이 요청에 포함되어 있는지 확인
    if 'image' not in request.files:
        return jsonify({"error": "No image provided in the request"}), 400

    # 2. 이미지 파일 읽기
    image = request.files['image']

    # 3. 이미지 저장(테스트 목적으로 로컬에 저장)
    save_path = os.path.join('static', image.filename)
    image.save(save_path)

    # 4. YOLO 모델로 이미지 처리
    try:
        results = model(save_path, conf=0.4, iou=0.5)  # YOLO 모델 예측 실행
        results = results[0]  # 첫 번째 이미지 결과
        boxes = results.boxes  # 예측된 bounding boxes
    except Exception as e:
        return jsonify({"error": f"YOLO model failed: {str(e)}"}), 500

    # 5. 예측 결과 처리
    predictions = []
    if boxes is not None and len(boxes.cls) > 0:
        for cls, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            class_name = results.names[int(cls)]  # 클래스 이름
            predictions.append({
                "class": class_name,
                "confidence": round(conf, 2)
            })
    else:
        predictions.append({"message": "No objects detected"})

    # 6. JSON 형태로 결과 반환
    return jsonify({"predictions": predictions})


if __name__ == '__main__':
    app.run(debug=True)
