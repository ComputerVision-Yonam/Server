from flask import Flask, request, jsonify
import os
from PIL import Image
from ultralytics import YOLO
from waitress import serve
app = Flask(__name__)

# YOLO 모델 로드 (모델 경로가 로컬 또는 특정 경로에 있다고 가정)
model_path = os.path.join('yolov8_model', 'bestsec.pt')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"YOLO 모델이 존재하지 않습니다: {model_path}")

model = YOLO(model_path)

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided in the request"}), 400

    image = request.files['image']

    # ✅ static 디렉토리가 없으면 생성
    os.makedirs('static', exist_ok=True)

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
    # app.run(debug=True)
    serve(app, host='0.0.0.0', port=5000)