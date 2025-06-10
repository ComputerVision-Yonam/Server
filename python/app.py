# app.py
from flask import Flask, request, jsonify
from yolo_service import detect_building
from ChatGptAI import get_building_description
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

building_summaries = {
    "Future Hall": "미래관은 학생들이 자유롭게 토론하고 공부하는 공간입니다.",
    "library": "도서관은 학습과 열람을 위한 시설이 마련된 공간입니다.",
    "industryHall": "산학협동관은 실습과 동아리 활동이 활발히 이루어지는 공간입니다."
}


@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "이미지가 포함되지 않았습니다."}), 400

    image = request.files['image']
    save_path = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(save_path)

    building_key = detect_building(save_path)
    if not building_key:
        return jsonify({"error": "건물을 인식하지 못했습니다."})

    # 요약 설명 가져오기
    summary = building_summaries.get(building_key, "이 건물에 대한 요약 정보가 없습니다.")

    return jsonify({
        "building": building_key,
        "summary": summary
    })


@app.route('/api/chatbot', methods=['POST'])
def chatbot_reply():
    data = request.get_json()
    building_key = data.get("building")
    question = data.get("question")

    if not building_key or not question:
        return jsonify({"error": "building 또는 question 파라미터가 누락되었습니다."}), 400

    response = get_building_description(building_key, question)
    return jsonify({"response": response})

@app.route('/api/chatbot/intro', methods=['GET'])
def chatbot_intro():
    return jsonify({"message": "안녕하세요! 어떤 건물에 대해 궁금하신가요? 😊"})

@app.route('/api/lectures/chatbot', methods=['POST'])
def lecture_question_with_gpt():
    data = request.get_json()
    building = data.get("building")
    question = data.get("question")

    if not building or not question:
        return jsonify({"error": "building과 question을 모두 포함해야 합니다."}), 400

    response = get_lecture_answer_with_gpt(building, question)
    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


