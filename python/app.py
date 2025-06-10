# app.py
from flask import Flask, request, jsonify , session
from yolo_service import detect_building

#from ChatGptAI import  load_lecture_chunks, answer_from_gpt
from ChatGptAI import get_general_response,get_lecture_answer_with_gpt
import os

app = Flask(__name__)
app.secret_key = "ABCD"
UPLOAD_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#lecture_chunks = load_lecture_chunks()  # 서버 시작 시 한 번만 로딩

# 대화 세션 상태 저장 (간단한 임시 메모리)
#session_messages = []


building_summaries = {
    "Future Hall": "미래관은 학생들이 자유롭게 토론하고 공부하는 공간입니다.",
    "library": "도서관은 학습과 열람을 위한 시설이 마련된 공간입니다.",
    "industryHall": "산학협동관 은 실습과 동아리 활동이 활발히 이루어지는 공간입니다."
}


def extract_building_key(question: str) -> str | None:
    mapping = {
        "산학": "industryHall",
        "산학협동관": "industryHall",
        "도서관": "library",
        "미래관": "Future Hall",
        "future hall": "Future Hall",
        "futurehall": "Future Hall"
    }
    for keyword, key in mapping.items():
        if keyword.lower() in question.lower():
            return key
    return None


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




# @app.route("/api/chatbot", methods=["POST"])
# def chatbot():
#     data = request.get_json()
#     user_msg = data.get("question", "")
#     if not user_msg:
#         return jsonify({"error": "질문이 비어 있습니다."}), 400
#
#     # 메시지 누적
#     session_messages.append({"role": "user", "content": user_msg})
#     print("session_message",session_messages)
#     # GPT 호출
#     answer = answer_from_gpt(session_messages, lecture_chunks)
#     print("answer",answer)
#     session_messages.append({"role": "assistant", "content": answer})
#
#     return jsonify({"answer": answer})







@app.route('/api/chatbot', methods=['POST'])
def chatbot_reply():
    data = request.get_json()
    question = data.get("question")

    if not question:
        return jsonify({"error": "질문이 없습니다."}), 400

    # 1. 강의 키워드 감지
    lecture_keywords = ["강의", "강좌", "수업"]
    contains_lecture = any(k in question for k in lecture_keywords)

    # 2. 건물 인식 시도
    building_key = extract_building_key(question)

    # 3. 인식 실패 시 이전 건물 참조
    if not building_key:
        building_key = session.get("last_building")

    # 4. 인식된 건물 저장
    if building_key:
        session["last_building"] = building_key

    # 5. 응답 처리
    if contains_lecture and building_key:
        response = get_lecture_answer_with_gpt(building_key, question)
    else:
        response = get_general_response(question)

    return jsonify({"answer": response})




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


