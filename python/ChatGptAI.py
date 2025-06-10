# ChatGptAI.py
import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from tabulate import tabulate

# 환경변수 불러오기
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "key.env"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# building_aliases.py 또는 ChatGptAI.py 상단에 추가
building_aliases = {
    "산학": "industryHall",
    "산학협동관": "industryHall",
    "산학동": "industryHall",
    "미래관": "Future Hall",
    "미래": "Future Hall",
    "library": "library",
    "도서관": "library"
}



# 건물 설명 context (기존 static 설명)
context = {
    "Future Hall": """미래관: 미래관은 학생들이 자유롭게 공부하고 토론할 수 있는 공간이 있는 건물입니다.
1층에는 카페, 수업 공간, 휴게공간, 다목적 강의실, 스마트 팩토리 1·2·3, 강의실 1, 휴계형 학습공간이 있습니다.
2층에는 도서관, 자습실 1·2, 스터디룸, 미디어 제작실이 있습니다.""",

    "library": """도서관: 도서관은 연암공과대학교 어플에 모바일 신분증으로 사용할 수 있습니다.
연건평 459평으로 종합자료실, 멀티미디어실, 미디어제작실, 스터디룸(1~4), 열람실(1~2)과 장서 62,000여 권, 전공 관련 학술지 및 잡지, 시청각자료를 갖추고 있으며, 완전 개가식으로 운영되고 있습니다.
또한 대출, 반납 등의 모든 업무가 전산화되어 있고, 도서관 홈페이지를 통해 도서 검색이 가능합니다.""",

    "industryHall": """산학 협동관 1층 전산정보센터/취창업 지원센터/ LINC3.0 사업단, 
2층 강의실/교수 연구실, SW 실습실1, SW 합동강의실, 스마트팩토리 미러형 실습실, 
3층 SW 개발실,전자 실습실, 
4층 SW실습실,동아리실(TrippleS,ERPro, Digital Playground,러닝메이트 실, 멘토링실)."""
}

CSV_PATH = "lectures.csv"

def infer_building_key_from_question(question: str) -> str:
    question = question.lower()  # 소문자 처리
    for alias, building_key in building_aliases.items():
        if alias in question:
            return building_key
    return None


def extract_lecture_text(building: str, max_rows: int = 10) -> str:
    try:
        df = pd.read_csv(CSV_PATH)
        filtered = df[df["강의실"].astype(str).str.contains(building, na=False)]

        if filtered.empty:
            return None

        selected = filtered[['교과목명', '교수', '강의시간표', '강의실']].head(max_rows)
        return tabulate(selected, headers='keys', tablefmt='pipe', showindex=False)

    except Exception as e:
        return f"[CSV 처리 오류] {str(e)}"

def get_building_description(building_key, question):
    # 자동 건물 추론 로직 추가
    if not building_key or building_key not in context:
        inferred_key = infer_building_key_from_question(question)
    else:
        inferred_key = building_key

    # 해당 키가 context에도 없고 강의도 없으면 종료
    lecture_text = extract_lecture_text(inferred_key)
    base_context = context.get(inferred_key, "")

    if not base_context and not lecture_text:
        return "죄송합니다. 어떤 건물인지 정확히 알 수 없거나 정보가 없습니다."

    # GPT 프롬프트 생성
    system_prompt = f"""너는 연암공과대학교 건물 및 강의 정보를 안내하는 챗봇이야.
아래는 '{inferred_key}' 건물에 대한 설명과 강의 목록이야:

[건물 설명]
{base_context}

[강의 목록]
{lecture_text or '관련 강의 정보 없음'}

질문에 맞게 자연스럽게 요약해서 답변해줘. 너무 딱딱하게 읽지 말고, 무조건 존댓말로 설명해야 해."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"GPT 오류: {str(e)}"
