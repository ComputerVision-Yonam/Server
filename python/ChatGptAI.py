import os
from openai import OpenAI
from dotenv import load_dotenv

# 환경변수 불러오기 (.env 또는 key.env)
load_dotenv(dotenv_path="key.env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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

def get_building_description(building_key, question):
    if building_key not in context:
        return "죄송합니다. 해당 건물에 대한 정보가 없습니다."

    system_prompt = f"""너는 연암공과대학교 건물 정보를 안내하는 챗봇이야. 너무 딱딱하게 Context를 다 보내지는 말고 요령껏 유연하게 설명해줘, 그리고 무조건 존댓말로 설명해야해
다음은 {building_key}에 대한 설명이야:
{context[building_key]} 
"""

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