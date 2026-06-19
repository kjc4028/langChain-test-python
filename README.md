### 소개
Hugging Face의 오픈소스 LLM 모델(Meta Llama-3)과 LangChain을 결합하여, 가볍고 빠르게 구동 가능한 Flask 기반의 AI 챗봇 API 서버 프로젝트입니다. 

본 프로젝트는 OpenAI의 유료 API 대안으로, 허깅페이스의 오픈소스 생태계를 활용하여 무료로 고성능 대화형 인공지능을 구현하는 것을 목표로 합니다.


### 가상환경 생성 및 활성화
프로젝트 패키지 충돌을 방지하기 위해 독립된 가상환경을 구축합니다.

-Windows
python -m venv .venv
.venv\Scripts\activate

-macOS / Linux
python -m venv .venv
source .venv/bin/activate

### 필수 패키지 설치
pip install -r requirements.txt

### 환경 변수 설정
.env
HUGGINGFACEHUB_API_TOKEN=hf_your_actual_token_here

### 실행 방법
python app.py

---

### 기본 API 테스트
Endpoint
URL: POST http://127.0.0.1:5000/chat
Content-Type: application/json

### request
{"message": "지구에서 가장 높은 산은 뭐야?"}

### response
{
    "response": "😊\n\n지구에서 가장 높은 산은मकалу누(Lhotse, 8,848m)입니다. Макалуну(Mount Everest)는 네팔과 중국의 국경에 있는 산으로, 8,848m의 높이로 지구의 최고점을 차지합니다."
}

<img width="1277" height="391" alt="image" src="https://github.com/user-attachments/assets/69b4b48f-9180-46fa-8724-a0b17576f989" />

---
### OCR 기능 추가
OCR 엔진 테서랙트(Tesseract)를 기반으로 메뉴 수량을 손글씨로 작성하는 메뉴 주문표에 대한 OCR 추출 기능

### OCR 결과
OCR 요청한 메뉴주문표와 OCR 결과

<img width="1073" height="817" alt="결과01" src="https://github.com/user-attachments/assets/75801f8c-44f0-4f5c-9a1b-0ad954c60bfb" />

OCR 추출텍스트데이터

<img width="561" height="185" alt="orc추출텍스트데이터" src="https://github.com/user-attachments/assets/a7397acd-4fa3-455c-85c7-5185c5fb6d87" />

###문제해결
- 표(Table) 형태의 메뉴주문표 이미지 인식 시 발생하는 문자 오추출 문제를 해결하여 시스템 파이프라인의 데이터 정밀도 확보
  - 문제 현상: 표 형식의 이미지에 대해 OCR(광학 문자 인식) 엔진을 직접 적용할 때, 수치 및 텍스트 데이터 사이에 특수문자 또는 온점(예: |, .)이 무작위로 혼입되는 오인 추출 현상 발생
  - 개선 대책: 오픈소스 컴퓨터 비전 라이브러리(OpenCV)를 파이프라인 전단에 도입하여 엔진 입력 전 이미지 그래픽 최적화 단계를 추가


