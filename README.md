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

### API 테스트
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
