import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 환경 변수(.env) 로드
load_dotenv()

app = Flask(__name__)

# 2. Hugging Face 기반 LangChain 대화 함수 정의
def ask_huggingface(question: str) -> str:
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    # 1. 기본 엔드포인트 생성 (task 명시)
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        task="conversational",
        max_new_tokens=512,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    
    # 2. 대화형 모델로 한 번 더 감싸기
    chat_model = ChatHuggingFace(llm=llm)
    
    # 3. 대화형 프롬프트 구조 정의 (System, Human 역할 분담 가능)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 친절한 AI 어시스턴트입니다. 한국어로 답변하세요."),
        ("human", "{question}")
    ])
    
    # 4. 체인 조립 및 실행
    chain = prompt | chat_model | StrOutputParser()
    return chain.invoke({"question": question})

# 3. Flask API 라우트 설정
@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get("message")
    
    if not user_message:
        return jsonify({"error": "메시지가 비어 있습니다."}), 400
    
    try:
        # Hugging Face 오픈소스 모델 실행
        ai_response = ask_huggingface(user_message)
        return jsonify({"response": ai_response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)