import os
import cv2
import numpy as np

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import JsonOutputParser
from PIL import Image
from pydantic import BaseModel, Field
from io import BytesIO

# LLM이 최종적으로 응답할 JSON 구조를 정의합니다.
class ReceiptInfo(BaseModel):
    menu_name: str = Field(description="메뉴명 (예: OOO김밥, OO라면, OO비빔밥, OO볶음밥)")
    menu_cnt: str = Field(description="수량 (예: 1, 2, 20 숫자형 문자)")
    menu_price: str = Field(description="금액 (예: 1,000, 2,500 금액으로 추출시 ,는 제외 처리)")
    
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

# 2. LangChain 파이프라인 초기화 (서버 가동 시 1회만 생성)
def init_langchain_chain():
    repo_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    # Endpoint 단에서는 호환성을 위해 속성을 최소화합니다.
    llm = HuggingFaceEndpoint(
        repo_id=repo_id,
        max_new_tokens=512,
        temperature=0.1,
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    
    # 👈 ChatHuggingFace 인터페이스가 내부적으로 'conversational' 태스크 규격을 
    # 허깅페이스 API가 원하는 패킷 형태로 강제 변환하여 전달합니다.
    chat_model = ChatHuggingFace(llm=llm)
    
    parser = JsonOutputParser(pydantic_object=ReceiptInfo)
    
    # 대화형 프롬프트 형태로 레이아웃 변경
    prompt = ChatPromptTemplate.from_messages([
        ("system",         "당신은 메뉴주문 분석 전문가입니다. 제공된 메뉴주문표 텍스트에서 필요한 정보를 정확히 추출하세요.\n"+
        "반드시 아래의 JSON 포맷 형식만을 준수하여 답변해야 하며, 추가적인 설명이나 인사말은 절대 포함하지 마세요.\n\n"+
        "메뉴명, 수량, 금액 형식이다. 숫자값은 ,(콤마)를 제거한 숫자만 추출한다.\n"+
        "포맷 지시사항:\n{format_instructions}\n\n"+
        "영수증 텍스트:\n{receipt_text}\n\n"+
        "결과 JSON:"),
        ("human", "영수증 텍스트:\n{receipt_text}")
    ])
    
    prompt_with_instructions = prompt.partial(format_instructions=parser.get_format_instructions())
    
    # 체인 구성에 llm 대신 chat_model을 꽂아줍니다.
    chain = prompt_with_instructions | chat_model | parser
    return chain

# 체인 객체 생성
receipt_chain = init_langchain_chain()

def remove_table_lines(file_stream):
    """
    업로드된 이미지 스트림을 받아 표의 선과 자잘한 점 노이즈를 제거하고,
    란초스 확대 및 샤프닝 필터를 적용한 Pillow 이미지 객체를 반환합니다.
    """
    # 1. 파일 스트림 포인터 리셋 및 OpenCV 로드
    file_stream.seek(0)
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    file_stream.seek(0)  # 다음 사용을 위해 커서 리셋
    
    # 2. 화질 저하 방지를 위해 고성능 란초스(Lanczos) 보간법으로 먼저 2배 확대
    img_resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 3. 이진화 처리 (Otsu 임계값 자동 제어)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # 4. 모폴로지 연산으로 가로/세로 표 선 탐지 및 제거
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    table_lines = cv2.add(detect_horizontal, detect_vertical)
    result_without_lines = cv2.subtract(thresh, table_lines)
    
    # -------------------------------------------------------------
    # 🎯 [면적 기반 필터링] 표 교차점의 점(`.`) 노이즈 전수 제거
    # -------------------------------------------------------------
    # 독립된 흰색 덩어리(글자 및 찌꺼기)들을 추적합니다.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(result_without_lines)
    
    # 2배 확대 기준, 픽셀 면적이 15 이하인 작은 파편들은 무조건 제거
    # (일부 글자 받침이 깨진다면 8~12로 낮추고, 노이즈가 남으면 20 단위로 높여보세요)
    min_area = 15 
    
    for i in range(1, num_labels):  # 0번 배경 제외
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area:
            result_without_lines[labels == i] = 0
            
    # 5. Tesseract 입력용 흑백 반전 (흰 배경에 검은 글씨)
    result_final = cv2.bitwise_not(result_without_lines)
    
    # 6. 흐려진 글자 경계를 칼같이 세워주는 샤프닝(선명화) 필터 적용
    sharpening_kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    sharpened_img = cv2.filter2D(result_final, -1, sharpening_kernel)
    
    # 7. 최종 가공된 OpenCV 이미지(numpy)를 Tesseract 인식용 Pillow 이미지로 변환하여 반환
    return Image.fromarray(sharpened_img)

def remove_table_lines2(file_stream):
    # 1. 이미지 로드 및 흑백화(Grayscale)
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. 이진화 (검은 글씨와 흰 배경으로 분리)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # 3. 가로선 탐지 및 제거
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    # 4. 세로선 탐지 및 제거
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    # 5. 추출된 선들을 결합한 후 원본에서 빼기 (선 제거)
    table_lines = cv2.add(detect_horizontal, detect_vertical)
    result_without_lines = cv2.subtract(thresh, table_lines)
    
    # 6. 배경을 다시 흰색으로, 글자를 검은색으로 반전 (Tesseract 최적화)
    result_final = cv2.bitwise_not(result_without_lines)
    
    # 가독성을 위해 2倍 선명도 개선을 위한 Resize 추가 가능
    result_final = cv2.resize(result_final, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    return Image.fromarray(result_final)


def get_masked_image_buffer2(file_stream):
    """
    이미지에서 표 선을 지우고, 2배 확대한 전처리 이미지를 
    파일 시스템 저장 없이 메모리 바이트 버퍼(BytesIO) 형태로 반환합니다.
    """
    file_stream.seek(0)
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    file_stream.seek(0) # 커서 리셋
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 이진화 및 모폴로지 선 제거
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    table_lines = cv2.add(detect_horizontal, detect_vertical)
    result_without_lines = cv2.subtract(thresh, table_lines)
    
    result_final = cv2.bitwise_not(result_without_lines)
    result_final = cv2.resize(result_final, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    
    # --- [다운로드용 메모리 버퍼 변환 로직 추가] ---
    # OpenCV 이미지(numpy)를 Pillow 이미지로 변환
    pil_img = Image.fromarray(result_final)
    
    # 메모리 스트림 객체 생성
    img_buffer = BytesIO()
    # Pillow 이미지를 PNG 포맷으로 메모리 버퍼에 저장
    pil_img.save(img_buffer, format='PNG')
    # 버퍼의 포인터를 처음으로 되돌림
    img_buffer.seek(0)
    
    return img_buffer

def get_masked_image_buffer(file_stream):
    """
    이미지에서 표 선을 제거하고, 면적 기반 필터링으로 자잘한 점 노이즈를 지운 뒤
    화질을 개선하여 메모리 버퍼(BytesIO) 형태로 반환합니다.
    """
    # 1. 파일 스트림 포인터 리셋 및 로드
    file_stream.seek(0)
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    file_stream.seek(0) 
    
    # 2. 화질 열화 방지를 위해 고성능 란초스(Lanczos) 보간법으로 먼저 2배 확대
    img_resized = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 3. 이진화 처리 (Otsu 알고리즘으로 임계값 자동 제어)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # 4. 모폴로지 연산으로 가로/세로 표 선 탐지 및 제거
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    detect_horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    detect_vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
    
    table_lines = cv2.add(detect_horizontal, detect_vertical)
    result_without_lines = cv2.subtract(thresh, table_lines)
    
    # -------------------------------------------------------------
    # 🎯 [핵심 추가] 3. 면적(Area) 기반 독립 덩어리 필터링 실전 적용
    # -------------------------------------------------------------
    # 현재 result_without_lines는 글자가 흰색(255), 배경이 검은색(0)인 상태입니다.
    # 이 상태에서 독립된 흰색 덩어리(글자 및 점 찌꺼기)들을 추적합니다.
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(result_without_lines)
    
    # 이미지 크기가 2배 확대되었으므로, 노이즈 판정 기준 면적을 설정합니다.
    # 픽셀 면적이 15 이하인 아주 작은 점들은 표 선이 지워지며 남은 파편으로 간주하고 제거합니다.
    # (글자가 함께 지워진다면 이 값을 8~12 정도로 낮추고, 노이즈가 남는다면 20~25로 높여보세요)
    min_area = 15 
    
    for i in range(1, num_labels):  # 0번은 검은색 전체 배경이므로 1번부터 순회
        area = stats[i, cv2.CC_STAT_AREA]
        
        if area < min_area:
            # 기준 면적보다 작은 덩어리 영역을 검은색 배경(0)으로 채워 흔적을 지웁니다.
            result_without_lines[labels == i] = 0
    
    # 5. Tesseract 입력 규격에 맞게 흑백 반전 (흰 배경에 검은 글씨)
    result_final = cv2.bitwise_not(result_without_lines)
    
    # 6. 화질 복구를 위한 샤프닝(선명화) 마스크 필터 적용
    sharpening_kernel = np.array([
        [ 0, -1,  0],
        [-1,  5, -1],
        [ 0, -1,  0]
    ])
    sharpened_img = cv2.filter2D(result_final, -1, sharpening_kernel)
    
    # 7. 최종 가공된 이미지를 메모리 버퍼(BytesIO)에 담아 반환
    pil_img = Image.fromarray(sharpened_img)
    img_buffer = BytesIO()
    pil_img.save(img_buffer, format='PNG')
    img_buffer.seek(0)
    
    return img_buffer