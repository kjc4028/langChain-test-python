import pytesseract


from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv


from utils import *;

# 1. 환경 변수(.env) 로드
load_dotenv()

app = Flask(__name__)

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Refine\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

   
@app.route('/ocr', methods=['POST'])
def extract_text_from_image():
    print('testing....')
        
    # 요청에 파일이 포함되어 있는지 확인
    if 'image' not in request.files:
        return jsonify({"error": "요청에 'image' 파일이 포함되어 있지 않습니다."}), 400
        
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"error": "선택된 파일이 없습니다."}), 400

    try:
        # 3. Pillow를 사용하여 업로드된 이미지 파일 열기
        image = Image.open(file.stream)
        
        # 4. Tesseract OCR 실행 (한글과 영어를 동시에 인식하도록 설정)
        # lang='kor+eng'로 지정하면 이미지 속 한글과 영어를 모두 추출합니다.
        extracted_text = pytesseract.image_to_string(image, lang='kor+eng')
        
        # 앞뒤 공백 제거
        extracted_text = extracted_text.strip()
        
        if not extracted_text:
            return jsonify({
                "status": "warning",
                "message": "이미지에서 텍스트를 감지하지 못했습니다. 해상도를 높이거나 다른 이미지를 시도해 보세요.",
                "text": ""
            })

        return jsonify({
            "status": "success",
            "text": extracted_text
        })

    except Exception as e:
        return jsonify({"error": f"이미지 처리 중 오류 발생: {str(e)}"}), 500
    
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




# 3. 통합 OCR + AI 분석 엔드포인트
@app.route('/ocr/order', methods=['POST'])
def analyze_receipt():
    if 'image' not in request.files:
        return jsonify({"error": "image 파일이 없습니다."}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "선택된 파일이 없습니다."}), 400

    try:
        # Step 1: 이미지 로드 및 오프라인 OCR 실행
        # image = Image.open(file.stream)
        image = remove_table_lines(file.stream)
        custom_config = r'--psm 6 --oem 3 -c preserve_interword_spaces=1'
        raw_text = pytesseract.image_to_string(image, lang='kor', config=custom_config)
        # raw_text = pytesseract.image_to_string(image, lang='kor')
        raw_text = raw_text.strip()
        
        if not raw_text:
            return jsonify({"error": "이미지에서 텍스트를 인식하지 못했습니다."}), 422

        app.logger.info(f"--- [OCR 추출 결과] ---\n{raw_text}\n----------------------")

        # Step 2: 추출된 날것의 텍스트를 LangChain 체인에 입력하여 정제
        # LLM이 텍스트를 분석하여 내부적으로 파싱한 뒤 파이썬 딕셔너리(Dict) 형태로 반환합니다.
        ai_analysis_result = receipt_chain.invoke({"receipt_text": raw_text})

        # Step 3: 최종 가공된 깔끔한 결과 반환
        return jsonify({
            "status": "success",
            "ocr_raw_length": len(raw_text),
            "data": ai_analysis_result
        })

    except Exception as e:
        app.logger.error(f"파이프라인 에러 발생: {str(e)}")
        return jsonify({"error": f"서버 내부 처리 오류: {str(e)}"}), 500



# [추가] 마스킹된 이미지 다운로드 전용 엔드포인트
@app.route('/ocr/table/download', methods=['POST'])
def download_processed_image():
    if 'image' not in request.files:
        return jsonify({"error": "image 파일이 요청에 포함되지 않았습니다."}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "선택된 파일이 없습니다."}), 400

    try:
        # 1. 이미지 전처리 후 메모리 버퍼 획득
        processed_image_buffer = get_masked_image_buffer(file.stream)
        
        # 2. Flask의 send_file을 사용하여 버퍼를 파일로 스트리밍 반환
        # download_name 설정 시 유저 브라우저에서 해당 이름으로 파일이 저장됩니다.
        return send_file(
            processed_image_buffer,
            mimetype='image/png',
            as_attachment=True,
            download_name='processed_masked_image.png'
        )
        
    except Exception as e:
        app.logger.error(f"이미지 다운로드 처리 중 실패: {str(e)}")
        return jsonify({"error": f"서버 내부 처리 오류: {str(e)}"}), 500
    
if __name__ == '__main__':
    app.run(debug=True, port=5000)
    

