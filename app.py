import os
import pytesseract
from flask import Flask, request, jsonify
from PIL import Image
import io
import openai
import logging
import cv2
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
from dotenv import load_dotenv
load_dotenv()

# Đặt đường dẫn đến tệp tesseract.exe
pytesseract.pytesseract.tesseract_cmd = r'F:\Tesseract-OCR\tesseract.exe'

# Đặt API key cho OpenAI GPT-4
openai.api_key = os.getenv("OPENAI_API_KEY")

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
learned_data = []

# ================================================================
# Hàm nạp dữ liệu đã học từ tệp văn bản
def load_learned_data():
    try:
        with open('learned_data.txt', 'r', encoding='utf-8') as file:
            return file.readlines()
    except FileNotFoundError:
        logging.warning("No learned data found, initializing with empty data.")
        return []

# Lưu dữ liệu đã học vào tệp văn bản
def save_learned_data(text):
    with open('learned_data.txt', 'a', encoding='utf-8') as file:
        file.write(text + "\n")
    logging.info("Learned data saved successfully.")

# Hàm để reset bộ nhớ học khi file không tồn tại
def reset_learned_data():
    global learned_data
    learned_data = []
    logging.info("Learned data has been reset.")

# Kiểm tra nếu file 'learned_data.txt' không tồn tại, reset bộ nhớ học
if not os.path.exists('learned_data.txt'):
    reset_learned_data()

# Nạp dữ liệu học vào bộ nhớ
learned_data = load_learned_data()

# ================================================================
# Hàm để làm sạch văn bản và chỉ giữ lại các ký tự quan trọng
def clean_text(text):
    if isinstance(text, str):
        return re.sub(r'[^a-zA-Z0-9\s.,!?áàảãạâấầẩẫậăắằẳẵặéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđÁÀẢÃẠÂẤẦẨẪẬĂẮẰẲẴẶÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐ-]', '', text).strip()
    return text

# Hàm tìm câu trả lời từ dữ liệu đã học
def find_answer_from_learned_data(question):
    question_cleaned = clean_text(question)
    documents = [clean_text(doc) for doc in learned_data]

    if not documents:  # Kiểm tra nếu không có dữ liệu học
        return None

    # Tạo vector TF-IDF từ câu hỏi và các văn bản đã học
    vectorizer = TfidfVectorizer().fit_transform([question_cleaned] + documents)
    vectors = vectorizer.toarray()

    # Tính toán độ tương đồng cosine giữa câu hỏi và các câu đã học
    cosine_similarities = cosine_similarity([vectors[0]], vectors[1:]).flatten()

    # Lấy văn bản có độ tương đồng cao nhất
    best_match_index = np.argmax(cosine_similarities)
    
    if cosine_similarities[best_match_index] > 0.3:  # Giảm ngưỡng để chấp nhận nhiều kết quả hơn
        return documents[best_match_index]
    return None

# ================================================================
# Hàm tìm thông tin cụ thể từ câu hỏi
def find_specific_information(question, data):
    keywords = re.findall(r'\b\w+\b', question.lower())  # Tìm tất cả các từ khóa trong câu hỏi
    filtered_data = []

    for d in data:
        if all(keyword in d.lower() for keyword in keywords):
            filtered_data.append(d)

    # Loại bỏ dấu ** trong dữ liệu
    cleaned_data = [re.sub(r'\*\*', '', d) for d in filtered_data]
    
    return cleaned_data

# Endpoint để lấy địa điểm từ lịch sử hội thoại
@app.route('/get_locations', methods=['POST'])
def get_locations():
    history = request.json['history']
    
    # Trích xuất địa điểm từ GPT
    locations = query_learned_data(history)
    
    if locations:
        return jsonify({"locations": locations}), 200
    else:
        return jsonify({"error": "Không tìm thấy địa điểm nào"}), 404

# Hàm query dữ liệu đã học và GPT-4
def query_learned_data(history):
    try:
        question = history[-1]['content']
        prompt = "Analyze this conversation and extract only the locations mentioned:\n" + "\n".join([h['content'] for h in history])

        # Lọc thông tin cụ thể từ dữ liệu đã học dựa trên từ khóa trong câu hỏi
        filtered_data = find_specific_information(question, learned_data)
        
        if filtered_data:
            return "\n".join(filtered_data)  # Trả về dữ liệu cụ thể đã lọc
        else:
            # Khi không có dữ liệu cụ thể, chuyển sang chế độ tư vấn linh hoạt
            messages = [
                {"role": "system", "content": "You are an assistant that extracts locations and itinerary from conversation history."},
                {"role": "system", "content": "\n".join(learned_data) },  
                {"role": "user", "content": question}
            ]
            
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=4096,  # Giới hạn token tối đa
                temperature=0.7,
                stop=None  
            )
            
            answer = response.choices[0].message['content'].strip()
            
            # Kiểm tra và yêu cầu thêm thông tin nếu câu trả lời bị ngắt quãng
            while len(answer.split()) < 50:
                messages.append({"role": "user", "content": "Hãy tiếp tục."})
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=4096,  # Tiếp tục yêu cầu thêm thông tin
                    temperature=0.7,
                    stop=None
                )
                answer += "\n" + response.choices[0].message['content'].strip()
            
            return answer  # Trả về câu trả lời từ GPT-4
    except Exception as e:
        logging.error(f"Lỗi khi truy vấn GPT-4: {str(e)}")
        return "Đã xảy ra lỗi khi xử lý yêu cầu của bạn."

# ================================================================
# Endpoint để học từ văn bản
@app.route('/learn', methods=['POST'])
def learn():
    data = request.json
    if not data or 'text' not in data:
        return jsonify({"error": "Không có văn bản được cung cấp"}), 400
    
    text = data['text'].strip()
    if not text:
        return jsonify({"error": "Văn bản rỗng"}), 400
    
    # Lưu văn bản đã học vào bộ nhớ và tệp văn bản
    learned_data.append(text)
    save_learned_data(text)
    
    return jsonify({"message": "Học văn bản thành công"}), 200

# ================================================================
# Hàm trích xuất văn bản từ hình ảnh và làm sạch dữ liệu
def extract_and_clean_text_from_image(image_content):
    try:
        image = Image.open(io.BytesIO(image_content))
        text = pytesseract.image_to_string(image, lang='eng')
        cleaned_text = clean_text(text)
        return cleaned_text
    except Exception as e:
        logging.error(f"Lỗi khi xử lý hình ảnh: {str(e)}")
        return None


# ================================================================
# Endpoint để gửi câu hỏi đến GPT-4 hoặc dữ liệu đã học
@app.route('/ask', methods=['POST'])
def ask():
    history = request.json['history']
    
    # Truy vấn thông tin từ dữ liệu đã học và lịch sử hội thoại
    answer = query_learned_data(history)
    
    return jsonify({"response": answer}), 200

if __name__ == '__main__':
    app.run(debug=True)
