# blueprints/document.py
from flask import Blueprint, request, jsonify

document_bp = Blueprint('document_bp', __name__)

documents = []  # Lưu trữ tài liệu dưới dạng văn bản

@document_bp.route('/upload_document', methods=['POST'])
def upload_document():
    content = request.json.get('content')
    documents.append(content)
    return jsonify({"message": "Document stored successfully", "documents": documents})

@document_bp.route('/query_document', methods=['POST'])
def query_document():
    question = request.json.get('question')
    result = [doc for doc in documents if question.lower() in doc.lower()]
    return jsonify({"result": result})
