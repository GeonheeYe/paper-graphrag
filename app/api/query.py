from flask import Blueprint, request, jsonify, current_app
from transformers.models.superpoint.modeling_superpoint import top_k_keypoints
from ..services.retrieval_service import RetrievalService

# query 관련 API 묶음
bp = Blueprint('query', __name__)

@bp.post('/query')
def query():
    f"""
    POST /v1/query
    JSON: 
    {{
        "doc_id": "doc_1234567890",
        "query": "query text",
        "top_k": 5
    }}
    """
    
    body = request.get_json(silent=True) or {}
    doc_id = body.get('doc_id')
    question = body.get('question')
    top_k = int(body.get('top_k', 5))

    if not doc_id or not question: 
        return jsonify({"error": "doc_id and question are required"}), 400

    service = RetrievalService.from_app_config(current_app.config)
    result = service.answer(doc_id=doc_id, question=question, top_k=top_k)

    return jsonify(result)


    