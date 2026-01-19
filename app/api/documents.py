from flask import Blueprint, request, jsonify, current_app
from ..utils.ids import new_id
from ..repositories.file_repo import FileRepo 
from ..services.ingest_service import IngestService

bp = Blueprint('documents', __name__)

@bp.post('/')
def upload_document():
    """
    POST /v1/documents
    multipart/from-data file=<pdf>
    """
    if 'file' not in request.files:
        return jsonify({'error': 'Missing file field'}), 400

    f = request.files['file']

    # pdf 파일 확장자 확인
    if not f.filename.lower().endswith('.pdf'):
        return jsonify({'error': 'Only .pdf is supported'}), 400
    
    # doc_id 생성
    doc_id = new_id("doc")

    # file_repo 인스턴스 생성
    file_repo = FileRepo(current_app.config['UPLOAD_DIR']) #./data/uploads

    # pdf 파일을 저장
    pdf_path = file_repo.save_pdf(doc_id, f)

    # ingest 서비스 인스턴스 생성
    ingest = IngestService.from_app_config(current_app.config)

    # 이미 인덱싱된 문서인지 확인
    if ingest.vector_repo.doc_exists(doc_id):
        return jsonify({
            "doc_id": doc_id,
            "file_name": f.filename,
            "status": "already_indexed",
            "message": "Document already exists in the vector database"
        }), 200
    else:
        # ingest 서비스 실행 (문서 인덱싱)   
        ingest.ingest(doc_id=doc_id, pdf_path=pdf_path)
        return jsonify({
            "doc_id": doc_id,
            "file_name": f.filename,
            "status": "indexed",
            "message": "Document indexed successfully"
        }), 201
