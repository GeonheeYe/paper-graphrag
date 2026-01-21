from flask import Flask
from .config import Config
from .services.llm_service import load_model, load_hf_model 

def create_app():
    app = Flask(__name__)
   
    # config 설정
    app.config.from_object(Config)
    
    # 모델 로드 (앱 시작 시 한 번만 로드)
    vl_processor, vl_model = load_model(app.config['VL_MODEL'])
    # 모델을 app.config에 저장하여 재사용
    app.config['VL_PROCESSOR'] = vl_processor
    app.config['VL_MODEL'] = vl_model
    # health 라우트 등록
    from .api.health import bp as health_bp

    # document 라우트 등록
    from .api.documents import bp as documents_bp 

    # query 라우트 등록
    from .api.query import bp as query_bp
    hf_model, hf_tokenizer = load_hf_model(app.config['HF_MODEL'])
    app.config['HF_MODEL'] = hf_model
    app.config['HF_TOKENIZER'] = hf_tokenizer

    # health 체크용 Blueprint를 Flask 앱에 등록하는 코드
    app.register_blueprint(health_bp, url_prefix='/health')

    # document 라우트 등록
    app.register_blueprint(documents_bp, url_prefix='/v1/documents')

    # query 라우트 등록
    app.register_blueprint(query_bp, url_prefix='/v1/query')

    return app

