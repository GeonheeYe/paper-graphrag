# Paper RAG & GraphRAG - PDF 문서 기반 질의응답 시스템

논문 PDF 문서를 업로드하고 벡터 데이터베이스에 인덱싱한 뒤, 자연어 질의를 통해 관련 정보를 검색할 수 있는 RAG (Retrieval-Augmented Generation) 시스템입니다.

## 주요 기능

- 📄 **PDF 문서 업로드 및 인덱싱**: PDF 파일을 업로드하여 벡터 데이터베이스에 저장
- 🔍 **의미 기반 검색**: Chroma DB를 활용한 벡터 유사도 검색
- 🖼️ **Vision-Language 모델**: 테이블 및 그림 처리 (Qwen3-VL)
- 🌐 **RESTful API**: Flask 기반의 간단한 API 인터페이스
- 🇰🇷 **한국어 임베딩 모델**: bge-m3-korean 모델을 사용한 한국어 최적화

## 프로젝트 구조

```
Paper RAG & GraphRAG /
├── app/
│   ├── __init__.py          # Flask 앱 팩토리
│   ├── config.py            # 설정 관리
│   ├── api/                 # API 엔드포인트
│   │   ├── documents.py     # 문서 업로드/인덱싱
│   │   └── health.py        # 헬스체크
│   ├── repositories/        # 데이터 저장소 레이어
│   │   ├── file_repo.py     # 파일 시스템 관리
│   │   └── vector_repo.py   # 벡터 DB 관리
│   ├── services/            # 비즈니스 로직
│   │   ├── ingest_service.py    # 문서 인덱싱 서비스
│   │   ├── retrieval_service.py # 검색 서비스 (예정)
│   │   └── llm_service.py      # LLM 서비스
│   └── utils/              # 유틸리티 함수
│       ├── ids.py           # ID 생성
│       ├── metadata.py      # 메타데이터 관리
│       ├── prompt.py        # 프롬프트 템플릿
│       └── utils.py         # 공통 유틸리티
├── data/                    # 데이터 디렉토리
│   ├── uploads/            # 업로드된 PDF 파일
│   ├── chroma/             # Chroma 벡터 DB
│   └── logs/               # 로그 파일
├── run.py                  # 애플리케이션 실행 파일
├── requirements.txt        # Python 패키지 의존성
└── README.md              # 프로젝트 문서
```

## 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd paper-graphrag
```

### 2. 가상환경 생성 및 활성화

```bash
# Python 3.12 
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 환경 변수 설정

프로젝트 루트에 `.env` 파일을 생성하고 다음 변수들을 설정하세요:

```env
# 디렉토리 설정
UPLOAD_DIR=./data/uploads
CHROMA_DIR=./data/chroma
SAVE_PATH=./data/logs
SAVE_OPTION=False

# 임베딩 모델
EMBED_MODEL=upskyy/bge-m3-korean

# Vision-Language 모델 (선택사항)
VL_MODEL=Qwen/Qwen3-VL-4B-Instruct

# OpenAI API (선택사항)
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
```

## 실행 방법

### 개발 서버 실행

```bash
python run.py
```

서버가 `http://localhost:5001`에서 실행됩니다.

### 프로덕션 환경

```bash
# Gunicorn 사용 예시
gunicorn -w 4 -b 0.0.0.0:5001 run:app
```

## API 엔드포인트

### 1. 헬스체크

```http
GET /health
```

**응답:**
```json
{
  "status": "ok"
}
```

### 2. 문서 업로드 및 인덱싱

```http
POST /v1/documents
Content-Type: multipart/form-data
```

**요청:**
- `file`: PDF 파일 (multipart/form-data)

**응답 (성공):**
```json
{
  "doc_id": "doc_550e8400e29b",
  "file_name": "document.pdf",
  "status": "indexed",
  "message": "Document indexed successfully"
}
```

**응답 (이미 인덱싱됨):**
```json
{
  "doc_id": "doc_550e8400e29b",
  "file_name": "document.pdf",
  "status": "already_indexed",
  "message": "Document already exists in the vector database"
}
```

### 3. 질의응답 (구현 예정)

현재 API로는 문서 업로드/인덱싱만 제공됩니다. 질의응답 엔드포인트는 추후 추가됩니다.

## 주요 기능 설명

### 문서 인덱싱 프로세스

1. **PDF 업로드**: 사용자가 PDF 파일을 업로드
2. **텍스트 추출**: PyMuPDF를 사용하여 PDF에서 텍스트 블록 추출
3. **전처리**: 불필요한 텍스트 제거 및 정제
4. **청킹**: 텍스트를 의미 있는 단위로 분할
5. **임베딩**: 각 청크를 벡터로 변환
6. **저장**: Chroma DB에 벡터 및 메타데이터 저장

### 검색 프로세스

1. **질의 임베딩**: 사용자 질의를 벡터로 변환
2. **유사도 검색**: 벡터 DB에서 유사한 청크 검색
3. **컨텍스트 구성**: 검색된 청크들을 컨텍스트로 구성
4. **LLM 응답 생성**: 컨텍스트와 질의를 기반으로 답변 생성

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.

## 문의

프로젝트 관련 문의사항이 있으시면 이슈를 생성해주세요.
