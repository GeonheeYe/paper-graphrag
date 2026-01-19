# RAG MVP - PDF 문서 기반 질의응답 시스템

PDF 문서를 업로드하고 벡터 데이터베이스에 인덱싱한 후, 자연어 질의를 통해 관련 정보를 검색할 수 있는 RAG (Retrieval-Augmented Generation) 시스템입니다.

## 주요 기능

- 📄 **PDF 문서 업로드 및 인덱싱**: PDF 파일을 업로드하여 벡터 데이터베이스에 저장
- 🔍 **의미 기반 검색**: Chroma DB를 활용한 벡터 유사도 검색
- 🖼️ **Vision-Language 모델 지원**: 테이블 및 그림 처리 (Qwen3-VL)
- 🌐 **RESTful API**: Flask 기반의 간단한 API 인터페이스
- 🇰🇷 **한국어 임베딩 모델**: bge-m3-korean 모델을 사용한 한국어 최적화

## 프로젝트 구조

```
RAGMVP/
├── app/
│   ├── __init__.py          # Flask 앱 팩토리
│   ├── config.py            # 설정 관리
│   ├── api/                 # API 엔드포인트
│   │   ├── documents.py     # 문서 업로드/인덱싱
│   │   ├── query.py         # 질의응답
│   │   └── health.py        # 헬스체크
│   ├── repositories/        # 데이터 저장소 레이어
│   │   ├── file_repo.py     # 파일 시스템 관리
│   │   └── vector_repo.py   # 벡터 DB 관리
│   ├── services/            # 비즈니스 로직
│   │   ├── ingest_service.py    # 문서 인덱싱 서비스
│   │   ├── retrieval_service.py # 검색 서비스
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
cd RAGMVP
```

### 2. 가상환경 생성 및 활성화

```bash
# Python 3.8 이상 필요
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
PROCESSOR_MODEL=Qwen/Qwen3-VL-4B-Instruct

# OpenAI API (선택사항)
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
```

## 실행 방법

### 개발 서버 실행

```bash
python run.py
```

서버가 `http://localhost:5000`에서 실행됩니다.

### 프로덕션 환경

```bash
# Gunicorn 사용 예시
gunicorn -w 4 -b 0.0.0.0:5000 run:app
```

## API 엔드포인트

### 1. 헬스체크

```http
GET /health/
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

```http
POST /v1/query
Content-Type: application/json
```

**요청:**
```json
{
  "doc_id": "doc_550e8400e29b",
  "query": "이 논문의 주요 내용은 무엇인가요?",
  "top_k": 5
}
```

## 기술 스택

### 백엔드
- **Flask**: 웹 프레임워크
- **Chroma**: 벡터 데이터베이스
- **LangChain**: LLM 통합 프레임워크

### 머신러닝 모델
- **HuggingFace Embeddings**: `upskyy/bge-m3-korean` (한국어 임베딩)
- **Qwen3-VL**: Vision-Language 모델 (테이블/그림 처리)
- **PyMuPDF**: PDF 파싱

### 기타
- **Python-dotenv**: 환경 변수 관리
- **Transformers**: 모델 로딩 및 추론

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

## 개발 가이드

### 코드 구조

- **Repository 패턴**: 데이터 접근 로직 분리
- **Service 패턴**: 비즈니스 로직 캡슐화
- **Blueprint**: Flask 라우트 모듈화

### 새로운 기능 추가

1. API 엔드포인트: `app/api/`에 새로운 Blueprint 추가
2. 서비스 로직: `app/services/`에 비즈니스 로직 구현
3. 저장소: `app/repositories/`에 데이터 접근 로직 추가

## 문제 해결

### 일반적인 문제

1. **모델 다운로드 실패**
   - 인터넷 연결 확인
   - HuggingFace 토큰 설정 (필요시)

2. **GPU 메모리 부족**
   - `VL_MODEL`을 더 작은 모델로 변경
   - 배치 크기 조정

3. **Chroma DB 오류**
   - `data/chroma` 디렉토리 삭제 후 재시작

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

## 기여

버그 리포트나 기능 제안은 이슈로 등록해주세요.

## 문의

프로젝트 관련 문의사항이 있으시면 이슈를 생성해주세요.
