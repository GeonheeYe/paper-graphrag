# 📄 Paper-based GraphRAG System

## 프로젝트 개요
GraphRAG를 활용하여 PDF 논문에서 텍스트, 이미지, 표 등 다양한 정보를 추출하고 구조화된 그래프를 구성하여 질의응답(QA) 성능을 향상시키는 프로젝트입니다.

## 주요 기능
- **PDF 처리**: PyMuPDF를 사용한 논문 PDF 파싱 및 메타데이터 추출
- **멀티모달 처리**: 텍스트, 이미지(Figure/Table) 정보 통합 활용
- **벡터 임베딩**: Sentence Transformers를 통한 의미 기반 임베딩
- **그래프 구성**: 정보 간의 관계를 나타내는 지식 그래프 생성
- **RAG 질의응답**: 구조화된 그래프를 기반으로 한 검색 증강 생성

## 설치 및 실행

### 1. 필수 라이브러리 설치
```bash
pip install -r requirements.txt
```

### 2. PDF 처리 및 RAG 구축
```bash
python PDF_to_RAG.py
```


## 필수 모듈
- `transformers` - 사전학습 모델 로딩
- `sentence-transformers` - 임베딩 생성
- `langchain_community` - RAG 파이프라인 구성
- `pymupdf` - PDF 파일 처리
- `accelerate` - GPU 가속 처리