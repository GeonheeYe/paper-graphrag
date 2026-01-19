import re
import pymupdf  # PyMuPDF
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from core.utils import clean_block_text
from core.utils import process_table, process_figure, process_text_block
import os

# 전역 설정
SAVE_OPTION = False
PERSIST_DIRECTORY = "./chroma_db"
FILE_PATH = "DeepSeekMath_ Pushing the Limits of Mathematical Reasoning in Open Language Models.pdf"
# 모델 돌아간 결과 저장 경로
SAVE_PATH = "./log"
os.makedirs(SAVE_PATH, exist_ok=True)

checkpoint_path = "Qwen/Qwen3-VL-4B-Instruct"
#checkpoint_path = "Qwen/Qwen3-VL-8B-Thinking"

# 모델 및 벡터 스토어 초기화
print("Loading embeddings model...")
embeddings = HuggingFaceEmbeddings(model_name="upskyy/bge-m3-korean")

print("Loading vision model...")
processor = AutoProcessor.from_pretrained(checkpoint_path)

# 모델 로드
model = AutoModelForImageTextToText.from_pretrained(
    checkpoint_path,
    dtype=torch.float16,
    device_map="cuda:0",
).to("cuda")

# 모델을 평가 모드로 설정 (드롭아웃 등 비활성화)
model.eval()

print("Initializing vector store...")
vector_store = Chroma(
    embedding_function=embeddings,
    persist_directory=PERSIST_DIRECTORY,
)

# 매 실행 시 컬렉션을 깨끗하게 다시 생성 (디렉터리를 지웠거나 스키마가 바뀐 경우 포함)
if os.path.exists(PERSIST_DIRECTORY):
    vector_store.reset_collection()

# PDF 처리
print(f"Processing PDF: {FILE_PATH}")

doc = pymupdf.open(FILE_PATH)
table_idx = 1
figure_idx = 1

# 텍스트/테이블/그림 중복 방지를 위한 집합
seen_text_blocks = set()
seen_table_captions = set()
seen_figure_captions = set()

# 텍스트 배치 처리를 위한 리스트 (루프 밖에서 선언)
pending_texts = []
pending_metadatas = []

paper_id =  doc[0].get_text('blocks')[0][4].replace('\n', '')
print("논문 제목:", paper_id)
for page_idx, page in enumerate(doc):
    blocks = page.get_text('blocks')

    for block_idx, block in enumerate(blocks):
        x0, y0, x1, y1, block_text, *_ = block
        # 숫자 점으로 시작하는 라인만 있고 단어가 8개 이하인 경우 스킵
        if re.match(r"^\s*\d+(\.\d+)+\.\s+\S+", block_text.strip()) and len(block_text.strip().split()) <= 8:
            continue
        # 테이블 처리 (캡션 기준 중복 제거)
        if re.match(r"(Table|TABLE)\s+\d+", block_text.strip()):
            caption_key = block_text.split('\n')[0].strip()
            if caption_key in seen_table_captions:
                continue
            seen_table_captions.add(caption_key)
            doc_text, metadatas, table_idx = process_table(
                paper_id,
                page,
                y0,
                y1,
                page_idx,
                block_idx,
                block_text,
                table_idx,
                processor,
                model,
                save_path=SAVE_PATH,
                save_option=SAVE_OPTION,
            )
            pending_texts.append(doc_text)
            pending_metadatas.append(metadatas)
        
        # 그림 처리 (캡션 기준 중복 제거)
        elif re.match(r"(figure|Figure)\s+\d+", block_text.strip()):
            caption_key = block_text.split('\n')[0].strip()
            if caption_key in seen_figure_captions:
                continue
            seen_figure_captions.add(caption_key)
            doc_text, metadatas, figure_idx = process_figure(
                paper_id,
                page,
                y0,
                y1,
                page_idx,
                block_idx,
                block_text,
                figure_idx,
                processor,
                model,
                save_path=SAVE_PATH,
                save_option=SAVE_OPTION,
            )
            
            pending_texts.append(doc_text)
            pending_metadatas.append(metadatas)
        
        # 텍스트 블록 처리
        elif block_text.strip():
            block_text = clean_block_text(block_text)
            # 전처리 후 내용이 비었으면 스킵
            if not block_text:
                continue

            # 같은 텍스트 블록은 한 번만 저장
            text_key = block_text.strip()
            if text_key in seen_text_blocks:
                continue
            seen_text_blocks.add(text_key)

            answer, metadatas = process_text_block(
                paper_id,
                block_text, 
                page_idx, 
                block_idx,
            )
            pending_texts.append(answer)
            pending_metadatas.append(metadatas)

        # 배치 크기 도달 시 일괄 저장
        if len(pending_texts) >= 64:
            vector_store.add_texts(
                pending_texts, 
                metadatas=pending_metadatas,
            )
            pending_texts = []
            pending_metadatas = []

# 남은 텍스트 처리
if pending_texts:
    vector_store.add_texts(pending_texts, metadatas=pending_metadatas)
    print(f"Added {len(pending_texts)} remaining text blocks")

print("PDF processing completed!")
