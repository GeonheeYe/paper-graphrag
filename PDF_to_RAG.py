import pymupdf  # PyMuPDF
import re
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import json
from io import BytesIO
from prompt import table_prompt, figure_prompt
import os

# 전역 설정
SAVE_OPTION = False
PERSIST_DIRECTORY = "./chroma_db"
FILE_PATH = "GRPO.pdf"

def save_or_return_name(page, y0, y1, page_idx, flag, idx, margin=500):
    """
    PDF 페이지에서 특정 영역을 이미지로 추출
    
    Args:
        page: PyMuPDF 페이지 객체
        y0, y1: 영역 좌표 (y축)
        page_idx: 페이지 인덱스
        flag: 이미지 타입 ('table' 또는 'figure')
        idx: 이미지 인덱스
        margin: 상단 여백 (기본값: 250)
    
    Returns:
        PIL Image 객체
    """
    rect = pymupdf.Rect(70, y0 - margin, 525, y1)
    pix = page.get_pixmap(clip=rect, dpi=200)
    
    if SAVE_OPTION:
        out_name = f"page{page_idx}_{flag}_{idx}.png"
        os.makedirs("figure_table", exist_ok=True)
        pix.save(f"figure_table/{out_name}")
    else:
        png_bytes = pix.tobytes("png")
        buffer = BytesIO(png_bytes)
        img = Image.open(buffer).convert("RGB")
        img.load()
        return img


def image_to_table_text(image, caption, flag):
    """
    이미지를 텍스트로 변환 (테이블 또는 그림)
    
    Args:
        image: PIL Image 객체
        flag: 이미지 타입 ('table' 또는 'figure')
    
    Returns:
        변환된 텍스트
    """
    prompt = figure_prompt if flag == 'figure' else table_prompt
    prompt = prompt.format(caption=caption) 
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt.format(flag=flag)},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=text,
        images=image_inputs,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            num_beams=1,
            pad_token_id=processor.tokenizer.eos_token_id,
        )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


def make_metadatas(source, page_idx, block_idx=-1, table_idx=-1, figure_idx=-1, 
                   llm_answer="", llm_summary="", llm_csv=""):
    """
    메타데이터 딕셔너리 생성
    
    Returns:
        메타데이터 딕셔너리
    """
    return {
        "source": source,
        "page_idx": page_idx,
        "block_idx": block_idx,
        "table_idx": table_idx,
        "figure_idx": figure_idx,
        "llm_answer": llm_answer,
        "llm_summary": llm_summary,
        "llm_csv": llm_csv,
    }


def extract_csv_and_summary(answer):
    """
    LLM 응답에서 CSV와 요약을 추출
    
    Args:
        answer: LLM 응답 텍스트
    
    Returns:
        (csv, summary) 튜플
    """
    csv_pos = answer.find('[CSV]')
    sum_pos = answer.find('[요약]')
    
    csv = ""
    summary = ""
    
    if csv_pos != -1:
        csv_start = csv_pos + len('[CSV]')
        csv_part = answer[csv_start:]
    else:
        csv_part = ""
    
    if sum_pos != -1:
        summary_start = sum_pos + len('[요약]')
        summary = answer[summary_start:].strip()
        # CSV 영역이 요약 앞에 있다면 CSV를 요약 전까지만 자르기
        if csv_pos != -1 and sum_pos > csv_pos:
            csv_part = answer[csv_start:sum_pos]
    else:
        # 요약이 없으면 전체를 summary로 fallback
        summary = answer.strip()
    
    # 코드블록 처리
    m = re.search(r"```(?:csv)?\s*(.*?)```", csv_part, re.S)
    csv = (m.group(1).strip() if m else csv_part.strip())
    
    return csv, summary


def process_table(page, y0, y1, page_idx, block_idx, block_text, table_idx):
    """
    테이블 처리 및 벡터 스토어에 추가
    
    Returns:
        다음 table_idx
    """
    flag = 'table'
    caption = block_text.split('\n')[0].strip()
    
    if SAVE_OPTION:
        save_or_return_name(page, y0, y1, page_idx, flag, idx=table_idx)
    else:
        img = save_or_return_name(page, y0, y1, page_idx, flag, idx=table_idx)
        answer = image_to_table_text(img, caption, flag=flag)
        csv, summary = extract_csv_and_summary(answer)
        
        metadatas = make_metadatas(
            source='table',
            page_idx=page_idx,
            table_idx=table_idx,
            llm_summary=summary,
            llm_csv=csv,
            block_idx=block_idx,
        )
        
        doc_text = f"[캡션]:{caption}\n[요약]:{metadatas['llm_summary']}"
        vector_store.add_texts([doc_text], metadatas=[metadatas])
        
        metadatas['caption'] = caption
        with open('log.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(metadatas, ensure_ascii=False) + '\n')
    
    return table_idx + 1


def process_figure(page, y0, y1, page_idx, block_idx, block_text, figure_idx):
    """
    그림 처리 및 벡터 스토어에 추가
    
    Returns:
        다음 figure_idx
    """
    caption = block_text.split('\n')[0].strip()
    flag = 'figure'
    
    if SAVE_OPTION:
        save_or_return_name(page, y0, y1, page_idx, flag, idx=figure_idx)
    else:
        img = save_or_return_name(page, y0, y1, page_idx, flag, idx=figure_idx)
        answer = image_to_table_text(img, caption, flag=flag)
        print("image to text\n", answer)
        metadatas = make_metadatas(
            source='figure',
            page_idx=page_idx,
            figure_idx=figure_idx,
            llm_answer=answer,
            block_idx=block_idx,
        )
        
        doc_text = f"[캡션]:{caption}\n[요약]:{metadatas['llm_answer']}"
        print(f'figure {figure_idx} added')
        vector_store.add_texts([doc_text], metadatas=[metadatas])
        
        metadatas['caption'] = caption
        with open('log.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(metadatas, ensure_ascii=False) + '\n')
    
    return figure_idx + 1


def process_text_block(block_text, page_idx, block_idx):
    """
    텍스트 블록 처리 및 벡터 스토어에 추가
    """
    answer = block_text.strip()
    metadatas = make_metadatas(
        source='text',
        page_idx=page_idx,
        block_idx=block_idx
    )
    return answer, metadatas



NUM_ONLY_RE = re.compile(r"^\s*\d+(\.\d+)?%?\s*$")   # 54, 2.9%, 12
SHORT_LINE_RE = re.compile(r"^\s{2,}\S{1,10}\s*$")   # 들여쓰기 + 짧은 토큰
WORD_RE = re.compile(r"[A-Za-z가-힣]")

def clean_block_text(block_text: str) -> str:
    """
    PyMuPDF block_text에서 쓸모없는 라인 제거
    """
    lines = block_text.splitlines()
    kept_lines = []

    for ln in lines:
        raw = ln
        ln = ln.strip()

        # 1) 빈 줄 제거
        if not ln:
            continue

        # 2) 숫자만 있는 줄 제거 (54, 2.9%, 100 등)
        if NUM_ONLY_RE.match(ln):
            continue

        # 3) 들여쓰기 + 짧은 한 줄 (표 셀 조각)
        if SHORT_LINE_RE.match(raw):
            continue

        # 4) 너무 짧고 단어도 거의 없는 줄
        if len(ln) < 15 and not WORD_RE.search(ln):
            continue

        # 5) 기호 위주 라인 제거
        alpha_num_ratio = sum(c.isalnum() for c in ln) / max(len(ln), 1)
        if alpha_num_ratio < 0.3 and len(ln) < 40:
            continue

        kept_lines.append(ln)

    # 최종 블록 판단
    if not kept_lines:
        return ""

    # 남은 게 너무 짧으면 블록 자체 제거
    merged = " ".join(kept_lines)
    if len(merged) < 40:
        return ""

    return merged


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
            table_idx = process_table(page, y0, y1, page_idx, block_idx, block_text, table_idx)
        
        # 그림 처리 (캡션 기준 중복 제거)
        elif re.match(r"(figure|Figure)\s+\d+", block_text.strip()):
            caption_key = block_text.split('\n')[0].strip()
            if caption_key in seen_figure_captions:
                continue
            seen_figure_captions.add(caption_key)
            figure_idx = process_figure(page, y0, y1, page_idx, block_idx, block_text, figure_idx)
        
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

            answer, metadatas = process_text_block(block_text, page_idx, block_idx)
            pending_texts.append(answer)
            pending_metadatas.append(metadatas)

        # 배치 크기 도달 시 일괄 저장
        if len(pending_texts) >= 64:
            vector_store.add_texts(pending_texts, metadatas=pending_metadatas)
            pending_texts = []
            pending_metadatas = []

# 남은 텍스트 처리
if pending_texts:
    vector_store.add_texts(pending_texts, metadatas=pending_metadatas)
    print(f"Added {len(pending_texts)} remaining text blocks")

print("PDF processing completed!")
