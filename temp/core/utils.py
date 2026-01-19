import re
import os
import json
from io import BytesIO

import pymupdf
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

from core.prompt import table_prompt, figure_prompt
from core.metadata import make_rag_metadata, make_graph_metadata


NUM_ONLY_RE = re.compile(r"^\s*\d+(\.\d+)?%?\s*$")   # 54, 2.9%, 12
SHORT_LINE_RE = re.compile(r"^\s{2,}\S{1,10}\s*$")   # 들여쓰기 + 짧은 토큰
WORD_RE = re.compile(r"[A-Za-z가-힣]")


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


def image_to_table_text(image, caption, flag, processor, model):
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


def save_or_return_name(page, y0, y1, page_idx, flag, idx, save_path, save_option=False, margin=500):
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

    if save_option:
        out_name = f"page{page_idx}_{flag}_{idx}.png"
        pix.save(os.path.join(save_path, out_name))
    else:
        png_bytes = pix.tobytes("png")
        buffer = BytesIO(png_bytes)
        img = Image.open(buffer).convert("RGB")
        img.load()
        return img


def process_table(
    paper_id,
    page,
    y0,
    y1, 
    page_idx,
    block_idx,
    block_text,
    table_idx,
    order,
    processor,
    model,
    type,
    save_path,
    save_option=False,
    GraphRAG_OPTION=False,
):
    """
    테이블 처리 및 벡터 스토어에 추가

    Returns:
        다음 table_idx
    """
    flag = 'table'
    caption = block_text.split('\n')[0].strip()

    if save_option:
        save_or_return_name(page, y0, y1, page_idx, flag, idx=table_idx, save_path=save_path, save_option=save_option)
    else:
        img = save_or_return_name(page, y0, y1, page_idx, flag, idx=table_idx, save_path=save_path, save_option=save_option)
        answer = image_to_table_text(img, caption, flag=flag, processor=processor, model=model)
        csv, summary = extract_csv_and_summary(answer)

        if GraphRAG_OPTION: # GraphRAG
            metadatas = make_graph_metadata(
                paper_id=paper_id,
                doc_id=f"{paper_id}_PAGE{page_idx}_TABLE{table_idx}",
                page_idx=page_idx,
                table_idx=table_idx,
                block_idx=block_idx,
                order=order,
                type=type,
            ).to_dict()
        else: # RAG
            metadatas = make_rag_metadata(
                paper_id=paper_id,
                doc_id=f"{paper_id}_PAGE{page_idx}_TABLE{table_idx}",
                page_idx=page_idx,
                type ="table",
                block_idx=block_idx,
                table_idx=table_idx,
            ).to_dict()
        doc_text = f"[캡션]:{caption}\n[요약]:{summary}"
        with open(os.path.join(save_path, f'table_{table_idx}.csv'), 'w', encoding='utf-8') as f:
            f.write(csv)
        
        print(f'table {table_idx} added')
        with open(os.path.join(save_path, 'log.jsonl'), 'a', encoding='utf-8') as f:
            temp = metadatas.copy()
            temp['llm_summary'] = summary
            temp['llm_csv'] = csv
            f.write(json.dumps(temp, ensure_ascii=False) + '\n')

    return doc_text, metadatas, table_idx + 1
    

def process_figure(
    paper_id,
    page,
    y0,
    y1,
    page_idx,
    block_idx,
    block_text,
    figure_idx,
    order,
    processor,
    model,
    type,
    save_path,
    save_option=False,
    GraphRAG_OPTION=False,
):
    """
    그림 처리 및 벡터 스토어에 추가

    Returns:
        다음 figure_idx
    """
    caption = block_text.split('\n')[0].strip()
    flag = 'figure'

    if save_option:
        save_or_return_name(page, y0, y1, page_idx, flag, idx=figure_idx, save_path=save_path, save_option=save_option)
    else:
        img = save_or_return_name(page, y0, y1, page_idx, flag, idx=figure_idx, save_path=save_path, save_option=save_option)
        answer = image_to_table_text(img, caption, flag=flag, processor=processor, model=model)

        if GraphRAG_OPTION: # GraphRAG
            metadatas = make_graph_metadata(
                paper_id=paper_id,
                doc_id=f"{paper_id}_PAGE{page_idx}_FIGURE{figure_idx}",
                page_idx=page_idx,
                block_idx=block_idx,
                figure_idx=figure_idx,
                order=order,
                type=type,
            ).to_dict()
        else: # RAG
            metadatas = make_rag_metadata(
                paper_id=paper_id,
                doc_id=f"{paper_id}_PAGE{page_idx}_FIGURE{figure_idx}",
                page_idx=page_idx,
                type="figure",
                block_idx=block_idx,
                figure_idx=figure_idx,
            ).to_dict()

        doc_text = f"[캡션]:{caption}\n[요약]:{answer}"
        print(f'figure {figure_idx} added')

        with open(os.path.join(save_path, 'log.jsonl'), 'a', encoding='utf-8') as f:
            temp = metadatas.copy()
            temp['llm_answer'] = answer
            f.write(json.dumps(temp, ensure_ascii=False) + '\n')

    return doc_text, metadatas, figure_idx + 1


def process_text_block(
        paper_id,
        block_text, 
        page_idx, 
        block_idx, 
        order,
        section,
        type,
        GraphRAG_OPTION=False):
    """
    텍스트 블록 처리 및 벡터 스토어에 추가
    """
    answer = block_text.strip()
    if GraphRAG_OPTION: # GraphRAG
        metadatas = make_graph_metadata(
            paper_id=paper_id,
            doc_id=f"{paper_id}_PAGE{page_idx}_BLOCK{block_idx}",
            page_idx=page_idx,
            block_idx=block_idx,
            order=order,
            section=section,
            type=type,
        ).to_dict()
    else: # RAG
        metadatas = make_rag_metadata(
            paper_id=paper_id,
            doc_id=f"{paper_id}_PAGE{page_idx}_BLOCK{block_idx}",
            page_idx=page_idx,
            type="normal_chunk",
            block_idx=block_idx,
        ).to_dict()
    return answer, metadatas
