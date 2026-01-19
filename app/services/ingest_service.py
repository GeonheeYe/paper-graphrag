import re
import pymupdf

from ..repositories.vector_repo import VectorRepo
from ..utils.utils import process_table, process_figure, process_text_block
from ..utils.utils import clean_block_text

class IngestService:
    def __init__(self, vector_repo: VectorRepo, config):
        self.vector_repo = vector_repo
        self.config = config
        
        # config에서 필요한 값들을 미리 추출
        self.vl_model = config.get('VL_MODEL')
        self.save_option = config.get('SAVE_OPTION')
        self.save_path = config.get('SAVE_PATH')
        
        if self.vl_model is None:
            raise ValueError("VL_MODEL must be loaded in create_app()")

    @classmethod
    def from_app_config(cls, config) -> 'IngestService':
        repo = VectorRepo(
            chroma_dir=config['CHROMA_DIR'], #./data/chroma
            embed_model=config['EMBED_MODEL'] # upskyy/bge-m3-korean
        )        
        return cls(vector_repo=repo, config=config)

    def ingest(self, doc_id: str, pdf_path: str) -> str:
        # 텍스트 배치 처리를 위한 리스트
        pending_texts = []
        pending_metadatas = []
        
        seen_text_blocks = set()
        seen_table_captions = set()
        seen_figure_captions = set()
        doc = pymupdf.open(pdf_path)
        
        paper_name =  doc[0].get_text('blocks')[0][4].replace('\n', '')
        print("논문 제목:", paper_name)
        for page_idx, page in enumerate(doc):
            blocks = page.get_text('blocks')

            for block_idx, block in enumerate(blocks):
                _, y0, _, y1, block_text, *_ = block
                # 숫자 점으로 시작하는 라인만 있고 단어가 8개 이하인 경우 스킵
                if re.match(r"^\s*\d+(\.\d+)+\.\s+\S+", block_text.strip()) and len(block_text.strip().split()) <= 8:
                    continue
                # # 테이블 처리 (캡션 기준 중복 제거)
                # if re.match(r"(Table|TABLE)\s+\d+", block_text.strip()):
                #     caption_key = block_text.split('\n')[0].strip()
                #     if caption_key in seen_table_captions:
                #         continue
                #     seen_table_captions.add(caption_key)
                #     doc_text, metadatas, table_idx = process_table(
                #         doc_id,
                #         paper_name,
                #         page,
                #         y0,
                #         y1,
                #         page_idx,
                #         block_idx,
                #         block_text,
                #         table_idx,
                #         self.processor,
                #         self.model,
                #         save_path=self.save_path,
                #         save_option=self.save_option,
                #     )
                #     pending_texts.append(doc_text)
                #     pending_metadatas.append(metadatas)
                
                # # 그림 처리 (캡션 기준 중복 제거)
                # elif re.match(r"(figure|Figure)\s+\d+", block_text.strip()):
                #     caption_key = block_text.split('\n')[0].strip()
                #     if caption_key in seen_figure_captions:
                #         continue
                #     seen_figure_captions.add(caption_key)
                #     doc_text, metadatas, figure_idx = process_figure(
                #         doc_id,
                #         paper_name,
                #         page,
                #         y0,
                #         y1,
                #         page_idx,
                #         block_idx,
                #         block_text,
                #         figure_idx,
                #         self.processor,
                #         self.model,
                #         save_path=self.save_path,
                #         save_option=self.save_option,
                #     )
                    
                #     pending_texts.append(doc_text)
                #     pending_metadatas.append(metadatas)
                
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
                        doc_id,
                        paper_name,
                        block_text, 
                        page_idx, 
                        block_idx,
                        GraphRAG_OPTION=False,
                    )
                    pending_texts.append(answer)
                    pending_metadatas.append(metadatas)

                # 배치 크기 도달 시 일괄 저장
                if len(pending_texts) >= 64:
                    self.vector_repo.upsert_chunks(
                        pending_texts, 
                        pending_metadatas,
                    )
                    # 남은 텍스트 처리
            if pending_texts:
                self.vector_repo.upsert_chunks(pending_texts, pending_metadatas)
                print(f"Added {len(pending_texts)} remaining text blocks")
            print("PDF processing completed!")                    
