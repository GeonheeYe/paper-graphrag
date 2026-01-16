#%%
import pymupdf
from langchain_huggingface import HuggingFaceEmbeddings
from neo4j import GraphDatabase
import re
from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from core.metadata import GraphMetadata
from langchain_core.documents import Document
from core.utils import clean_block_text
from core.utils import process_table, process_figure, process_text_block
import os

# Neo4j 연결 설정 
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
FILE_PATH = "DeepSeekMath_ Pushing the Limits of Mathematical Reasoning in Open Language Models.pdf"
SAVE_PATH = "./log"
os.makedirs(SAVE_PATH, exist_ok=True)

driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
# 기존 노드 삭제
with driver.session() as session:
    res = session.run("""
        SHOW INDEXES YIELD name, type
        WHERE type IN ['VECTOR', 'FULLTEXT']
        RETURN name
        """)
    index_names = [r["name"] for r in res]
    print("Indexes to drop:", index_names)

    for idx in index_names:
        print(idx)
        session.run(f"DROP INDEX `{idx}`")
        print("Dopped", idx)
    session.run("MATCH (n) DETACH DELETE n")
    print("All nodes deleted")

checkpoint_path = "Qwen/Qwen3-VL-4B-Instruct"
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

SECTION_NUM_RE = re.compile(
    r"""^\s*
    (?P<num>\d+(\.\d+){0,3})      # 1, 1.2, 2.3.4
    [\)\.]?\s+                    # optional ) or .
    (?P<title>[A-Za-z][A-Za-z0-9 \-,:/]{2,120}?)  # title
    \s*$""",
    re.VERBOSE
)
SECTION_FIX_RE = re.compile(
    r"""^\s*(abstract|introduction|related work|background|
    method|methods|approach|model|preliminaries|
    experiments?|experimental setup|results?|evaluation|
    discussion|limitations|conclusion|conclusions|
    acknowledg(e)?ments?|references|appendix|supplementary material)\s*$""",
    re.IGNORECASE | re.VERBOSE
)
#%%
import pymupdf
doc = pymupdf.open(FILE_PATH)
docs = [] 

paper_id =  doc[0].get_text('blocks')[0][4].replace('\n', '')
print("논문 제목:", paper_id)

table_idx = 1
figure_idx = 1
order = 0
section = None
for page_idx, page in enumerate(doc):
    blocks = page.get_text('blocks')
    for block_idx, block in enumerate(blocks):
        x0, y0, x1, y1, block_text, *_ = block
        # 노드 생성
        block_text = block_text.strip()
        # 섹션 처리 1.1 ~ 
        if SECTION_FIX_RE.match(block_text) or SECTION_NUM_RE.match(block_text):
            section = block_text
            doc_text, metadatas = process_text_block(
                paper_id,
                block_text, 
                page_idx, 
                block_idx,
                order,
                section,
                type='section_head',
                GraphRAG_OPTION=True
            )
            print("섹션 발견:", doc_text, metadatas)
            docs.append(Document(page_content=doc_text, metadata=metadatas))
        
        # 테이블 처리
        elif re.match(r"(Table|TABLE)\s+\d+", block_text.strip()):
            caption_key = block_text.split('\n')[0].strip()
            doc_text, metadatas, table_idx = process_table(
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
                type='table',
                save_path=SAVE_PATH,
                save_option=False,
                GraphRAG_OPTION=True
            )
            print("테이블 발견:", doc_text, metadatas)
            docs.append(Document(page_content=doc_text, metadata=metadatas))
        
        # 그림 처리
        elif re.match(r"(figure|Figure)\s+\d+", block_text.strip()):
            caption_key = block_text.split('\n')[0].strip()
            doc_text, metadatas, figure_idx = process_figure(
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
                type='figure',
                save_path=SAVE_PATH,
                save_option=False,
                GraphRAG_OPTION=True
            )
            docs.append(Document(page_content=doc_text, metadata=metadatas))
            print("그림 발견:", doc_text, metadatas)
        
        #chunk처리
        else: 
            block_text = clean_block_text(block_text)
            if not block_text:
                continue
            doc_text, metadatas = process_text_block(
                paper_id,
                block_text, 
                page_idx, 
                block_idx,
                order,
                section,
                type='paragraph',
                GraphRAG_OPTION=True
            )
            docs.append(Document(page_content=doc_text, metadata=metadatas))
            print("텍스트 발견:", doc_text, metadatas)

        # global count
        order += 1 


from langchain_neo4j import Neo4jVector
vector_store = Neo4jVector.from_documents(
    documents=docs,
    embedding=embeddings,
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name="paper_index",
    node_label="Chunk", # 통일
    text_node_property="text",
    embedding_node_property="embedding",
    search_type="hybrid"
)
# %