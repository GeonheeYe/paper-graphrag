#%%
from langchain_neo4j import Neo4jVector
from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from neo4j import GraphDatabase
import pymupdf
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
"""
docker run -d \
  --name neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
"""

#%%
# Neo4j 연결 설정
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"

#기존 노드 삭제
driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
with driver.session() as session: 
    # 1) 삭제할 인덱스 이름들 조회
    res = session.run("""
        SHOW INDEXES YIELD name, type
        WHERE type IN ['VECTOR', 'FULLTEXT']
        RETURN name
    """)
    index_names = [r["name"] for r in res]

    print("Indexes to drop:", index_names)

    # # 2) 하나씩 DROP
    for idx in index_names:
        print(idx)
        session.run(f"DROP INDEX `{idx}`")
        print("Dropped:", idx)
    print("All nodes deleted")
    session.run("MATCH (n) DETACH DELETE n")
    print("All nodes deleted")
driver.close()
#%%
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"

INDEX_NAME = "paper_index"
NODE_LABEL = "Chunk"            # 라벨은 통일 추천
TEXT_PROP = "text"
EMB_PROP = "embedding"

embeddings = HuggingFaceEmbeddings(model_name="upskyy/bge-m3-korean")


docs = [
    # ===== GRPO 논문 =====
    Document(
        page_content="GRPO 논문에서는 Group Relative Policy Optimization을 제안한다.",
        metadata={
            "paper_id": "paper_grpo_001",
            "paper_title": "GRPO: Group Relative Policy Optimization",
            "page": 1,
            "chunk_id": "paper_grpo_001_c000001",
            "order": 1,
            "type": "abstract"
        }
    ),
    Document(
        page_content="저자 정보: DeepSeek AI 연구팀",
        metadata={
            "paper_id": "paper_grpo_001",
            "paper_title": "GRPO: Group Relative Policy Optimization",
            "page": 1,
            "chunk_id": "paper_grpo_001_c000002",
            "order": 2,
            "type": "authors"
        }
    ),

    # ===== GSPO 논문 =====
    Document(
        page_content="GSPO 논문은 기존 PPO 대비 안정적인 정책 최적화를 목표로 한다.",
        metadata={
            "paper_id": "paper_gspo_002",
            "paper_title": "GSPO: Generalized Stable Policy Optimization",
            "page": 1,
            "chunk_id": "paper_gspo_002_c000001",
            "order": 1,
            "type": "abstract"
        }
    ),
    Document(
        page_content="표 1번: GSPO 성능 평가 결과 (PPO 대비 성능 향상)",
        metadata={
            "paper_id": "paper_gspo_002",
            "paper_title": "GSPO: Generalized Stable Policy Optimization",
            "page": 1,
            "chunk_id": "paper_gspo_002_c000002",
            "order": 2,
            "type": "table_caption"
        }
    ),
]

# 벡터 스토어 생성 (새로 생성)
# 1) 노드 생성 + embedding + 인덱스 생성
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
rows = []
for d in docs:
    rows.append({
        "paper_id": d.metadata["paper_id"],
        "chunk_id": d.metadata["chunk_id"],
        "order": d.metadata.get("order"),   # 없으면 None
        "page": d.metadata.get("page")
    })
driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))
with driver.session() as session:
    # 1) Paper 노드 생성 + Paper-Chunk 연결
    session.run(
        f"""
        UNWIND $rows AS row
        MERGE (p:Paper {{paper_id: row.paper_id}})
        MERGE (c:{NODE_LABEL} {{chunk_id: row.chunk_id}})
        MERGE (p)-[:HAS_CHUNK]->(c)
        SET c.paper_id = row.paper_id
        """,
        rows=rows
    )

    # 2) 같은 paper_id 내에서 order 기준으로 Chunk-NEXT 연결
    #    (order를 넣어둔 경우에만 유효)
    session.run(
        f"""
        UNWIND $rows AS row
        MATCH (c:{NODE_LABEL} {{chunk_id: row.chunk_id}})
        SET c.order = row.order, c.page = row.page

        WITH row.paper_id AS paper_id
        MATCH (a:{NODE_LABEL} {{paper_id: paper_id}}), (b:{NODE_LABEL} {{paper_id: paper_id}})
        WHERE a.order IS NOT NULL AND b.order IS NOT NULL AND b.order = a.order + 1
        MERGE (a)-[:NEXT]->(b)
        """,
        rows=rows
    )

driver.close()
print("Edges created: (Paper)-[:HAS_CHUNK]->(Chunk), (Chunk)-[:NEXT]->(Chunk)")
print("Vector store created successfully!")
# results = vector_store.similarity_search("사과?", k=10)
# for r in results: 
#     print(r.metadata, r.page_content[:120])
#%%
# 벡터 스토어 생성 (새로 생성)
vector_store = Neo4jVector.from_documents(
    documents=docs,
    embedding=embeddings,
    url=NEO4J_URL,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
    index_name="paper_index",
    node_label="chunks",
    text_node_property="text",
    embedding_node_property="embedding",
    search_type="hybrid"
)

print("Vector store created successfully!")
results = vector_store.similarity_search("사과?", k=10)
for r in results: 
    print(r.metadata, r.page_content[:120])