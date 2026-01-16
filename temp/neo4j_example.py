"""
langchain_neo4j 사용 예시
Neo4j를 벡터 스토어 및 그래프 데이터베이스로 사용하는 방법
"""

from langchain_neo4j import Neo4jVector, Neo4jGraph
from langchain_neo4j import GraphCypherQAChain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ============================================================================
# 설정
# ============================================================================
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"  # 실제 비밀번호로 변경
EMBEDDING_MODEL = "upskyy/bge-m3-korean"

# ============================================================================
# 예시 1: Neo4jVector - 벡터 스토어로 사용 (Chroma 대신)
# ============================================================================

def example_neoj4_vector_store():
    """Neo4j를 벡터 스토어로 사용하는 예시"""
    print("=" * 50)
    print("예시 1: Neo4jVector - 벡터 스토어")
    print("=" * 50)
    
    # 임베딩 모델 초기화
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Neo4jVector 초기화
    vector_store = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name="paper_chunks",  # 인덱스 이름
        node_label="Chunk",  # 노드 레이블
        text_node_property="text",  # 텍스트 속성
        embedding_node_property="embedding",  # 임베딩 속성
    )
    
    # 또는 새로 생성
    # vector_store = Neo4jVector.from_documents(
    #     documents=documents,  # Document 객체 리스트
    #     embedding=embeddings,
    #     url=NEO4J_URI,
    #     username=NEO4J_USERNAME,
    #     password=NEO4J_PASSWORD,
    #     index_name="paper_chunks",
    #     node_label="Chunk",
    # )
    
    # 텍스트 추가
    texts = [
        "GRPO는 Group Relative Policy Optimization의 약자입니다.",
        "GRPO는 PPO의 개선된 버전입니다.",
    ]
    metadatas = [
        {"source": "text", "page": 1},
        {"source": "text", "page": 2},
    ]
    
    vector_store.add_texts(texts=texts, metadatas=metadatas)
    
    # 유사도 검색
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke("GRPO가 무엇인가?")
    
    for doc in docs:
        print(f"Score: {doc.metadata.get('score', 'N/A')}")
        print(f"Content: {doc.page_content}")
        print("-" * 30)


# ============================================================================
# 예시 2: Neo4jGraph - 그래프 데이터베이스 연결
# ============================================================================

def example_neo4j_graph():
    """Neo4j 그래프 데이터베이스 연결 및 쿼리 예시"""
    print("\n" + "=" * 50)
    print("예시 2: Neo4jGraph - 그래프 데이터베이스")
    print("=" * 50)
    
    # Neo4j 그래프 연결
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
    )
    
    # 스키마 확인
    schema = graph.get_schema
    print("Graph Schema:")
    print(schema)
    
    # Cypher 쿼리 실행
    query = """
    MATCH (n:Chunk)
    RETURN n.text AS text, n.page AS page
    LIMIT 5
    """
    result = graph.query(query)
    print("\nQuery Result:")
    for record in result:
        print(record)
    
    # 노드와 관계 생성 예시
    create_query = """
    CREATE (p:Paper {title: "GRPO Paper", id: "GRPO"})
    CREATE (c:Chunk {text: "GRPO는 Group Relative Policy Optimization입니다", page: 1})
    CREATE (p)-[:CONTAINS]->(c)
    RETURN p, c
    """
    # graph.query(create_query)


# ============================================================================
# 예시 3: GraphCypherQAChain - 그래프 기반 QA
# ============================================================================

def example_graph_cypher_qa():
    """그래프 기반 질의응답 체인 예시"""
    print("\n" + "=" * 50)
    print("예시 3: GraphCypherQAChain - 그래프 기반 QA")
    print("=" * 50)
    
    # Neo4j 그래프 연결
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
    )
    
    # LLM 초기화 (OpenAI 또는 로컬 모델)
    llm = ChatOpenAI(
        model="gpt-4",  # 또는 로컬 모델
        temperature=0,
    )
    
    # GraphCypherQAChain 생성
    chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        verbose=True,
        return_intermediate_steps=True,
    )
    
    # 질문 실행
    question = "GRPO가 무엇인가?"
    result = chain.invoke({"query": question})
    
    print(f"Question: {question}")
    print(f"Answer: {result['result']}")
    print(f"Intermediate Steps: {result.get('intermediate_steps', [])}")


# ============================================================================
# 예시 4: PDF에서 추출한 데이터를 Neo4j에 저장
# ============================================================================

def example_pdf_to_neo4j():
    """PDF에서 추출한 텍스트를 Neo4j에 저장하는 예시"""
    print("\n" + "=" * 50)
    print("예시 4: PDF 데이터를 Neo4j에 저장")
    print("=" * 50)
    
    import pymupdf
    from langchain_core.documents import Document
    
    # PDF 읽기
    FILE_PATH = "GRPO.pdf"
    doc = pymupdf.open(FILE_PATH)
    
    # 임베딩 모델 초기화
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    
    # Neo4jVector 초기화 (새 인덱스 생성)
    vector_store = Neo4jVector(
        embedding=embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name="paper_chunks",
        node_label="Chunk",
        text_node_property="text",
        embedding_node_property="embedding",
    )
    
    # PDF에서 텍스트 추출 및 저장
    documents = []
    for page_idx, page in enumerate(doc):
        blocks = page.get_text('blocks')
        for block_idx, block in enumerate(blocks):
            x0, y0, x1, y1, block_text, *_ = block
            if block_text.strip():
                doc_obj = Document(
                    page_content=block_text.strip(),
                    metadata={
                        "source": "text",
                        "page_idx": page_idx,
                        "block_idx": block_idx,
                        "paper_id": FILE_PATH.split('.')[0],
                    }
                )
                documents.append(doc_obj)
    
    # Neo4j에 추가
    vector_store.add_documents(documents)
    print(f"Added {len(documents)} chunks to Neo4j")
    
    # 검색 테스트
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke("GRPO")
    
    print(f"\nFound {len(docs)} relevant chunks:")
    for i, doc in enumerate(docs, 1):
        print(f"\n[{i}] Page {doc.metadata.get('page_idx', 'N/A')}")
        print(f"    {doc.page_content[:100]}...")


# ============================================================================
# 메인 실행
# ============================================================================

if __name__ == "__main__":
    # 원하는 예시만 실행
    # example_neoj4_vector_store()
    # example_neo4j_graph()
    # example_graph_cypher_qa()
    example_pdf_to_neo4j()

