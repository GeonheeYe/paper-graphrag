import os
from typing import List, Dict, Any, Tuple

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class VectorRepo:
    COLLECTION_NAME = "pdf"
    
    def __init__(self, chroma_dir: str, embed_model: str):
        os.makedirs(chroma_dir, exist_ok=True)
        self.embeddings = HuggingFaceEmbeddings(model_name=embed_model)
        self.vdb = Chroma(
            collection_name = self.COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=chroma_dir, 
        )
    def upsert_chunks(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> int:
        self.vdb.add_texts(texts=texts, metadatas=metadatas)

    def similarity_search(self, doc_id: str, query: str, top_k: int) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Returns list of (text, metadata, score)
        """
        results = self.vdb.similarity_search_with_score(
            query=query,
            k=top_k,
            filter={"doc_id": doc_id},
        )
        out = []
        for doc, score in results:
            out.append((doc.page_content, doc.metadata, float(score)))
        return out

    def count_by_doc(self, doc_id: str) -> int:
        """특정 doc_id로 저장된 청크 개수를 반환"""
        try:
            # Chroma의 get 메서드로 필터링하여 개수 확인
            results = self.vdb.get(where={"doc_id": doc_id})
            return len(results.get('ids', [])) if results else 0
        except Exception:
            # fallback: similarity_search로 확인
            res = self.vdb.similarity_search(query="the", k=1, filter={"doc_id": doc_id})
            return 1 if res else 0
    
    def doc_exists(self, doc_id: str) -> bool:
        """문서가 이미 인덱싱되어 있는지 확인"""
        return self.count_by_doc(doc_id) > 0