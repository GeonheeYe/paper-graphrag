from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class RAGMetadata:
    """
    doc_id : 논문 번호
    paper_name : 논문 제목
    page_idx : 페이지 번호
    block_idx : 블록 번호
    table_idx : 테이블 번호
    figure_idx : 그림 번호
    order : 전역 인덱스
    type : 타입
    """
    doc_id: Optional[str] = None
    paper_name: Optional[str] = None
    page_idx: Optional[int] = None
    block_idx: Optional[int] = None
    table_idx: Optional[int] = None
    figure_idx: Optional[int] = None
    order: Optional[int] = None 
    type: Optional[str] = None
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GraphMetadata:
    """
    doc_id : 논문 번호
    paper_name : 논문 제목
    page_idx : 페이지 번호
    block_idx : 블록 번호
    table_idx : 테이블 번호
    figure_idx : 그림 번호
    order : 전역 인덱스
    section : 섹션
    type : 타입
    """
    doc_id: Optional[str] = None
    paper_name: Optional[str] = None
    page_idx: Optional[int] = None
    block_idx: Optional[int] = None
    table_idx: Optional[int] = None
    figure_idx: Optional[int] = None
    order: Optional[int] = None 
    section: Optional[str] = None
    type: Optional[str] = None
    def to_dict(self) -> dict:
        return asdict(self)


def make_rag_metadata(**kwargs) -> RAGMetadata:
    """
    RAGMetadata dataclass 생성 (필드만 덮어쓰기).
    """
    return RAGMetadata(**kwargs)


def make_graph_metadata(**kwargs) -> GraphMetadata:
    """
    GraphMetadata dataclass 생성.

    kind: "table", "figure", "normal_chunk"
    """
    return GraphMetadata(**kwargs)