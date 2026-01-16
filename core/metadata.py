from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class RAGMetadata:
    paper_id: Optional[str] = None
    doc_id: Optional[str] = None
    page_idx: Optional[int] = None
    block_idx: Optional[int] = None
    table_idx: Optional[int] = None
    figure_idx: Optional[int] = None
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GraphMetadata:
    paper_id: Optional[str] = None
    doc_id: Optional[str] = None
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