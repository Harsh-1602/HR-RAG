"""
Pydantic Models for the RAG Pipeline
Type-safe data structures for documents, metadata, and responses.
"""

from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    """Metadata extracted from a document."""
    doc_type: Literal["policy", "cafeteria", "general"] = Field(
        description="Document classification type"
    )
    effective_date: int = Field(
        description="Unix timestamp for sorting by recency"
    )
    effective_date_str: str = Field(
        description="Human-readable date string (YYYY-MM-DD)"
    )
    filename: str = Field(
        description="Original filename of the document"
    )
    summary: str = Field(
        description="Brief summary of the document content"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Relevant keyword tags"
    )


class DocumentClassification(BaseModel):
    """LLM output for document classification."""
    doc_type: Literal["policy", "cafeteria", "general"] = Field(
        description="Document type: policy, cafeteria, or general"
    )
    summary: str = Field(
        description="One-sentence summary of the document (max 50 words)"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="3-5 relevant keyword tags"
    )


class QueryClassification(BaseModel):
    """LLM output for query classification."""
    category: Literal["policy", "cafeteria", "general"] = Field(
        description="Query category: policy, cafeteria, or general"
    )
    reasoning: str = Field(
        description="Brief explanation of why this category was chosen"
    )


class RetrievedChunk(BaseModel):
    """A single chunk retrieved from the vector store (raw DB result)."""
    chunk_id: str = Field(description="Chunk ID (e.g., filename_chunk_0)")
    content: str = Field(description="Chunk content")
    filename: str = Field(description="Parent document filename")
    chunk_index: int = Field(default=0, description="Index of this chunk in the document")
    total_chunks: int = Field(default=1, description="Total chunks in parent document")
    doc_type: str = Field(description="Document type")
    effective_date_str: str = Field(description="Effective date")
    distance: float = Field(description="Vector similarity distance")


class RetrievedDocument(BaseModel):
    """A document retrieved from the vector store (reassembled from chunks)."""
    id: str = Field(description="Document ID (usually filename)")
    content: str = Field(description="Full document content (reassembled chunks)")
    metadata: DocumentMetadata = Field(description="Document metadata")
    distance: Optional[float] = Field(
        default=None,
        description="Vector similarity distance (lower is more similar)"
    )
    chunks_used: List[int] = Field(
        default_factory=list,
        description="List of chunk indices that were retrieved and used"
    )


class RetrievalResult(BaseModel):
    """Result of the retrieval step."""
    query_classification: str = Field(
        description="How the query was classified"
    )
    retrieved_doc: Optional[RetrievedDocument] = Field(
        default=None,
        description="The single most recent relevant document"
    )
    all_candidates: List[RetrievedDocument] = Field(
        default_factory=list,
        description="All candidate documents before recency filter"
    )
    raw_chunks_from_db: List[RetrievedChunk] = Field(
        default_factory=list,
        description="Raw chunks retrieved from ChromaDB before grouping"
    )
    chunks_passed_to_llm: List[RetrievedChunk] = Field(
        default_factory=list,
        description="Chunks that were passed to LLM (from selected document)"
    )


class RAGResponse(BaseModel):
    """Complete RAG pipeline response."""
    query: str = Field(description="Original user query")
    query_classification: str = Field(description="Query classification result")
    answer: str = Field(description="Generated answer with citation")
    source: Optional[str] = Field(
        default=None,
        description="Source filename used for the answer"
    )
    retrieved_doc: Optional[RetrievedDocument] = Field(
        default=None,
        description="The document used to generate the answer"
    )
    all_candidates: List[RetrievedDocument] = Field(
        default_factory=list,
        description="All candidate documents considered"
    )
    raw_chunks_from_db: List[RetrievedChunk] = Field(
        default_factory=list,
        description="Raw chunks retrieved from ChromaDB (debug info)"
    )
    chunks_passed_to_llm: List[RetrievedChunk] = Field(
        default_factory=list,
        description="Chunks that were passed to LLM (debug info)"
    )


class ChatMessage(BaseModel):
    """A single chat message."""
    role: Literal["user", "assistant"] = Field(description="Message role")
    content: str = Field(description="Message content")
    metadata: Optional[dict] = Field(
        default=None,
        description="Optional metadata (sources, classification, etc.)"
    )
