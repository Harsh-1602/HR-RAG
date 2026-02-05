"""
RAG Pipeline Module
Core ingestion and retrieval logic with dual-filter system using LangChain.
Supports document chunking for large files.
"""

import os
from pathlib import Path
from typing import List, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import (
    DocumentMetadata,
    RetrievedDocument,
    RetrievedChunk,
    RetrievalResult,
    RAGResponse
)
from metadata_extractor import extract_metadata
from query_classifier import classify_query


class RAGPipeline:
    """
    RAG Pipeline with dual-filter retrieval:
    1. Filter by document type (policy/cafeteria/general)
    2. Filter by recency (most recent effective_date wins)
    
    Supports chunking for large documents.
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "techcorp_docs",
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between consecutive chunks
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize embedding model (local, free)
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize text splitter for chunking
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Initialize ChromaDB with persistence
        self.chroma_client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "TechCorp document knowledge base"}
        )
        
        # Initialize LangChain LLM
        self.llm = self._get_llm()
    
    def _get_llm(self, model: str = "llama-3.3-70b-versatile") -> ChatGroq:
        """Initialize Groq LLM via LangChain."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        return ChatGroq(
            api_key=api_key,
            model_name=model,
            temperature=0.3
        )
    
    def _embed_text(self, text: str) -> List[float]:
        """Generate embedding for text."""
        return self.embedding_model.encode(text).tolist()
    
    def _delete_document_chunks(self, filename: str):
        """Delete all chunks associated with a document."""
        # Get all IDs that start with the filename
        all_docs = self.collection.get()
        ids_to_delete = [
            doc_id for doc_id in all_docs["ids"]
            if doc_id == filename or doc_id.startswith(f"{filename}_chunk_")
        ]
        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            print(f"Deleted {len(ids_to_delete)} existing chunk(s) for {filename}")
    
    def ingest_document(
        self,
        content: str,
        filename: str,
        upload_timestamp: Optional[datetime] = None
    ) -> DocumentMetadata:
        """
        Ingest a document into the vector store with chunking support.
        
        For small documents: stored as single chunk
        For large documents: split into multiple chunks with same metadata
        
        Args:
            content: Document text content
            filename: Name of the file
            upload_timestamp: Optional upload timestamp
        
        Returns:
            DocumentMetadata for the ingested document
        """
        # Delete existing chunks for this document
        self._delete_document_chunks(filename)
        
        # Extract metadata using LLM (uses truncated content internally)
        metadata: DocumentMetadata = extract_metadata(content, filename, upload_timestamp)
        
        # Split content into chunks
        chunks = self.text_splitter.split_text(content)
        
        # If document is small enough, store as single chunk
        if len(chunks) == 0:
            chunks = [content]
        
        # Prepare data for batch insertion
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            # Create unique ID for each chunk
            chunk_id = filename if len(chunks) == 1 else f"{filename}_chunk_{i}"
            
            ids.append(chunk_id)
            embeddings.append(self._embed_text(chunk))
            documents.append(chunk)
            metadatas.append({
                "doc_type": metadata.doc_type,
                "effective_date": metadata.effective_date,
                "effective_date_str": metadata.effective_date_str,
                "filename": metadata.filename,  # Same filename for all chunks
                "chunk_index": i,
                "total_chunks": len(chunks),
                "summary": metadata.summary,
                "tags": ",".join(metadata.tags) if metadata.tags else ""
            })
        
        # Batch insert all chunks
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        
        chunk_info = f"{len(chunks)} chunk(s)" if len(chunks) > 1 else "1 chunk"
        print(f"Ingested: {filename} → {chunk_info} (type: {metadata.doc_type}, date: {metadata.effective_date_str})")
        return metadata
    
    def ingest_directory(self, directory_path: str) -> List[DocumentMetadata]:
        """
        Ingest all .txt files from a directory.
        
        Args:
            directory_path: Path to directory containing documents
        
        Returns:
            List of DocumentMetadata for ingested documents
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory not found: {directory_path}")
        
        results = []
        for file_path in directory.glob("*.txt"):
            content = file_path.read_text(encoding="utf-8")
            metadata = self.ingest_document(content, file_path.name)
            results.append(metadata)
        
        print(f"\nIngested {len(results)} documents from {directory_path}")
        return results
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> RetrievalResult:
        """
        Retrieve the most relevant and recent document using dual-filter.
        
        Filter 1: By document type (based on query classification)
        Filter 2: By recency (return only the most recent document)
        
        Handles chunked documents by grouping chunks by filename.
        
        Args:
            query: User query
            top_k: Number of chunk candidates to retrieve before grouping
        
        Returns:
            RetrievalResult with the most recent relevant document
        """
        # Step 1: Classify query
        query_class = classify_query(query)
        print(f"Query classified as: {query_class}")
        
        # Step 2: Generate query embedding
        query_embedding = self._embed_text(query)
        
        # Step 3: Retrieve with document type filter
        if query_class == "general":
            # For general queries, search all documents
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
        else:
            # Filter by document type
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"doc_type": query_class},
                include=["documents", "metadatas", "distances"]
            )
        
        # Check if any results
        if not results["ids"] or not results["ids"][0]:
            return RetrievalResult(
                query_classification=query_class,
                retrieved_doc=None,
                all_candidates=[],
                raw_chunks_from_db=[],
                chunks_passed_to_llm=[]
            )
        
        # Step 4: Build raw chunks list for debug
        raw_chunks_from_db: List[RetrievedChunk] = []
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i] if results["distances"] else 0
            raw_chunks_from_db.append(RetrievedChunk(
                chunk_id=doc_id,
                content=results["documents"][0][i],
                filename=meta["filename"],
                chunk_index=meta.get("chunk_index", 0),
                total_chunks=meta.get("total_chunks", 1),
                doc_type=meta["doc_type"],
                effective_date_str=meta["effective_date_str"],
                distance=distance
            ))
        
        # Step 5: Group chunks by filename and aggregate
        # This handles both single-chunk and multi-chunk documents
        doc_groups = {}  # filename -> {chunks, chunk_indices, metadata, best_distance}
        
        for i, doc_id in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i]
            filename = meta["filename"]
            chunk_content = results["documents"][0][i]
            distance = results["distances"][0][i] if results["distances"] else 0
            
            if filename not in doc_groups:
                doc_groups[filename] = {
                    "chunks": {},
                    "chunk_indices": [],
                    "metadata": meta,
                    "best_distance": distance
                }
            
            # Store chunk by index for proper ordering
            chunk_index = meta.get("chunk_index", 0)
            doc_groups[filename]["chunks"][chunk_index] = chunk_content
            doc_groups[filename]["chunk_indices"].append(chunk_index)
            
            # Track best (lowest) distance for this document
            if distance < doc_groups[filename]["best_distance"]:
                doc_groups[filename]["best_distance"] = distance
        
        # Step 6: Build candidate list with reassembled content
        candidates: List[RetrievedDocument] = []
        for filename, group in doc_groups.items():
            meta = group["metadata"]
            
            # Reassemble chunks in order
            sorted_indices = sorted(group["chunks"].keys())
            sorted_chunks = [group["chunks"][idx] for idx in sorted_indices]
            full_content = "\n\n".join(sorted_chunks)
            
            candidates.append(RetrievedDocument(
                id=filename,
                content=full_content,
                metadata=DocumentMetadata(
                    doc_type=meta["doc_type"],
                    effective_date=meta["effective_date"],
                    effective_date_str=meta["effective_date_str"],
                    filename=meta["filename"],
                    summary=meta.get("summary", ""),
                    tags=meta.get("tags", "").split(",") if meta.get("tags") else []
                ),
                distance=group["best_distance"],
                chunks_used=sorted_indices
            ))
        
        # Step 7: Sort by effective_date (descending) and return the most recent
        candidates.sort(
            key=lambda x: x.metadata.effective_date,
            reverse=True
        )
        
        most_recent = candidates[0] if candidates else None
        
        # Step 8: Build chunks_passed_to_llm from the most recent document
        chunks_passed_to_llm: List[RetrievedChunk] = []
        if most_recent:
            # Find chunks that belong to the selected document
            for chunk in raw_chunks_from_db:
                if chunk.filename == most_recent.metadata.filename:
                    chunks_passed_to_llm.append(chunk)
            # Sort by chunk_index
            chunks_passed_to_llm.sort(key=lambda x: x.chunk_index)
        
        return RetrievalResult(
            query_classification=query_class,
            retrieved_doc=most_recent,
            all_candidates=candidates,
            raw_chunks_from_db=raw_chunks_from_db,
            chunks_passed_to_llm=chunks_passed_to_llm
        )
    
    def generate_answer(
        self,
        query: str,
        retrieved_doc: Optional[RetrievedDocument]
    ) -> str:
        """
        Generate an answer using LangChain + Groq LLM with citation.
        
        Args:
            query: User query
            retrieved_doc: The retrieved document with metadata
        
        Returns:
            Answer string with citation
        """
        if not retrieved_doc:
            return "I couldn't find any relevant documents to answer your question."
        
        filename = retrieved_doc.metadata.filename
        content = retrieved_doc.content
        effective_date = retrieved_doc.metadata.effective_date_str
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful HR assistant for TechCorp Inc. Always cite your sources."),
            ("human", """Answer the employee's question based ONLY on the provided document.

IMPORTANT RULES:
1. Base your answer ONLY on the provided document content
2. You MUST cite the source filename at the end of your answer
3. If the document doesn't contain enough information, say so
4. Be concise and direct

Document Source: {filename}
Document Effective Date: {effective_date}

Document Content:
---
{content}
---

Employee Question: {query}

Provide a helpful answer and end with "Source: {filename}" """)
        ])
        
        chain = prompt | self.llm | StrOutputParser()
        
        try:
            return chain.invoke({
                "filename": filename,
                "effective_date": effective_date,
                "content": content,
                "query": query
            })
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def query(self, user_query: str) -> RAGResponse:
        """
        Full RAG pipeline: retrieve + generate.
        
        Args:
            user_query: The user's question
        
        Returns:
            RAGResponse with answer and metadata
        """
        # Retrieve
        retrieval_result = self.retrieve(user_query)
        
        # Generate answer
        answer = self.generate_answer(user_query, retrieval_result.retrieved_doc)
        
        return RAGResponse(
            query=user_query,
            query_classification=retrieval_result.query_classification,
            answer=answer,
            source=retrieval_result.retrieved_doc.metadata.filename if retrieval_result.retrieved_doc else None,
            retrieved_doc=retrieval_result.retrieved_doc,
            all_candidates=retrieval_result.all_candidates,
            raw_chunks_from_db=retrieval_result.raw_chunks_from_db,
            chunks_passed_to_llm=retrieval_result.chunks_passed_to_llm
        )
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        self.chroma_client.delete_collection(self.collection_name)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "TechCorp document knowledge base"}
        )
        print("Collection cleared.")
    
    def list_documents(self) -> List[DocumentMetadata]:
        """
        List all unique documents in the collection.
        Groups chunks by filename to return one entry per document.
        """
        results = self.collection.get(include=["metadatas"])
        
        # Group by filename to handle chunked documents
        seen_filenames = {}
        for i, doc_id in enumerate(results["ids"]):
            meta = results["metadatas"][i]
            filename = meta["filename"]
            
            # Only keep the first occurrence (all chunks have same metadata)
            if filename not in seen_filenames:
                seen_filenames[filename] = DocumentMetadata(
                    doc_type=meta["doc_type"],
                    effective_date=meta["effective_date"],
                    effective_date_str=meta["effective_date_str"],
                    filename=meta["filename"],
                    summary=meta.get("summary", ""),
                    tags=meta.get("tags", "").split(",") if meta.get("tags") else []
                )
        
        return list(seen_filenames.values())


def main():
    """Main function to demonstrate the RAG pipeline."""
    from dotenv import load_dotenv
    load_dotenv()
    
    print("=" * 60)
    print("TechCorp RAG Pipeline Demo")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Clear and re-ingest for demo
    pipeline.clear_collection()
    
    # Ingest documents
    print("\n📥 Ingesting documents from knowledge_base/...")
    pipeline.ingest_directory("./knowledge_base")
    
    # List ingested documents
    print("\n📋 Ingested documents:")
    for doc in pipeline.list_documents():
        print(f"  - {doc.filename} (type: {doc.doc_type}, date: {doc.effective_date_str})")
    
    # Test query
    test_query = "Can I work fully remotely this Friday?"
    
    print(f"\n❓ Query: {test_query}")
    print("-" * 60)
    
    result = pipeline.query(test_query)
    
    print(f"\n🏷️  Query Classification: {result.query_classification}")
    print(f"\n📄 Retrieved Source: {result.source}")
    print(f"\n💬 Answer:\n{result.answer}")
    
    # Show all candidates for transparency
    print(f"\n🔍 All candidates considered (before recency filter):")
    for i, candidate in enumerate(result.all_candidates):
        print(f"  {i+1}. {candidate.metadata.filename} (date: {candidate.metadata.effective_date_str}, distance: {candidate.distance:.4f})")


if __name__ == "__main__":
    main()
