"""
Streamlit App for TechCorp RAG Chatbot
File upload, chat interface, and metadata display.
"""

import os
import streamlit as st
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from rag_pipeline import RAGPipeline
from metadata_extractor import extract_metadata


# Page configuration
st.set_page_config(
    page_title="TechCorp HR Assistant",
    page_icon="🏢",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #F0F9FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        margin-top: 1rem;
        border-radius: 0 8px 8px 0;
    }
    .metadata-box {
        background-color: #F0FDF4;
        border: 1px solid #86EFAC;
        padding: 0.75rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .filter-info {
        background-color: #FEF3C7;
        border: 1px solid #FCD34D;
        padding: 0.5rem;
        border-radius: 8px;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    """Initialize and cache the RAG pipeline."""
    return RAGPipeline()


def initialize_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents" not in st.session_state:
        st.session_state.documents = []


def display_chat_message(role: str, content: str, metadata: dict = None):
    """Display a chat message with optional metadata."""
    with st.chat_message(role):
        st.markdown(content)
        if metadata:
            with st.expander("📋 View Source Details"):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Source:** `{metadata.get('filename', 'N/A')}`")
                    st.markdown(f"**Type:** `{metadata.get('doc_type', 'N/A')}`")
                with col2:
                    st.markdown(f"**Effective Date:** `{metadata.get('effective_date_str', 'N/A')}`")
                    st.markdown(f"**Query Class:** `{metadata.get('query_class', 'N/A')}`")


def main():
    """Main Streamlit application."""
    initialize_session_state()
    pipeline = get_pipeline()
    
    # Header
    st.markdown('<p class="main-header">🏢 TechCorp HR Assistant</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about company policies and get accurate, cited answers.</p>', unsafe_allow_html=True)
    
    # Sidebar - Document Management
    with st.sidebar:
        st.header("📁 Document Management")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["txt"],
            help="Upload .txt files to add to the knowledge base"
        )
        
        if uploaded_file:
            content = uploaded_file.read().decode("utf-8")
            filename = uploaded_file.name
            
            with st.spinner("Analyzing document..."):
                try:
                    metadata = pipeline.ingest_document(
                        content=content,
                        filename=filename,
                        upload_timestamp=datetime.now()
                    )
                    
                    st.success(f"✅ Uploaded: {filename}")
                    
                    st.markdown('<div class="metadata-box">', unsafe_allow_html=True)
                    st.markdown(f"**Type:** `{metadata.doc_type}`")
                    st.markdown(f"**Date:** `{metadata.effective_date_str}`")
                    st.markdown(f"**Summary:** {metadata.summary}")
                    if metadata.tags:
                        st.markdown(f"**Tags:** {', '.join(metadata.tags)}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error uploading: {e}")
        
        st.divider()
        
        # Load default documents button
        if st.button("📥 Load Default Knowledge Base", use_container_width=True):
            with st.spinner("Loading knowledge base..."):
                try:
                    kb_path = Path("./knowledge_base")
                    if kb_path.exists():
                        pipeline.clear_collection()
                        results = pipeline.ingest_directory(str(kb_path))
                        st.success(f"Loaded {len(results)} documents!")
                    else:
                        st.error("knowledge_base folder not found")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        st.divider()
        
        # Document list
        st.subheader("📋 Indexed Documents")
        docs = pipeline.list_documents()
        
        if docs:
            for doc in docs:
                with st.container():
                    st.markdown(f"**{doc.filename}**")
                    st.caption(f"Type: {doc.doc_type} | Date: {doc.effective_date_str}")
        else:
            st.info("No documents indexed yet. Upload files or load the default knowledge base.")
        
        st.divider()
        
        # Clear collection button
        if st.button("🗑️ Clear All Documents", use_container_width=True):
            pipeline.clear_collection()
            st.session_state.messages = []
            st.success("Collection cleared!")
            st.rerun()
        
        st.divider()
        
        # Debug mode toggle (enabled by default)
        st.session_state.debug_mode = st.checkbox("🔧 Debug Mode", value=True)
    
    # Main chat area
    st.divider()
    
    # Display chat history
    for message in st.session_state.messages:
        display_chat_message(
            message["role"],
            message["content"],
            message.get("metadata")
        )
    
    # Chat input
    if prompt := st.chat_input("Ask a question about company policies..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        
        # Generate response
        with st.spinner("Searching documents and generating answer..."):
            try:
                result = pipeline.query(prompt)
                
                answer = result.answer
                
                # Build metadata for display
                metadata = {
                    "query_class": result.query_classification,
                    "filename": result.source,
                    "doc_type": result.retrieved_doc.metadata.doc_type if result.retrieved_doc else None,
                    "effective_date_str": result.retrieved_doc.metadata.effective_date_str if result.retrieved_doc else None
                }
                
                # Add assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": metadata
                })
                
                display_chat_message("assistant", answer, metadata)
                
                # Debug info
                if st.session_state.get("debug_mode"):
                    with st.expander("🔍 Debug: Retrieval Details", expanded=True):
                        st.markdown("### Query Classification")
                        st.code(result.query_classification)
                        
                        st.markdown("---")
                        st.markdown("### 📦 Raw Chunks from DB")
                        st.caption(f"Total chunks retrieved: {len(result.raw_chunks_from_db)}")
                        
                        if result.raw_chunks_from_db:
                            for i, chunk in enumerate(result.raw_chunks_from_db):
                                with st.container():
                                    st.markdown(f"**Chunk {i+1}** - `{chunk.filename}` (Chunk {chunk.chunk_index + 1}/{chunk.total_chunks})")
                                    col1, col2, col3 = st.columns(3)
                                    col1.metric("Distance", f"{chunk.distance:.4f}")
                                    col2.metric("Type", chunk.doc_type)
                                    col3.metric("Date", chunk.effective_date_str)
                                    with st.expander("View content", expanded=False):
                                        st.text(chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content)
                        else:
                            st.info("No chunks retrieved from DB")
                        
                        st.markdown("---")
                        st.markdown("### 📄 All Candidate Documents (grouped by file)")
                        st.caption("Chunks are grouped by filename and reassembled")
                        
                        for i, candidate in enumerate(result.all_candidates):
                            marker = "✅ " if result.retrieved_doc and candidate.metadata.filename == result.retrieved_doc.metadata.filename else ""
                            st.markdown(f"{marker}**{i+1}. `{candidate.metadata.filename}`**")
                            col1, col2, col3 = st.columns(3)
                            col1.caption(f"Type: `{candidate.metadata.doc_type}`")
                            col2.caption(f"Date: `{candidate.metadata.effective_date_str}`")
                            col3.caption(f"Distance: `{candidate.distance:.4f}`")
                            if candidate.chunks_used:
                                st.caption(f"Chunks used: {[c + 1 for c in candidate.chunks_used]}")
                        
                        st.markdown("---")
                        st.markdown("### 🚀 Chunks Passed to LLM")
                        st.caption(f"From selected document: `{result.retrieved_doc.metadata.filename if result.retrieved_doc else 'None'}`")
                        
                        if result.chunks_passed_to_llm:
                            for i, chunk in enumerate(result.chunks_passed_to_llm):
                                with st.container():
                                    st.markdown(f"**Chunk {chunk.chunk_index + 1}/{chunk.total_chunks}**")
                                    with st.expander("View content", expanded=False):
                                        st.text(chunk.content)
                        else:
                            st.info("No chunks passed to LLM")
                        
                        st.markdown("---")
                        st.markdown("### 📝 Selected Document (most recent)")
                        if result.retrieved_doc:
                            st.success(f"**{result.retrieved_doc.metadata.filename}** (Effective: {result.retrieved_doc.metadata.effective_date_str})")
                        else:
                            st.warning("No document selected")
                
            except Exception as e:
                error_msg = f"Error: {e}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                display_chat_message("assistant", error_msg)
    
    # Footer with instructions
    st.divider()
    
    # Features Overview Section
    with st.expander("🚀 Features Overview", expanded=False):
        st.markdown("""
        ### Key Features
        
        | Feature | Description |
        |---------|-------------|
        | **🔍 Dual-Filter Retrieval** | Classification + Recency filtering for accurate answers |
        | **📄 Smart Document Chunking** | Large documents are split into 500-char chunks with overlap |
        | **🤖 LLM-Based Metadata** | Document type & dates extracted automatically via LLaMA |
        | **📊 Debug Mode** | Full visibility into chunks retrieved vs. passed to LLM |
        | **📝 Source Citations** | Every answer includes source document attribution |
        | **📁 File Upload** | Add your own .txt documents to the knowledge base |
        
        ### Tech Stack
        
        - **LLM**: Groq (LLaMA 3.1 8B for classification, LLaMA 3.3 70B for generation)
        - **Embeddings**: Sentence-Transformers (all-MiniLM-L6-v2)
        - **Vector DB**: ChromaDB with persistence
        - **Framework**: LangChain + Pydantic
        """)
    
    with st.expander("ℹ️ How the Dual-Filter Works"):
        st.markdown("""
        ### Dual-Filter RAG System
        
        This system uses a two-stage filtering approach:
        
        ```
        Query → [Classify] → Filter by Type → [Retrieve Chunks] → Group by File → Filter by Recency → [Generate Answer]
        ```
        
        **Filter 1 - Classification**
        - Your query is classified as: `policy` | `cafeteria` | `general`
        - Only documents matching the classification are retrieved
        - Example: Policy questions won't retrieve cafeteria menus
        
        **Filter 2 - Recency**
        - Among matching documents, only the **most recent** one is used
        - Based on effective date extracted from filename or upload time
        - Example: 2024 policy overrides 2021 policy
        
        This ensures:
        - ✅ Noise documents (like cafeteria menus) are filtered out
        - ✅ Newer policies automatically override older ones
        - ✅ All answers include source citations
        """)
    
    with st.expander("🔧 Debug Mode Guide"):
        st.markdown("""
        ### Understanding Debug Mode
        
        Enable **Debug Mode** in the sidebar to see the full retrieval process after each query.
        
        #### 📦 Raw Chunks from DB
        Shows all individual chunks retrieved from ChromaDB:
        - **Chunk ID**: Unique identifier
        - **Distance**: Cosine distance (lower = more relevant)
        - **Chunk Index**: Position in original document (e.g., "Chunk 2/5")
        
        #### 📄 All Candidate Documents
        Shows documents after chunks are grouped by filename:
        - Multiple chunks from the same file are reassembled
        - ✅ marks the selected document (most recent)
        
        #### 🚀 Chunks Passed to LLM
        Shows the exact content sent to the LLM for answer generation:
        - Only chunks from the selected (most recent) document
        - This is what the AI actually "sees"
        
        #### 📝 Selected Document
        Confirms which document was used for the final answer.
        """)
    
    with st.expander("💡 Example Queries"):
        st.markdown("""
        Try these sample questions:
        
        | Query | Expected Behavior |
        |-------|-------------------|
        | *"Can I work fully remotely this Friday?"* | Returns 2024 RTO policy (not 2021 WFH policy) |
        | *"What's on the menu this Friday?"* | Returns cafeteria menu (filters out policies) |
        | *"What are the company policies?"* | Returns most recent policy document |
        | *"Tell me about TechCorp"* | General query - searches all documents |
        """)


if __name__ == "__main__":
    main()
