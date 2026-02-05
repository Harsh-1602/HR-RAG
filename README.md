# TechCorp RAG System - Conflicting Policy Handler

A production-ready Retrieval-Augmented Generation (RAG) system that handles conflicting HR policies by using intelligent document classification and recency-based filtering.

## 🎯 Problem Statement

TechCorp's HR department frequently updates policies without deleting old versions, leading to a knowledge base with conflicting information. This RAG system:

1. **Handles Conflicts**: Automatically identifies and uses the most recent policy version
2. **Filters Noise**: Ignores irrelevant documents (like cafeteria menus) even if they share keywords
3. **Cites Sources**: Always provides source attribution for transparency

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER QUERY                               │
│              "Can I work fully remotely this Friday?"            │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QUERY CLASSIFIER                              │
│                    (Groq LLaMA 3.1 8B)                          │
│                                                                  │
│  Classifies query → "policy" | "cafeteria" | "general"          │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FILTER 1: CLASSIFICATION                      │
│                                                                  │
│  ChromaDB query with: where={"doc_type": "policy"}              │
│  → Excludes cafeteria menu despite "Friday" keyword match       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FILTER 2: RECENCY                             │
│                                                                  │
│  Sort by effective_date DESC → Return ONLY most recent doc      │
│  → Selects policy_v2_2024.txt, ignores policy_v1_2021.txt       │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ANSWER GENERATION                             │
│                    (Groq LLaMA 3.1 70B)                         │
│                                                                  │
│  Generates answer with mandatory source citation                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         RESPONSE                                 │
│  "No, fully remote work is revoked as of Jan 2024. Remote work  │
│   is capped at 1 day per week with manager approval.            │
│   Source: policy_v2_2024.txt"                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **LLM (Classification/Extraction)** | Groq LLaMA 3.1 8B Instant |
| **LLM (Answer Generation)** | Groq LLaMA 3.3 70B Versatile |
| **Embeddings** | Sentence-Transformers (all-MiniLM-L6-v2) |
| **Vector Database** | ChromaDB (persistent) |
| **UI Framework** | Streamlit |
| **Language** | Python 3.10+ |

## 📁 Project Structure

```
rag/
├── knowledge_base/
│   ├── policy_v1_2021.txt      # Outdated WFH policy
│   ├── policy_v2_2024.txt      # Current RTO mandate
│   └── friday_cafeteria_menu.txt  # Noise document
├── app.py                      # Streamlit UI with debug panel
├── rag_pipeline.py             # Core RAG logic with chunk tracking
├── metadata_extractor.py       # LLM-based doc classification
├── query_classifier.py         # Query intent classification
├── models.py                   # Pydantic models for type safety
├── requirements.txt            # Dependencies
├── .env.example                # Environment variable template
└── README.md                   # This file
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example env file
cp .env.example .env

# Edit .env and add your Groq API key
# Get your free API key at: https://console.groq.com/
```

### 3. Run the Application

**Option A: Streamlit UI (Recommended)**
```bash
streamlit run app.py
```

**Option B: Command Line Demo**
```bash
python rag_pipeline.py
```

## 📖 Usage

### Streamlit UI

1. Open the app in your browser (usually http://localhost:8501)
2. Click **"Load Default Knowledge Base"** in the sidebar to ingest the test documents
3. Ask questions in the chat input, e.g., *"Can I work fully remotely this Friday?"*
4. View the answer with source citations
5. Enable **Debug Mode** to see the dual-filter process in action

### Uploading Custom Documents

1. Use the file uploader in the sidebar
2. The system automatically:
   - Classifies the document type (policy/cafeteria/general)
   - Extracts the effective date from filename or uses upload time
   - Generates a summary and tags
3. The document is immediately available for querying

## 📊 Example Output

**Query:** "Can I work fully remotely this Friday?"

**Answer:**
> No, you cannot work fully remotely this Friday. According to the current policy, remote work is capped at 1 day per week and must be approved by a manager. The 100% remote work policy from 2021 has been officially revoked, and employees are expected to be in the office 4 days a week.
> 
> **Source: policy_v2_2024.txt**

**Debug Info:**
- Query Classification: `policy`
- Candidates Retrieved: `policy_v1_2021.txt`, `policy_v2_2024.txt`
- Selected (most recent): `policy_v2_2024.txt`
- Noise Filtered Out: `friday_cafeteria_menu.txt`

## 🔍 Debug Mode - Chunk Visibility Feature

The system includes a comprehensive **Debug Mode** that provides full visibility into the RAG retrieval process. This is essential for understanding how documents are chunked, retrieved, and filtered.

### Enabling Debug Mode

Debug Mode is **enabled by default**. You can toggle it via the checkbox in the sidebar: **🔧 Debug Mode**

### Debug Panel Sections

When Debug Mode is enabled, after each query you'll see an expanded debug panel with four sections:

#### 1. 📦 Raw Chunks from DB

Shows **all individual chunks** retrieved from ChromaDB before any grouping or filtering:

| Field | Description |
|-------|-------------|
| **Chunk ID** | Unique identifier for the chunk |
| **Filename** | Source document name |
| **Chunk Index** | Position in the original document (e.g., "Chunk 2/5") |
| **Distance** | Cosine distance from query (lower = more relevant) |
| **Type** | Document classification (policy/cafeteria/general) |
| **Date** | Effective date of the document |
| **Content** | Expandable preview of chunk text |

#### 2. 📄 All Candidate Documents (Grouped)

Shows documents **after chunks are grouped by filename**:

- Chunks from the same document are reassembled
- Shows which chunk indices were used per document
- The selected document (most recent) is marked with ✅
- Displays aggregated distance (best chunk distance)

#### 3. 🚀 Chunks Passed to LLM

Shows the **exact chunks sent to the LLM** for answer generation:

- Only chunks from the selected (most recent) document
- Sorted by chunk index for proper context ordering
- Full content visible in expandable sections
- This is what the LLM actually "sees"

#### 4. 📝 Selected Document

Confirms the **final document selection** after dual-filter:

- Filename of the document used for answer generation
- Effective date that determined its selection

### Why Debug Mode Matters

| Use Case | Benefit |
|----------|--------|
| **Debugging retrieval issues** | See exactly which chunks matched and why |
| **Understanding chunking** | Verify how large documents are split |
| **Validating recency filter** | Confirm the correct document is selected |
| **Analyzing distances** | Understand relevance scoring |
| **Auditing LLM context** | See exactly what information the LLM receives |

### Example Debug Output

```
📦 Raw Chunks from DB (3 chunks)
├── Chunk 1: policy_v2_2024.txt (1/2) - Distance: 0.3421
├── Chunk 2: policy_v2_2024.txt (2/2) - Distance: 0.5123
└── Chunk 3: policy_v1_2021.txt (1/1) - Distance: 0.4532

📄 All Candidates (2 documents)
├── ✅ policy_v2_2024.txt - Date: 2024-01-15 - Chunks: [1, 2]
└── policy_v1_2021.txt - Date: 2021-03-01 - Chunks: [1]

🚀 Chunks Passed to LLM (2 chunks)
├── policy_v2_2024.txt - Chunk 1/2
└── policy_v2_2024.txt - Chunk 2/2

📝 Selected: policy_v2_2024.txt (Effective: 2024-01-15)
```

## 🧪 How It Solves the Key Challenges

### 1. Handling Conflicts ✅

The system uses **effective_date** metadata (extracted from filename year or upload timestamp) to always select the most recent document when multiple policies match. The 2024 policy automatically overrides the 2021 policy.

### 2. Noise Filtering ✅

The dual-filter approach ensures:
- **Filter 1**: Query classification routes policy questions to policy documents only
- The cafeteria menu is classified as `doc_type: "cafeteria"` and excluded from policy queries

### 3. Source Citations ✅

The LLM prompt explicitly requires source attribution. Every answer ends with the source filename, enabling human verification.

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | Your Groq API key | Yes |

### Pipeline Parameters

In `rag_pipeline.py`, you can configure:

```python
RAGPipeline(
    persist_directory="./chroma_db",  # ChromaDB storage location
    collection_name="techcorp_docs"   # Collection name
)
```

## 📝 API Reference

### RAGPipeline Class

```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

# Ingest a document
metadata = pipeline.ingest_document(content, filename)

# Ingest a directory
results = pipeline.ingest_directory("./knowledge_base")

# Query the system
result = pipeline.query("Can I work remotely?")
print(result["answer"])
print(result["source"])
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure code quality
5. Submit a pull request

## 📄 License

MIT License

---

Built for the SquareYards RAG Assignment
