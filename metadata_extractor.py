"""
Metadata Extractor Module
Extracts document classification and metadata using LangChain + Groq.
Handles large documents by truncating content for LLM classification.
"""

import os
import re
from datetime import datetime
from typing import Optional

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from models import DocumentMetadata, DocumentClassification


# Max characters to send to LLM for classification (≈ 2000 tokens)
MAX_CONTENT_FOR_CLASSIFICATION = 8000


def get_llm(model: str = "llama-3.1-8b-instant", temperature: float = 0.1) -> ChatGroq:
    """Initialize Groq LLM via LangChain."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY environment variable is not set")
    return ChatGroq(
        api_key=api_key,
        model_name=model,
        temperature=temperature
    )


def extract_year_from_filename(filename: str) -> Optional[int]:
    """
    Extract year from filename using regex.
    Examples: policy_v1_2021.txt -> 2021, policy_v2_2024.txt -> 2024
    """
    match = re.search(r'(\d{4})', filename)
    if match:
        year = int(match.group(1))
        if 1900 <= year <= 2100:
            return year
    return None


def _truncate_for_classification(content: str) -> str:
    """
    Truncate content for LLM classification while preserving context.
    
    Strategy: Take beginning + end of document (where metadata usually appears)
    - Beginning: titles, headers, document type indicators
    - End: effective dates, signatures, policy numbers
    """
    if len(content) <= MAX_CONTENT_FOR_CLASSIFICATION:
        return content
    
    # Take first 60% and last 40% of allowed content
    first_part_len = int(MAX_CONTENT_FOR_CLASSIFICATION * 0.6)
    last_part_len = MAX_CONTENT_FOR_CLASSIFICATION - first_part_len - 60  # 60 for separator
    
    first_part = content[:first_part_len]
    last_part = content[-last_part_len:]
    
    return f"{first_part}\n\n[... middle content truncated ({len(content) - MAX_CONTENT_FOR_CLASSIFICATION} chars) ...]\n\n{last_part}"


def extract_metadata(
    content: str,
    filename: str,
    upload_timestamp: Optional[datetime] = None
) -> DocumentMetadata:
    """
    Extract metadata from document content using LangChain + Groq.
    
    Args:
        content: Document text content
        filename: Name of the file
        upload_timestamp: Optional upload timestamp (defaults to now)
    
    Returns:
        DocumentMetadata pydantic model with extracted information
    """
    # Determine effective date
    year_from_filename = extract_year_from_filename(filename)
    if year_from_filename:
        effective_date = datetime(year_from_filename, 1, 1)
    elif upload_timestamp:
        effective_date = upload_timestamp
    else:
        effective_date = datetime.now()
    
    # Truncate content if too large for LLM context
    content_for_llm = _truncate_for_classification(content)
    
    # Set up LangChain components
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=DocumentClassification)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a document classifier. Respond only with valid JSON matching the specified format."),
        ("human", """Analyze this document and classify it.

Document filename: {filename}

Document content:
---
{content}
---

Classification rules:
- "policy": HR policies, work rules, company mandates, leave policies, work-from-home rules
- "cafeteria": Food menus, cafeteria schedules, dining information
- "general": Everything else (announcements, newsletters, etc.)

{format_instructions}""")
    ])
    
    chain = prompt | llm | parser
    
    try:
        result: DocumentClassification = chain.invoke({
            "filename": filename,
            "content": content_for_llm,  # Use truncated content for classification
            "format_instructions": parser.get_format_instructions()
        })
        
        return DocumentMetadata(
            doc_type=result.doc_type,
            effective_date=int(effective_date.timestamp()),
            effective_date_str=effective_date.strftime("%Y-%m-%d"),
            filename=filename,
            summary=result.summary,
            tags=result.tags
        )
        
    except Exception as e:
        print(f"Warning: LLM classification failed for {filename}: {e}")
        return _fallback_classification(content, filename, effective_date)


def _fallback_classification(
    content: str,
    filename: str,
    effective_date: datetime
) -> DocumentMetadata:
    """Fallback classification using keyword matching."""
    content_lower = content.lower()
    filename_lower = filename.lower()
    
    if "policy" in filename_lower or "mandate" in content_lower or "policy" in content_lower:
        doc_type = "policy"
    elif "menu" in filename_lower or "cafeteria" in filename_lower or "menu" in content_lower:
        doc_type = "cafeteria"
    else:
        doc_type = "general"
    
    summary = content[:100] + "..." if len(content) > 100 else content
    
    return DocumentMetadata(
        doc_type=doc_type,
        effective_date=int(effective_date.timestamp()),
        effective_date_str=effective_date.strftime("%Y-%m-%d"),
        filename=filename,
        summary=summary,
        tags=[]
    )


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    test_content = """TechCorp Return to Office Mandate (Effective Date: Jan 1, 2024)
    We are excited to welcome everyone back!
    Remote work is now capped at 1 day per week."""
    
    metadata = extract_metadata(test_content, "policy_v2_2024.txt")
    print("Extracted metadata:")
    print(metadata.model_dump_json(indent=2))
