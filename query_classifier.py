"""
Query Classifier Module
Classifies user queries to determine the relevant document category using LangChain + Groq.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

from models import QueryClassification


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


def classify_query(query: str) -> str:
    """
    Classify a user query to determine which document category to search.
    
    Args:
        query: The user's question
    
    Returns:
        One of: "policy", "cafeteria", or "general"
    """
    llm = get_llm()
    parser = PydanticOutputParser(pydantic_object=QueryClassification)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a query classifier. Classify the user's question into the appropriate category."),
        ("human", """Classify this employee question into exactly ONE category:

Categories:
- "policy": Questions about work rules, remote work, office attendance, leave, HR policies, company mandates
- "cafeteria": Questions about food, menus, cafeteria hours, dining options
- "general": Questions that don't fit the above categories

Question: "{query}"

{format_instructions}""")
    ])
    
    chain = prompt | llm | parser
    
    try:
        result: QueryClassification = chain.invoke({
            "query": query,
            "format_instructions": parser.get_format_instructions()
        })
        
        category = result.category.lower()
        if category in ["policy", "cafeteria", "general"]:
            return category
        return "general"
        
    except Exception as e:
        print(f"Warning: Query classification failed: {e}")
        return _fallback_classification(query)


def _fallback_classification(query: str) -> str:
    """
    Fallback classification using keyword matching.
    """
    query_lower = query.lower()
    
    policy_keywords = [
        "remote", "work from home", "wfh", "office", "policy", "leave",
        "vacation", "pto", "mandate", "attendance", "schedule", "hours"
    ]
    
    cafeteria_keywords = [
        "food", "menu", "cafeteria", "lunch", "breakfast", "dinner",
        "eat", "meal", "restaurant", "dining"
    ]
    
    policy_score = sum(1 for kw in policy_keywords if kw in query_lower)
    cafeteria_score = sum(1 for kw in cafeteria_keywords if kw in query_lower)
    
    if policy_score > cafeteria_score:
        return "policy"
    elif cafeteria_score > policy_score:
        return "cafeteria"
    else:
        return "general"


if __name__ == "__main__":
    # Test the classifier
    from dotenv import load_dotenv
    load_dotenv()
    
    test_queries = [
        "Can I work fully remotely this Friday?",
        "What's on the menu today?",
        "When is the company meeting?",
        "What is the remote work policy?",
        "Is the cafeteria open on weekends?"
    ]
    
    print("Query Classification Tests:")
    print("-" * 50)
    for query in test_queries:
        category = classify_query(query)
        print(f"Query: {query}")
        print(f"Category: {category}")
        print("-" * 50)
