"""Custom lightweight tools used by the assistant (summarize sources, generate code examples)."""
from __future__ import annotations
from typing import List
from langchain_core.documents import Document


def summarize_sources(docs: List[Document], max_chars: int = 800) -> str:
    parts = []
    for d in docs[:5]:
        snippet = d.page_content[:200].replace('\n', ' ')
        parts.append(f"- {snippet}...")
    joined = '\n'.join(parts)
    return f"Key snippets from retrieved context:\n{joined}"[:max_chars]


def generate_code_example(topic: str) -> str:
    # Static templates (could be upgraded with LLM later)
    topic_lower = topic.lower()
    if "pinecone" in topic_lower:
        return (
            "from langchain_pinecone import PineconeVectorStore\n"
            "from langchain_google_vertexai import VertexAIEmbeddings\n\n"
            "emb = VertexAIEmbeddings(model_name='text-embedding-005')\n"
            "store = PineconeVectorStore(index_name='your-index', embedding=emb)\n"
            "retriever = store.as_retriever(search_kwargs={'k':5})\n"
            "query = 'How to upsert vectors?'\n"
            "print(retriever.get_relevant_documents(query))\n"
        )
    if "rag" in topic_lower:
        return (
            "from langchain.chains.retrieval import create_retrieval_chain\n"
            "# Assume retriever + llm already created\n"
            "chain = create_retrieval_chain(retriever, combine_docs_chain)\n"
            "resp = chain.invoke({'input':'Explain RAG in one line'})\n"
            "print(resp['answer'])\n"
        )
    return (
        "# Example not available for this topic yet.\n"
        "# Add more patterns in tools.generate_code_example()\n"
    )
