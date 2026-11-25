"""
Chapter 6: RAG with LLM & Prompt Engineering
Demonstrates strict context grounding, fallback instructions, temperature=0.0,
and source citations using the OpenAI Python SDK.
"""

import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from openai import OpenAI

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("Please set OPENAI_API_KEY in your environment or .env file.")

client = OpenAI()


def generate_rag_answer(query: str, retrieved_chunks: list[Document]) -> str:
    """Generate a strictly grounded answer with source citations."""
    context = "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}, Page {doc.metadata.get('page', 'N/A')}] "
        f"{doc.page_content}"
        for doc in retrieved_chunks
    )

    prompt = (
        "You are a helpful enterprise assistant. Use ONLY the context below to answer "
        "the question.\n"
        'If the answer cannot be found in the context, respond with "I don\'t know."\n'
        "Do not use outside knowledge. Cite the source document and page number.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        "Answer:"
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def main() -> None:
    sample_chunks = [
        Document(
            page_content=(
                "Full-time employees are entitled to 20 days of paid annual leave "
                "per calendar year."
            ),
            metadata={"source": "company_policy.pdf", "page": 4},
        ),
        Document(
            page_content="Travel expenses require itemized receipts submitted within 14 days.",
            metadata={"source": "finance_guide.pdf", "page": 9},
        ),
    ]

    in_scope_query = "How many days of paid annual leave do full-time employees get?"
    print(f"Q1: {in_scope_query}")
    print(f"A1: {generate_rag_answer(in_scope_query, sample_chunks)}\n")

    out_of_scope_query = "What is the company stock ticker symbol?"
    print(f"Q2: {out_of_scope_query}")
    print(f"A2: {generate_rag_answer(out_of_scope_query, sample_chunks)}")


if __name__ == "__main__":
    main()
