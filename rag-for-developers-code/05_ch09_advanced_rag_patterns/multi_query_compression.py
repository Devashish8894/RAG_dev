"""
Chapter 9: Advanced RAG Patterns — Multi-Query Retrieval & Context Compression
Generates paraphrased sub-queries to maximize recall, deduplicates retrieved chunks,
and compresses context to strip irrelevant sentences before final generation.
"""

import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("Please set OPENAI_API_KEY in your environment or .env file.")


def generate_sub_queries(llm: ChatOpenAI, question: str) -> list[str]:
    """Use an LLM to generate 3 paraphrased sub-queries."""
    prompt = ChatPromptTemplate.from_template(
        "Generate 3 diverse search queries for the following question (one per line):\n{question}"
    )
    raw = (prompt | llm | StrOutputParser()).invoke({"question": question})
    return [line.strip("- ").strip() for line in raw.splitlines() if line.strip()][:3]


def compress_context(llm: ChatOpenAI, question: str, docs: list[Document]) -> str:
    """Strip irrelevant sentences from retrieved chunks to cut prompt token cost."""
    joined = "\n---\n".join(doc.page_content for doc in docs)
    prompt = ChatPromptTemplate.from_template(
        "Extract ONLY the sentences from the passages below that are directly relevant "
        "to answering the question. Do not paraphrase.\n\n"
        "Question: {question}\n\nPassages:\n{passages}\n\nRelevant Sentences:"
    )
    return (prompt | llm | StrOutputParser()).invoke({"question": question, "passages": joined})


def main() -> None:
    docs = [
        Document(
            page_content=(
                "Company Picnic: Held every July at Riverside Park. "
                "Paid Leave Policy: Full-time employees receive 20 days of paid annual leave. "
                "Office Dress Code: Smart casual Monday through Thursday."
            ),
            metadata={"chunk_id": "d1", "source": "handbook.pdf"},
        ),
        Document(
            page_content=(
                "Cafeteria Menu: Open 8 AM to 3 PM. "
                "Leave Carryover: Up to 5 unused annual leave days may be carried forward. "
                "Parking: Underground badge required."
            ),
            metadata={"chunk_id": "d2", "source": "handbook.pdf"},
        ),
    ]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(docs, embeddings, collection_name="ch09_multi_query")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    question = "How many paid leaves do I get and can I carry them over?"
    sub_queries = [question] + generate_sub_queries(llm, question)

    # Deduplicate retrieved chunks across all sub-queries
    seen: dict[str, Document] = {}
    for q in sub_queries:
        for doc in retriever.invoke(q):
            seen[doc.metadata["chunk_id"]] = doc

    merged_docs = list(seen.values())
    compressed = compress_context(llm, question, merged_docs)

    print("Sub-queries:", sub_queries)
    print("\nCompressed Context:\n", compressed)

    answer_prompt = ChatPromptTemplate.from_template(
        "Answer using ONLY this compressed context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    answer = (answer_prompt | llm | StrOutputParser()).invoke(
        {"context": compressed, "question": question}
    )
    print("\nFinal Answer:\n", answer)


if __name__ == "__main__":
    main()
