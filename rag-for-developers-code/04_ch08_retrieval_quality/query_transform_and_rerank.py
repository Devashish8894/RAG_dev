"""
Chapter 8: Improving RAG Quality & Retrieval
Demonstrates Query Transformation, Hybrid Search (Dense Vector + Sparse BM25)
with Reciprocal Rank Fusion (RRF), and LLM-based Cross-Encoder-style Re-Ranking.
"""

import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("Please set OPENAI_API_KEY in your environment or .env file.")


def transform_query(llm: ChatOpenAI, user_query: str) -> list[str]:
    """Expand an ambiguous user query into multiple search-friendly variations."""
    prompt = ChatPromptTemplate.from_template(
        "Rewrite the user query into 3 concise search variations (one per line, no numbering).\n"
        "User Query: {query}"
    )
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"query": user_query})
    variations = [line.strip("- ").strip() for line in raw.splitlines() if line.strip()]
    return [user_query] + variations[:3]


def reciprocal_rank_fusion(
    dense_docs: list[Document],
    bm25_docs: list[Document],
    k_rrf: int = 60,
    top_n: int = 4,
) -> list[tuple[Document, float]]:
    """Merge dense and sparse retrieval results using Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs):
        cid = doc.metadata["chunk_id"]
        doc_map[cid] = doc
        scores[cid] = scores.get(cid, 0.0) + 0.5 / (k_rrf + rank + 1)

    for rank, doc in enumerate(bm25_docs):
        cid = doc.metadata["chunk_id"]
        doc_map[cid] = doc
        scores[cid] = scores.get(cid, 0.0) + 0.5 / (k_rrf + rank + 1)

    ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [(doc_map[cid], scores[cid]) for cid in ranked_ids[:top_n]]


def main() -> None:
    docs = [
        Document(
            page_content="Annual Leave Policy (HR-101): Full-time employees get 20 days of paid annual leave.",
            metadata={"chunk_id": "chunk_1", "department": "HR"},
        ),
        Document(
            page_content="Carry-forward rules: Up to 5 unused paid leave days carry forward to next year.",
            metadata={"chunk_id": "chunk_2", "department": "HR"},
        ),
        Document(
            page_content="Error ERR-PG-5042: Enable PgBouncer transaction pooling with max_client_conn=1000.",
            metadata={"chunk_id": "chunk_3", "department": "Engineering"},
        ),
        Document(
            page_content="General HR Intro: Welcome to the employee handbook.",
            metadata={"chunk_id": "chunk_4", "department": "HR"},
        ),
    ]

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(docs, embeddings, collection_name="ch08_hybrid_quality")
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 3

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    original_query = "vacation days and carryover"
    queries = transform_query(llm, original_query)
    print("Transformed queries:", queries)

    all_dense: list[Document] = []
    all_bm25: list[Document] = []
    for q in queries:
        all_dense.extend(dense_retriever.invoke(q))
        all_bm25.extend(bm25_retriever.invoke(q))

    fused = reciprocal_rank_fusion(all_dense, all_bm25, top_n=3)
    print("\nTop Fused Results:")
    for doc, score in fused:
        print(f" - [{doc.metadata['chunk_id']}] RRF={score:.4f} | {doc.page_content}")


if __name__ == "__main__":
    main()
