"""
Practice (2): Semantic Chunking, Metadata Tagging & Hybrid Search (BM25 + Vector)
Combines SemanticChunker, metadata pre-filtering, and Reciprocal Rank Fusion (RRF).
"""

import os
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("Please set OPENAI_API_KEY before running.")

# 1. Source Text with Mixed Topics & Exact Error Codes
long_technical_text = (
    "Authentication Service Architecture: All external API requests must include a signed JWT bearer token "
    "issued by the OAuth2 gateway. Tokens expire every 60 minutes and require refresh token rotation. "
    "Database Connection Troubleshooting: When PostgreSQL connection pools exceed 200 active sockets, "
    "the gateway throws error code ERR-PG-5042. To resolve ERR-PG-5042, enable PgBouncer transaction pooling "
    "and set max_client_conn to 1000 in pgbouncer.ini. "
    "Billing & Invoice Rules: Enterprise SaaS invoices are generated on the 1st of every month with Net-30 terms."
)

# 2. Semantic Chunking (Splits automatically where topic/meaning shifts)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=85,
)
semantic_chunks = semantic_splitter.create_documents(
    texts=[long_technical_text],
    metadatas=[{"department": "engineering", "doc_version": "v2.4", "source": "runbook.md"}],
)

# Enrich each chunk with deterministic chunk_id metadata
for idx, chunk in enumerate(semantic_chunks):
    chunk.metadata["chunk_id"] = f"eng_runbook_{idx + 1}"

print(f"Created {len(semantic_chunks)} semantically coherent chunks:\n")
for c in semantic_chunks:
    print(f" [{c.metadata['chunk_id']} | dept={c.metadata['department']}] -> {c.page_content[:85]}...")

# 3. Build Dense Vector Retriever (with Metadata Filter) + Sparse BM25 Retriever
vectorstore = Chroma.from_documents(
    documents=semantic_chunks,
    embedding=embeddings,
    collection_name="hybrid_semantic_v1",
)
dense_retriever = vectorstore.as_retriever(
    search_kwargs={"k": 2, "filter": {"department": "engineering"}}
)

bm25_retriever = BM25Retriever.from_documents(semantic_chunks)
bm25_retriever.k = 2


# 4. Hybrid Search via Reciprocal Rank Fusion (RRF)
def hybrid_search_rrf(
    query: str,
    dense_weight: float = 0.5,
    bm25_weight: float = 0.5,
    k_rrf: int = 60,
    top_n: int = 2,
):
    dense_docs = dense_retriever.invoke(query)
    bm25_docs = bm25_retriever.invoke(query)

    scores = {}
    doc_map = {}

    for rank, doc in enumerate(dense_docs):
        cid = doc.metadata["chunk_id"]
        doc_map[cid] = doc
        scores[cid] = scores.get(cid, 0.0) + dense_weight * (1.0 / (k_rrf + rank + 1))

    for rank, doc in enumerate(bm25_docs):
        cid = doc.metadata["chunk_id"]
        doc_map[cid] = doc
        scores[cid] = scores.get(cid, 0.0) + bm25_weight * (1.0 / (k_rrf + rank + 1))

    ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [(doc_map[cid], scores[cid]) for cid in ranked_ids[:top_n]]


# 5. Test Hybrid Search on Exact Error Code + Semantic Concept
test_query = "How do I fix ERR-PG-5042 database pool exhaustion?"
print(f"\n=== HYBRID SEARCH RESULTS FOR: '{test_query}' ===")
for doc, rrf_score in hybrid_search_rrf(test_query):
    print(f"RRF Score: {rrf_score:.4f} | Source: {doc.metadata['source']} ({doc.metadata['chunk_id']})")
    print(f"Content  : {doc.page_content}\n")
