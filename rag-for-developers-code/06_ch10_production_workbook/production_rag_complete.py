"""
All-in-One: Complete Integrated RAG Pipeline (Hybrid + Cache + Stream + Metrics)
Combines Practices (1), (2), (6), and (7) into one unified runnable app.
"""

import os
import sys
import time
from dotenv import load_dotenv
from langchain_core.globals import set_llm_cache
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.cache import SQLiteCache
from langchain_community.retrievers import BM25Retriever
from langchain_community.storage import LocalFileStore
from langchain_community.vectorstores import Chroma
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from langchain.embeddings import CacheBackedEmbeddings
except ImportError:
    from langchain_classic.embeddings import CacheBackedEmbeddings

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("Set OPENAI_API_KEY environment variable first.")

# --- [PRACTICE 6] Persistent Disk Embedding Cache + SQLite Response Cache ---
os.makedirs("./prod_cache", exist_ok=True)
base_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    base_embeddings,
    LocalFileStore("./prod_cache/embeddings"),
    namespace=base_embeddings.model,
)
set_llm_cache(SQLiteCache(database_path="./prod_cache/responses.db"))

# --- [PRACTICE 2] Semantic Chunking + Metadata + Hybrid Search (Dense + BM25) ---
corpus = (
    "Leave Policy (HR-101): Full-time staff receive 20 paid annual leaves and 10 sick leaves per year. "
    "Incident Runbook (OPS-5042): Error code ERR-5042 indicates Redis cache eviction saturation. "
    "Resolve ERR-5042 by increasing maxmemory to 4GB and setting maxmemory-policy to allkeys-lru. "
    "API Security (SEC-202): All production webhooks must verify HMAC-SHA256 signatures."
)

splitter = SemanticChunker(
    cached_embeddings,
    breakpoint_threshold_type="percentile",
    breakpoint_threshold_amount=80,
)
chunks = splitter.create_documents([corpus], metadatas=[{"org": "ramel_tech", "env": "prod"}])
for i, c in enumerate(chunks):
    c.metadata["chunk_id"] = f"doc_{i + 1}"

vector_db = Chroma.from_documents(chunks, cached_embeddings, collection_name="unified_prod_rag")
dense_ret = vector_db.as_retriever(search_kwargs={"k": 2, "filter": {"env": "prod"}})
bm25_ret = BM25Retriever.from_documents(chunks)
bm25_ret.k = 2


def hybrid_retrieve(query: str) -> str:
    dense_hits = dense_ret.invoke(query)
    bm25_hits = bm25_ret.invoke(query)
    merged, scores = {}, {}
    for r, d in enumerate(dense_hits):
        cid = d.metadata["chunk_id"]
        merged[cid] = d
        scores[cid] = scores.get(cid, 0.0) + 0.5 / (60 + r + 1)
    for r, d in enumerate(bm25_hits):
        cid = d.metadata["chunk_id"]
        merged[cid] = d
        scores[cid] = scores.get(cid, 0.0) + 0.5 / (60 + r + 1)
    top_docs = [merged[cid] for cid in sorted(scores, key=scores.get, reverse=True)[:2]]
    return "\n".join(f"[{d.metadata['chunk_id']}] {d.page_content}" for d in top_docs)


# --- [PRACTICE 7 & 1] Streaming Chain + Latency/TTFT Telemetry ---
prompt = ChatPromptTemplate.from_template(
    "Use ONLY the context below to answer accurately. Cite [chunk_id].\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
chain = (
    {"context": RunnableLambda(hybrid_retrieve), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def ask_streaming(question: str) -> None:
    print(f"\nQ: {question}\nA: ", end="")
    t0 = time.perf_counter()
    ttft = None
    for token in chain.stream(question):
        if ttft is None:
            ttft = (time.perf_counter() - t0) * 1000
        sys.stdout.write(token)
        sys.stdout.flush()
    total_ms = (time.perf_counter() - t0) * 1000
    print(f"\n[Telemetry] TTFT: {ttft:.1f} ms | Total: {total_ms:.1f} ms")


if __name__ == "__main__":
    ask_streaming("How do we resolve ERR-5042 in production?")
    ask_streaming("How do we resolve ERR-5042 in production?")  # Second run hits SQLite response cache!
