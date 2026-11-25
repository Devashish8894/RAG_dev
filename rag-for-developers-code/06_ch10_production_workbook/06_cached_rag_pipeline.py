"""
Practice (6): Cache Embeddings and Responses to Cut API Spend
Layer 1: Disk-backed Embedding Cache (LocalFileStore + CacheBackedEmbeddings).
Layer 2: Persistent SQLite LLM Response Cache (SQLiteCache).
"""

import os
import time
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.globals import set_llm_cache
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.cache import SQLiteCache
from langchain_community.storage import LocalFileStore
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from langchain.embeddings import CacheBackedEmbeddings
except ImportError:
    from langchain_classic.embeddings import CacheBackedEmbeddings

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("Please set OPENAI_API_KEY before running.")

os.makedirs("./cache", exist_ok=True)

# 1. Layer 1 Cache: Persistent Disk Embedding Cache
underlying_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
fs_store = LocalFileStore("./cache/embeddings_store")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings=underlying_embeddings,
    document_store=fs_store,
    namespace=underlying_embeddings.model,
)

# 2. Layer 2 Cache: Persistent SQLite LLM Response Cache
set_llm_cache(SQLiteCache(database_path="./cache/llm_responses.db"))

docs = [
    Document(
        page_content="Production SLA guarantees 99.95% monthly uptime with 15-minute P1 incident response."
    ),
    Document(
        page_content="Database backups run every 6 hours and are retained in encrypted S3 buckets for 35 days."
    ),
]

# Benchmark Embedding Indexing (Run 1 = API Call, Run 2 = Disk Cache Hit)
t1 = time.perf_counter()
vectorstore = Chroma.from_documents(docs, cached_embedder, collection_name="cached_rag_demo")
print(f"Embedding & Indexing Time : {(time.perf_counter() - t1) * 1000:.1f} ms")

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
prompt = ChatPromptTemplate.from_template(
    "Context: {context}\nQuestion: {question}\nConcise Answer:"
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
chain = prompt | llm | StrOutputParser()

question = "What is our production uptime SLA and backup retention?"
context_text = retriever.invoke(question)[0].page_content

# 3. Benchmark LLM Query (Cold Call vs. Warm Cache Hit)
start_cold = time.perf_counter()
ans_1 = chain.invoke({"context": context_text, "question": question})
cold_ms = (time.perf_counter() - start_cold) * 1000

start_warm = time.perf_counter()
ans_2 = chain.invoke({"context": context_text, "question": question})
warm_ms = (time.perf_counter() - start_warm) * 1000

print(f"\nAnswer         : {ans_1}")
print(f"Cold LLM Call  : {cold_ms:.1f} ms (Tokens billed)")
print(f"Cached LLM Hit : {warm_ms:.1f} ms ($0.00 API cost — {cold_ms / max(warm_ms, 0.1):.1f}x faster!)")
