"""
Practice (7): Stream Outputs for Sub-Second Perceived Latency
Delivers tokens via LCEL .stream() and measures Time-To-First-Token (TTFT).
"""

import os
import sys
import time
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("Please set OPENAI_API_KEY before running.")

# 1. Setup Vector Store
docs = [
    Document(
        page_content=(
            "Microservice Deployment Guide: Deploy React frontends on Vercel, stateless FastAPI backends "
            "on Docker containers behind Nginx reverse proxy, and PostgreSQL with automated daily snapshots."
        ),
        metadata={"source": "deploy_architecture.pdf", "page": 3},
    ),
    Document(
        page_content=(
            "Security Hardening: Enforce TLS 1.3, rotate JWT secrets every 90 days, enable rate limiting "
            "at 100 requests/minute per IP, and mask PII in application logs."
        ),
        metadata={"source": "security_checklist.pdf", "page": 1},
    ),
]

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="streaming_rag_demo",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 2. Streaming-Enabled LLM & LCEL Chain
prompt = ChatPromptTemplate.from_template(
    "You are a senior solutions architect. Answer the question clearly using ONLY the context below.\n"
    "Include source citations at the end.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)

streaming_llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    streaming=True,
)


def format_docs(retrieved_docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata['source']} - Page {d.metadata['page']}]: {d.page_content}"
        for d in retrieved_docs
    )


streaming_rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | streaming_llm
    | StrOutputParser()
)

# 3. Execute & Measure Time-To-First-Token (TTFT) vs. Total Time
query = "How should we deploy our full-stack app and harden its security?"
print(f"Query: {query}\nStreaming Answer: ", end="")

start_time = time.perf_counter()
first_token_time = None

for chunk in streaming_rag_chain.stream(query):
    if first_token_time is None:
        first_token_time = (time.perf_counter() - start_time) * 1000
    sys.stdout.write(chunk)
    sys.stdout.flush()

total_time_ms = (time.perf_counter() - start_time) * 1000
print("\n\n--- STREAMING TELEMETRY ---")
print(f"Time to First Token (TTFT) : {first_token_time:.1f} ms  <-- Sub-second perceived latency!")
print(f"Total Generation Time      : {total_time_ms:.1f} ms")
