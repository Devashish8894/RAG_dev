"""
Practice (1): Start with a Simple Baseline and Iterate with Metrics
Measures Indexing Time, End-to-End Query Latency (ms), and Context Recall@2.
"""

import os
import time
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("Please set OPENAI_API_KEY before running.")

# 1. Sample Knowledge Base (Replace with PyPDFLoader("company_policy.pdf").load())
raw_docs = [
    Document(
        page_content=(
            "Annual Leave Policy: Full-time employees receive 20 days of paid annual leave per year. "
            "Up to 5 unused leave days can be carried forward to the next calendar year."
        ),
        metadata={"source": "hr_policy.pdf", "page": 1},
    ),
    Document(
        page_content=(
            "Remote Work Policy: Engineers may work remotely up to 3 days per week with manager approval. "
            "Core collaboration hours are 10:00 AM to 4:00 PM IST."
        ),
        metadata={"source": "hr_policy.pdf", "page": 2},
    ),
    Document(
        page_content=(
            "Expense Reimbursement: Travel and internet bills up to INR 3,000/month are reimbursed within "
            "7 business days after uploading GST receipts on the HRMS portal."
        ),
        metadata={"source": "finance_guide.pdf", "page": 4},
    ),
]

# 2. Baseline Chunking & Vector Indexing
t0 = time.perf_counter()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=40)
chunks = splitter.split_documents(raw_docs)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="baseline_rag_v1",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
index_time_ms = (time.perf_counter() - t0) * 1000

# 3. Grounded RAG Chain (LCEL)
prompt = ChatPromptTemplate.from_template(
    "Answer ONLY using the provided context. If unsure, say 'I don't know.'\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{d.metadata['source']} p.{d.metadata['page']}] {d.page_content}" for d in docs
    )


rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 4. Iterate with Objective Metrics (Latency + Recall@K)
eval_dataset = [
    {"question": "How many paid leaves do full-time employees get?", "expected_keyword": "20 days"},
    {"question": "What is the monthly internet reimbursement limit?", "expected_keyword": "3,000"},
    {"question": "What are the core remote collaboration hours?", "expected_keyword": "10:00 AM"},
]

print(f"=== BASELINE INDEXED {len(chunks)} CHUNKS IN {index_time_ms:.1f} ms ===\n")
hits = 0
total_latency = 0.0

for item in eval_dataset:
    q = item["question"]
    start = time.perf_counter()
    retrieved = retriever.invoke(q)
    answer = rag_chain.invoke(q)
    latency_ms = (time.perf_counter() - start) * 1000
    total_latency += latency_ms

    # Check if expected ground-truth phrase is present in retrieved chunks (Recall@2)
    is_hit = any(item["expected_keyword"].lower() in d.page_content.lower() for d in retrieved)
    hits += int(is_hit)

    print(f"Q: {q}")
    print(f"A: {answer}")
    print(f"Metrics -> Latency: {latency_ms:.1f} ms | Recall@2 Hit: {is_hit}\n")

print(
    f"SUMMARY -> Recall@2: {(hits / len(eval_dataset)) * 100:.1f}% | "
    f"Avg Latency: {total_latency / len(eval_dataset):.1f} ms"
)
