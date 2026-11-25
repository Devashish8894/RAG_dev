"""
Chapter 9: Advanced RAG Patterns — Cohere Rerank API
Re-ranks initial candidate documents for maximum Top-N precision.
"""

import os
from cohere import Client
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise EnvironmentError("Please set COHERE_API_KEY in your environment or .env file.")

co = Client(api_key=api_key)

candidate_docs = [
    "Our office cafeteria serves lunch between 12:00 PM and 2:00 PM.",
    "Full-time employees are entitled to 20 days of paid annual leave per calendar year.",
    "Unused paid leaves can be carried forward up to a maximum of 5 days.",
    "All visitors must check in at the front reception desk.",
]

# Re-rank initial candidate documents for maximum precision
results = co.rerank(
    model="rerank-english-v3.0",
    query="How many paid leaves are allowed per year?",
    documents=candidate_docs,
    top_n=3,
)

for r in results.results:
    print(f"Rank Index: {r.index} | Relevance Score: {r.relevance_score:.3f}")
    print(f"Document  : {candidate_docs[r.index]}\n")
