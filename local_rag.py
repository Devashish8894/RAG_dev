import os
import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in your .env file.")

client = genai.Client(api_key=api_key)

documents = [
    "Python is interpreted high-level programming language.",
    "C++ is a compiled language often used for high-performance applications.",
    "RAG stands for Retrieval-Augmented Generation and improves LLM accuracy." 
]

def get_embedding(text):
    # Valid embedding model endpoint
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    assert response.embeddings is not None
    return np.array(response.embeddings[0].values, dtype=np.float32)

# Build FAISS vector store
doc_embedding = [get_embedding(doc) for doc in documents]
doc_matrix = np.array(doc_embedding, dtype=np.float32)

dim = doc_matrix.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(doc_matrix)

# Retrieval
query = "What is RAG in AI?"
query_vec = get_embedding(query).reshape(1, -1)

D, I = index.search(query_vec, k=2)
retrieved_docs = [documents[i] for i in I[0]]

# Format retrieval prompt
context_block = "\n".join(retrieved_docs)
prompt = f"""Use the following context to answer the question.
Context:
{context_block}

Question: {query}
"""

# Generation using Flash-Lite
resp = client.models.generate_content(
    model="gemini-3.5-flash-lite",
    contents=prompt  # Passes the prompt variable, not "prompt" as a literal string
)

print("Answer:\n", resp.text)
print("\nTop matches:")
for rank, doc in enumerate(retrieved_docs, start=1):
    print(f"{rank}. {doc}")