"""
Chapter 5: Build a Basic RAG Pipeline
Loads a PDF (or falls back to sample HR policy documents), splits into overlapping
chunks, embeds with OpenAI, persists in ChromaDB, and runs similarity search.
"""

import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("Please set OPENAI_API_KEY in your environment or .env file.")


def load_documents(pdf_path: str = "company_policy.pdf") -> list[Document]:
    """Load documents from a PDF if it exists, otherwise return sample policy docs."""
    if os.path.exists(pdf_path):
        print(f"Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        return loader.load()

    print(f"'{pdf_path}' not found. Using built-in sample HR policy documents.")
    return [
        Document(
            page_content=(
                "Annual Leave Policy: Full-time employees are entitled to 20 days of "
                "paid annual leave per calendar year. Unused leaves can be carried "
                "forward up to a maximum of 5 days."
            ),
            metadata={"source": "company_policy.pdf", "page": 4},
        ),
        Document(
            page_content=(
                "Remote Work Policy: Engineers may work remotely up to 3 days per week "
                "with manager approval. Core collaboration hours are 10:00 AM to 4:00 PM."
            ),
            metadata={"source": "company_policy.pdf", "page": 7},
        ),
    ]


def main() -> None:
    # 1. Load document(s)
    documents = load_documents("company_policy.pdf")

    # 2. Split into overlapping chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")

    # 3. Generate embeddings and store in ChromaDB
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./rag_db",
        collection_name="ch05_basic_rag",
    )
    print("Documents successfully indexed in ChromaDB!")

    # 4. Query & Retrieve Relevant Chunks
    query = "What is the leave policy?"
    results = db.similarity_search(query, k=3)

    print(f"\nQuery: {query}")
    for i, doc in enumerate(results):
        print(f"\n--- Chunk {i + 1} (Page {doc.metadata.get('page', 'N/A')}) ---")
        print(doc.page_content)


if __name__ == "__main__":
    main()
