"""
Chapter 7: Complete RAG Application
End-to-end PDF Question-Answering pipeline using LangChain, OpenAI, and ChromaDB.
"""

import os
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("Please set OPENAI_API_KEY in your environment or .env file.")


def load_pdf_or_fallback(pdf_path: str = "sample.pdf") -> list[Document]:
    """Load PDF pages if available, otherwise return sample documents."""
    if os.path.exists(pdf_path):
        loader = PyPDFLoader(pdf_path)
        return loader.load()

    print(f"'{pdf_path}' not found. Using built-in sample documents.")
    return [
        Document(
            page_content=(
                "According to the company policy, employees are entitled to 20 days "
                "of paid leave per year. Unused leaves can be carried forward up to "
                "a maximum of 5 days."
            ),
            metadata={"source": "sample.pdf", "page": 4},
        ),
        Document(
            page_content=(
                "Parental Leave: Primary caregivers receive 16 weeks of paid parental "
                "leave, and secondary caregivers receive 6 weeks."
            ),
            metadata={"source": "sample.pdf", "page": 5},
        ),
    ]


def main() -> None:
    # Step 1: Load and split the uploaded PDF document
    documents = load_pdf_or_fallback("sample.pdf")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Step 2: Create embeddings and persist in ChromaDB
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./rag_db",
        collection_name="ch07_complete_rag",
    )

    # Step 3: Configure Retriever and RetrievalQA Chain
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    # Step 4: Execute user query
    query = "What is the leave policy?"
    response = qa_chain.invoke({"query": query})
    print("Query:", query)
    print("Answer:", response["result"])
    print("\nSources:")
    for doc in response.get("source_documents", []):
        print(f" - {doc.metadata.get('source', 'unknown')} (Page {doc.metadata.get('page', 'N/A')})")


if __name__ == "__main__":
    main()
