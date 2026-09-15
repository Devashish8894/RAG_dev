import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# Load .env variables
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set in your .env file.")

# 1. Corpus
documents = [
    Document(page_content="LangChain is framework for building LLM application."),
    Document(page_content="Chroma is a vector database used for storing embedding."),
    Document(page_content="Gemini is a multimodel large lanaguage model developed by Google."),
    Document(page_content="Retrieval Augmented Generation improve LLM response using external knowledge.")
]

# 2. Text Splitting
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks from {len(documents)} documents.")

# 3. Embeddings & Vector Store
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print("Chroma vector db created and persisted at ./chroma_db.")

retriever = vectordb.as_retriever(search_kwargs={"k": 2})

# 4. LLM & Question-Answering Function
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    temperature=0
)

def rag_answer(query: str) -> str:
    docs = retriever.invoke(query)
    context = "\n".join([d.page_content for d in docs])

    prompt = f"""You are an AI assistant. Use only the following context to answer the question.
If the answer is not in the context, say 'I don't know.'

Context:
{context}

Question: {query}
Answer:"""

    response = llm.invoke(prompt)
    return response.content

# 5. Execution
if __name__ == "__main__":
    query1 = "What is RAG and why it is useful?"
    print("\nQuery 1:", query1)
    print("Gemini Answer:\n", rag_answer(query1))

    query2 = "Who developed gemini?"
    print("\nQuery 2:", query2)
    print("Gemini Answer:\n", rag_answer(query2))