import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Load local environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# 1. Setup embeddings and load existing Chroma vector DB from disk
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 2. Setup Prompt Template
prompt = ChatPromptTemplate.from_template(
    """Use the following context to answer the question.
If not in context, say 'I don't know'.

Context:
{context}

Question: {question}
Answer:"""
)

# 3. Setup LLM and QA chain
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")

query = "What is RAG and why is it useful?"
retrieved_docs = retriever.invoke(query)
context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])

# Format prompt and invoke
formatted_prompt = prompt.format(context=context_text, question=query)
response = llm.invoke(formatted_prompt)

# Cleanly print the text content
if isinstance(response.content, list):
    answer = "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in response.content])
else:
    answer = str(response.content)

print("Query:", query)
print("\nAnswer:\n", answer)