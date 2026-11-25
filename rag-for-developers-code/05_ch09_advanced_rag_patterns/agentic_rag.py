"""
Chapter 9: Advanced RAG Patterns — Agentic RAG (Tool-Using Autonomous Retrieval)
Demonstrates an agent that plans sub-tasks, selects between a Vector DB search
tool and a calculator tool, evaluates evidence, and synthesizes a final answer.
"""

import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise EnvironmentError("Please set OPENAI_API_KEY in your environment or .env file.")

# Build knowledge base
docs = [
    Document(
        page_content="Standard paid annual leave is 20 days per year. Maximum carryover is 5 days.",
        metadata={"source": "hr_policy.pdf"},
    ),
    Document(
        page_content="Engineering bonus pool is 12% of base salary for exceeding expectations.",
        metadata={"source": "comp_guide.pdf"},
    ),
]
vectorstore = Chroma.from_documents(
    docs,
    OpenAIEmbeddings(model="text-embedding-3-small"),
    collection_name="ch09_agentic_rag",
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


@tool
def search_company_docs(query: str) -> str:
    """Search internal company HR and compensation documents."""
    results = retriever.invoke(query)
    return "\n".join(f"[{d.metadata['source']}] {d.page_content}" for d in results)


@tool
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression (e.g., '20 + 5')."""
    allowed = set("0123456789+-*/(). ")
    if not all(ch in allowed for ch in expression):
        return "Invalid expression."
    return str(eval(expression, {"__builtins__": {}}, {}))


def main() -> None:
    tools = [search_company_docs, calculate]
    tool_map = {t.name: t for t in tools}

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)
    query = (
        "If an employee has the maximum annual leave plus maximum carryover, "
        "how many total leave days do they have?"
    )
    messages = [HumanMessage(content=query)]

    # Iterative Agentic Loop
    for _ in range(5):
        response = llm.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            print("Final Agentic Answer:\n", response.content)
            return

        for call in response.tool_calls:
            result = tool_map[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    print("Reached max reasoning iterations.")


if __name__ == "__main__":
    main()
