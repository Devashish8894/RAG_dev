"""
Chapter 10: Enterprise Security & Data Privacy — PII Redaction & RBAC Filtering
Masks sensitive PII (emails, SSNs, phone numbers) prior to embedding and enforces
document-level access control (RBAC) via Chroma metadata pre-filtering.
"""

import os
import re
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

PII_PATTERNS = [
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[REDACTED_SSN]"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), "[REDACTED_PHONE]"),
]


def redact_pii(text: str) -> str:
    """Mask sensitive PII before text is embedded or stored."""
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def main() -> None:
    raw_docs = [
        Document(
            page_content="Employee Jane Doe (SSN: 123-45-6789, email: jane@acme.com) is approved for Level 4 access.",
            metadata={"access_role": "hr_admin", "source": "hr_confidential.pdf"},
        ),
        Document(
            page_content="All employees receive 20 days of paid annual leave. Contact hr@acme.com for questions.",
            metadata={"access_role": "employee", "source": "employee_handbook.pdf"},
        ),
    ]

    sanitized_docs = [
        Document(page_content=redact_pii(d.page_content), metadata=d.metadata) for d in raw_docs
    ]

    print("Sanitized Documents (PII Masked):")
    for d in sanitized_docs:
        print(f" - [{d.metadata['access_role']}] {d.page_content}")

    if not os.getenv("OPENAI_API_KEY"):
        print("\nSet OPENAI_API_KEY to also run the live RBAC Chroma vector search.")
        return

    vectorstore = Chroma.from_documents(
        sanitized_docs,
        OpenAIEmbeddings(model="text-embedding-3-small"),
        collection_name="ch10_rbac_demo",
    )

    # Enforce RBAC: an 'employee' user can only retrieve 'employee' documents
    employee_results = vectorstore.similarity_search(
        "What is the leave policy or access level?",
        k=2,
        filter={"access_role": "employee"},
    )
    print("\nRBAC Filtered Results for role='employee':")
    for doc in employee_results:
        print(f" - ({doc.metadata['source']}) {doc.page_content}")


if __name__ == "__main__":
    main()
