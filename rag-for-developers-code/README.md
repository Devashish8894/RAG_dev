# RAG for Developers — From Basics to Advanced (Working Code)

Document-wise Python implementations accompanying the handbook **"RAG for Developers — From Basics to Advanced"** by **Devashish Ahire**.

Every script is self-contained, includes sample data fallbacks so it runs out of the box without needing external PDFs, and supports loading API keys from `.env` or environment variables.

## Folder Structure (Document-Wise)

| Folder | Handbook Chapter | Scripts Included |
| :--- | :--- | :--- |
| `01_ch05_basic_rag_pipeline/` | Ch. 5 — Build a Basic RAG Pipeline | `basic_rag_pipeline.py` |
| `02_ch06_llm_prompt_engineering/` | Ch. 6 — RAG with LLM & Prompt Engineering | `grounded_rag_prompt.py` |
| `03_ch07_complete_rag_app/` | Ch. 7 — Complete RAG Application | `complete_rag_app.py` |
| `04_ch08_retrieval_quality/` | Ch. 8 — Improving RAG Quality & Retrieval | `query_transform_and_rerank.py` |
| `05_ch09_advanced_rag_patterns/` | Ch. 9 — Advanced RAG Patterns | `multi_query_compression.py`, `cohere_rerank_example.py`, `agentic_rag.py` |
| `06_ch10_production_workbook/` | Ch. 10 & Hands-On Workbook | `01_baseline_rag_metrics.py`, `02_hybrid_semantic_rag.py`, `06_cached_rag_pipeline.py`, `07_streaming_rag_pipeline.py`, `production_rag_complete.py`, `pii_redaction_and_rbac.py` |

## Quick Setup

### 1. Create and Activate a Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux / macOS / WSL2:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure API Keys

Copy `.env.example` to `.env` and add your `OPENAI_API_KEY` (and `COHERE_API_KEY` if running the Cohere rerank example), or export them in your shell:

**Windows (PowerShell):**
```powershell
$env:OPENAI_API_KEY="sk-your-actual-openai-api-key"
```

**Linux / macOS / WSL2:**
```bash
export OPENAI_API_KEY="sk-your-actual-openai-api-key"
```

### 4. Run Any Script

```bash
python 01_ch05_basic_rag_pipeline/basic_rag_pipeline.py
python 06_ch10_production_workbook/production_rag_complete.py
```
