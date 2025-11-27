# PiRAG — Cloud-ready Open-Source Repo (no-cost defaults)

This repository is a ready-to-deploy, open-source **PiRAG** demo (Process Intelligence + RAG)
that uses only free/open-source components by default so you can run it in the cloud at zero API cost.

It uses:
- **PM4Py** for Process Intelligence (CSV/XES ingestion & variant discovery)
- **sentence-transformers** for embeddings (all-MiniLM-L6-v2)
- **FAISS (faiss-cpu)** as the vector store
- **transformers** (small T5) for generation (google/flan-t5-small) — optional local LLM fallback
- **Streamlit** for the UI

## Quick start (cloud)

1. Create a GitHub repo and push the files from this bundle.
2. Add your `pirag_docs/` folder (unzip the ZIP you downloaded) to the repo or upload the PDFs to the cloud container on first run.
3. Deploy to Streamlit Community Cloud or Render / Cloud Run using the provided Dockerfile.
4. On first run in the deployed environment run:
   - `python ingest_docs.py --docs-folder=./pirag_docs` (optional: the app can do this step too)
5. Open the Streamlit app URL, upload the CSV event log (or use the provided synthetic CSV), and run Pi discovery and RAG queries.

Important: The app defaults to **local models + FAISS**. No API keys required.

For full file descriptions see the repo files.
