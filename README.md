# Unified AI Documentation & Knowledge Assistant

An end-to-end Retrieval-Augmented Generation (RAG) assistant for learning from your own PDFs, Markdown, text files, and crawled web pages. It uses LangChain with Google Vertex AI for the LLM, Pinecone as the vector database, and Streamlit for the UI.

Key features:
- Ask questions about ingested docs and websites with citations
- History-aware retrieval for follow-ups
- Dynamic retrieval depth (k) via LLM-based query classification
- Long‑term memory summaries that personalize answers over time
- Namespaces to isolate topics or projects in Pinecone
- On-demand crawl-and-ingest from the sidebar
- One-off batch ingestion from a file of URLs

## Project layout
- UI: `app.py` (Streamlit)
- Core RAG: `rag_system.py` (retrieval chains, history, memory, classification)
- Vector DB: `vector_store.py` (Pinecone with namespace support)
- Ingestion: `ingestion.py`, `document_processor.py` (txt/md/pdf)
- Web crawler: `web_scraper.py` (BFS within domain)
- Memory: `memory_store.py` (LLM-based long-term summary)
- Query classification & rephrasing: `query_classifier.py`
- Utilities: `tools.py` (context summarization, code examples)
- Batch ingestion: `ingest_from_links.py`

Folder: `GenAi/LangChain/Documentation_Assistant/`

## Prerequisites
- Python 3.10+ (3.11 recommended)
- Google Cloud project with Vertex AI enabled
- Service Account with Vertex AI permissions; JSON key file
- Pinecone account and an index

Environment variables (Windows PowerShell examples):
- `GOOGLE_APPLICATION_CREDENTIALS` — path to your SA JSON
- `GOOGLE_CLOUD_PROJECT` — your GCP Project ID
- `PINECONE_API_KEY` — Pinecone API key
- `PINECONE_INDEX` — Pinecone index name (optional if selected in UI)

## Setup
1) Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install dependencies

```powershell
pip install -r requirements.txt
# or install core deps if a requirements file isn’t present
pip install streamlit langchain langchain-google-vertexai google-cloud-aiplatform pinecone-client beautifulsoup4 requests tiktoken pypdf langchain-community
```

3) Set environment variables (examples)

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS = "C:\\path\\to\\service-account.json"
$env:GOOGLE_CLOUD_PROJECT = "your-gcp-project-id"
$env:PINECONE_API_KEY = "your-pinecone-api-key"
$env:PINECONE_INDEX = "your-index-name" 
```

## Run the app
From the `Documentation_Assistant` folder:

```powershell
streamlit run app.py
```

You’ll see:
- Chat with answers and sources
- Meta panel showing query type, chosen k, and a short context summary
- Sidebar for namespace selection and crawl + ingest

## Using the app
- Choose a namespace to isolate topics.
- Optionally paste a URL and max pages to crawl, then ingest.
- Ask your question. The system may rephrase it, retrieve context, and answer with citations.
- Tip: Type `/example` for suggested prompts.

## Troubleshooting
- Vertex AI auth: Check `GOOGLE_APPLICATION_CREDENTIALS` and `GOOGLE_CLOUD_PROJECT`.
- Pinecone: Verify index exists and API key/region are correct.
- Empty answers: Ingest more content, check namespace, or ask more concretely.
- Costs: Retrieval depth (k) and chunk size affect cost/latency.

## License
This is a learning/reference project. Apply your preferred license for derivative work.

***

For a production README (badges, CI, Docker), we can extend this upon request.