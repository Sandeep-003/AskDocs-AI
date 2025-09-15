# Theory: Building a Practical RAG Assistant

This document explains the concepts behind the Unified AI Documentation & Knowledge Assistant and why certain design choices were made.

## 1) Retrieval-Augmented Generation (RAG)
RAG augments an LLM with an external knowledge source. Instead of relying solely on the LLM’s parametric memory, we:
- Index your documents as vector embeddings
- Retrieve the most relevant chunks for a query
- Feed those chunks to the LLM with a question-specific prompt (a "stuff" chain)

Benefits:
- Reduces hallucinations by grounding answers in your data
- Keeps model size/cost lower than fine-tuning
- Enables fast updates (re-ingest content without retraining)

## 2) Embeddings and chunking
- Embeddings map text to vectors so that semantically similar content is nearby in vector space.
- Documents are split into small chunks (e.g., 500–1,000 tokens) with overlaps to preserve context.
- Each chunk is embedded and stored in Pinecone with metadata (source URL, path, page number, etc.).

Trade-offs:
- Smaller chunks: more precise matching, but more tokens to read
- Larger chunks: fewer calls, but might dilute relevance
- Overlap: improves continuity at minor cost increase

## 3) Vector search and similarity
- Given a query, we compute its embedding and run a similarity search in Pinecone to get top-k chunks.
- The `k` parameter matters: too low misses context; too high adds noise and token cost.

We use dynamic `k`:
- LLM-based `query_classifier` categorizes the question (definition, how-to, deep dive, broad exploration, etc.)
- Suggests a `k` tuned to the question type
- Optionally rephrases ambiguous questions for better retrieval

## 4) The retrieval + generation chain
- Retrieval: Pinecone retriever with a chosen `k`
- Prompting: LangChain’s `retrieval-qa-chat` prompt for grounded answers
- Generation: Vertex AI (model/config from `config.py`)

For follow-ups, we use a history-aware retriever:
- LangChain’s `create_history_aware_retriever` rewrites the user’s question considering prior turns
- Prevents loss of context across multiple messages

## 5) Conversation memory (long-term)
Short-term history informs rephrasing, but we also maintain a long-term summary:
- `memory_store.py` periodically consolidates user goals, preferences, and recurring themes
- `rag_system.py` prepends a compact “context prefix” (e.g., tech stack, goals) to queries

Why a long-term summary?
- Keeps the assistant aligned to the user’s ongoing focus across sessions and namespaces
- Saves tokens vs. passing the entire history every time

We generate the summary using the LLM, with a heuristic fallback for robustness.

## 6) Namespaces for multi-topic isolation
Pinecone namespaces act like logical databases within an index:
- Each namespace holds embeddings for a particular topic or project
- The app surfaces a namespace selector so you can switch contexts without cross-contamination

This is simpler than maintaining separate indexes and keeps costs manageable.

## 7) Crawling and ingestion pipeline
- `web_scraper.py` crawls within-domain up to N pages (BFS), stores text files
- `document_processor.py` loads `.txt`, `.md`, and `.pdf` via loaders and splits into chunks
- `ingestion.py` embeds and upserts to Pinecone, preserving source metadata and namespace
- The sidebar in `app.py` allows on-demand crawl + ingest; `ingest_from_links.py` supports batch ingestion from a URLs file

## 8) Prompting and tools
- We rely on a proven chat-with-retrieval prompt from LangChain Hub
- `tools.py` includes helpers like `summarize_sources(docs)` to produce a compact meta-summary for the UI
- Slash commands (e.g., `/example`) provide starter prompts to guide users

## 9) Reliability and fallbacks
- LLM-based components (classifier, summary) include heuristic fallbacks to handle outages or quota errors
- Chat history coercion ensures we pass valid `BaseMessage` lists to LangChain chains

## 10) Performance and cost considerations
- Dynamic `k` keeps token usage efficient per query type
- Chunk sizes and overlaps affect ingestion and retrieval latency
- Caching embeddings and avoiding duplicate upserts reduces cost
- Namespaces keep retrieval focused, improving speed and quality

## 11) What’s not included (by design)
- No custom vector DB (we use Pinecone only)
- No continuous scraping (on-demand/batch only)
- No evaluation telemetry dashboards (can be added later)
- No environment mutation scripts (manual env var setup)

## 12) Extending this system
- Add reranking: post-retrieval cross-encoder rerank for better ordering
- Add multi-vector retrieval: summaries + titles + full chunks
- Add structured output: JSON answers with citations for API consumption
- Add guardrails: PII detection and redaction during ingestion and generation
- Add tests: unit tests for ingestion, classifier, and summary functions

---