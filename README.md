# **AI Pulse – Backend Pipeline**

### Automated AI News Ingestion, Deduplication, Summarization & Delivery

## **Introduction**

AI Pulse is a fully automated backend pipeline designed to collect, filter, deduplicate, summarize, categorize, and deliver daily AI‑related news to EY employees.  
The platform solves a critical business challenge: **AI news is fast‑moving, scattered, inconsistent in quality, and overwhelming to track manually**.

This backend implements the entire workflow defined in the AI Pulse Business Requirements Document (BRD), including:

*   Daily ingestion of AI‑related articles
*   Credibility & relevance filtering
*   Full‑text content extraction
*   Vector‑based deduplication using a similarity threshold (≥ 0.8)
*   LLM‑driven summarization & categorization
*   Structured JSON output stored in CosmosDB
*   Versioned API delivery to a read‑only, filterable frontend

The system ensures **accuracy, reliability, consistency, and freshness** of AI news summaries while maintaining strict limits on redundancy and factual correctness.

***

## **Getting Started**

This section helps you set up the backend pipeline on your own machine or development environment.

***

### **1. Installation Process**

# AI Pulse — Backend

A concise backend for automated AI news ingestion, deduplication, summarization, and delivery.

This repository contains the server-side pipeline that ingests articles, extracts and cleans text, runs embeddings and similarity checks for deduplication, uses LLMs for summarization/categorization, and persists curated stories.

Repository layout

- app/: main application code
  - routers/: FastAPI routes (health, news, scraper)
  - services/: scrapers, embedding & vector db logic, image generator
  - utils/: helpers including qdrant client and scraper utilities
  - vector_db_pipeline/: scripts for similarity search and insertion
- tests/: pytest test suite (vector DB, pipeline, image generation)
- output/: sample article outputs and raw text used for development
- config.ini, pyproject.toml: configuration and Python packaging

Quick start

1) Install dependencies (Poetry):

```bash
poetry install
```

2) Run the app locally:

```bash
poetry run python app/run.py
```

3) Run tests:

```bash
pytest tests/
```

VS Code tasks

- "Create Poetry Environment": runs `poetry install`
- "Upgrade qdrant-client to latest": runs `poetry add qdrant-client@latest`

Notes

- Configuration: review `config.ini` and `app/config.py` for required environment variables and service endpoints (Qdrant, CosmosDB, LLM provider keys).
- Qdrant (local): this project defaults to a locally hosted Qdrant instance (see `config.ini` — `qdrant.url = http://localhost:6333`). If you prefer Qdrant Cloud, update `qdrant.url` and set `qdrant.api_key` accordingly.
- Image generation: image generation is implemented in `app/services/image_generator.py`. It calls an image API using either `NORA_API_URL`/`NORA_API_KEY` or the Azure OpenAI vars (`AZURE_OPENAI_ENDPOINT`/`AZURE_OPENAI_KEY`). The module exposes a POST `/generate-image` handler and a `process_scraper_results()` helper that:
  - saves generated images to `app/files/images/responses/`,
  - returns both `image_url` (remote) and `local_image_path` (saved copy), and
  - throttles calls in batches (it waits ~1 minute every 3 calls to avoid rate limits).
- Main entry points: `app/run.py` and `app/main.py`.
- Vector DB utilities and pipeline scripts live in `app/vector_db_pipeline/`.
- If you run into dependency issues, activate the Poetry shell first: `poetry shell`.

Contact / Next steps

If you'd like, I can:
- expand this README with example API calls and sample responses,
- add a minimal dev README in `app/` with local debugging tips,
- or run the test suite and report failures.
