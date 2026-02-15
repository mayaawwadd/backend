# Qdrant Vector DB Pipeline (Cloud)

This module contains scripts for interacting with Qdrant Cloud, focusing on vector ingestion, similarity search, and deduplication. The main logic is implemented in the following scripts:

- **insert_new_vectors.py**: Handles the insertion of new vectors into the Qdrant database.
- **similarity_and_insert.py**: Combines similarity search and vector insertion logic.
- **similarity_search.py**: Performs similarity searches on the Qdrant database.

---

## Setup

1. Ensure you have [Poetry](https://python-poetry.org/) installed.
2. Install dependencies:
   ```
   poetry install
   ```
3. Set up your Qdrant Cloud credentials as environment variables or in a `.env` file (see project root or `app/.env.example` for reference).

---

## How to Run

From the project root, use Poetry to run any of the scripts. For example:

```
poetry run python app/vector_db_pipeline/similarity_search.py
```

---

## About the Scripts

- **insert_new_vectors.py**: Ensures the collection and vector schema exist, and handles the ingestion of new vectors.
- **similarity_and_insert.py**: Combines the logic for similarity search and vector insertion, ensuring deduplication and efficient vector management.
- **similarity_search.py**: Focuses on performing similarity searches, leveraging the Qdrant database for vector comparisons.

Configure your cloud Qdrant endpoint and API key in your environment or config file as needed.
