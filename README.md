# Arabic News Agentic RAG

A bilingual-friendly Arabic news assistant that combines a LangGraph-based agent with hybrid retrieval over a local Qdrant vector database. The system routes Arabic questions to specialized tools, retrieves relevant news passages, and generates a polished Arabic response.

## Overview

This project implements an agentic retrieval-augmented generation (RAG) workflow for Arabic news:

- A LangGraph state machine decides which tool should handle a user query.
- Retrieval uses hybrid search: dense embeddings plus sparse BM25 retrieval.
- The backend exposes a FastAPI endpoint that runs the agent.
- A Streamlit frontend provides a simple Arabic chat-style interface.

## Main components

- [api/main.py](api/main.py): FastAPI service with the `/query` and `/health` endpoints.
- [frontend/app.py](frontend/app.py): Streamlit web UI for sending Arabic queries to the backend.
- [agent/graph.py](agent/graph.py): LangGraph graph, routing logic, and response generation flow.
- [agent/tools.py](agent/tools.py): Tool implementations for search, summarization, timeline-style retrieval, and direct answering.
- [data/](data/): Notebooks and scripts used for preparing and exploring the Arabic news dataset and vector index.
- [data/qdrant_db/](data/qdrant_db/): Local Qdrant database files used by the retrieval layer.

## How the agent works

The agent supports four tool choices:

- `search_news`: for specific factual questions about a topic or event.
- `summarize_topic`: for broad overview or summary requests.
- `compare_timeline`: for questions about development over time or comparisons.
- `answer_direct`: for general knowledge questions unrelated to news.

The workflow is:

1. The user sends an Arabic query.
2. The router selects the most appropriate tool.
3. The tool performs hybrid retrieval from the Qdrant index.
4. The agent generates a final Arabic response using the retrieved context.

## Requirements

- Python 3.11
- A Groq API key available in the environment as `GROQ_API_KEY`
- Local access to the Qdrant database under [data/qdrant_db](data/qdrant_db)

## Setup

1. Activate the project environment:
   - Windows PowerShell: `.\\rag_env\\Scripts\\Activate.ps1`
   - Linux/macOS: `source rag_env/bin/activate`

2. Install the required Python packages:

   ```bash
   pip install fastapi uvicorn streamlit requests python-dotenv langchain-groq langchain-huggingface sentence-transformers qdrant-client fastembed
   ```

3. Create a `.env` file in the project root with your Groq credentials:

   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

## Run the application

Start the backend first:

```bash
uvicorn api.main:api --reload --host 0.0.0.0 --port 8000
```

Then start the frontend in a second terminal:

```bash
streamlit run frontend/app.py
```

Open the Streamlit app at `http://localhost:8501`.

## API usage

The backend exposes:

- `GET /health`: health check
- `POST /query`: sends a query to the agent and receives the response payload

Example:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"ما آخر التطورات في الأخبار الاقتصادية؟"}'
```

## Notes

- The current implementation uses a local Qdrant store, which is suitable for development and small-to-medium deployments.
- The notebooks under [data/](data/) are useful for experimenting with indexing and retrieval.
- The project expects to run from the repository root so that the relative Qdrant path resolves correctly.

