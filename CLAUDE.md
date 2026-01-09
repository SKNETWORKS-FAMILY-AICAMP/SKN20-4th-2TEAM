# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HuggingFace WeeklyPapers RAG Chatbot: An AI-powered system that crawls AI/ML/DL/LLM papers from HuggingFace WeeklyPapers, builds a RAG system using LangGraph, and provides a web interface for users to query papers through Django (frontend) and FastAPI (RAG API backend).

## Architecture

### Two-Server Architecture
1. **Django Web Server** (Port 8000): User authentication, chat history management, web UI
2. **FastAPI RAG Server** (Port 8001): RAG system with LangGraph pipeline, vector search, paper retrieval

Django communicates with FastAPI via HTTP requests to `http://localhost:8001` (or `http://rag_api:8001` in Docker).

### Key Components

**Backend (Django)**
- `backend/hugging_project/`: Django project configuration
  - `settings.py`: Korean locale (ko-kr, Asia/Seoul), SQLite database, static files config
  - `urls.py`: Routes to admin and chatbot app
- `backend/chatbot/`: Main Django app
  - `models.py`: `User`, `ChatProject` (folders for organizing chats), `ChatHistory` (Q&A records with project_id)
  - `views.py`: Handles user auth, chat interface, proxies requests to FastAPI, manages chat history and projects
  - `templates/`: HTML templates for UI

**RAG System (FastAPI)**
- `src/rag/`:
  - `api.py`: FastAPI endpoints (`/api/chat`, `/api/stats`, `/api/trending-keywords`, `/api/health`)
  - `rag_system.py`: LangGraph-based RAG pipeline with 8 nodes (translation, classification, hybrid search, reranking, answer generation)
  - `prompts.py`: Prompt templates for LLM interactions
- `src/utils/`:
  - `crawling.py`: Scrapes HuggingFace WeeklyPapers (2025-W41 ~ 2026-W01)
  - `vectordb.py`: ChromaDB vector database management
  - `data_init.py`: Integrated script for crawling and vector DB initialization

**Data Storage**
- `data/documents/`: Crawled paper JSON files
- `data/vector_db/`: ChromaDB vector embeddings
- `backend/db.sqlite3`: User accounts, chat history, project folders

### RAG Pipeline (LangGraph)

8-node workflow in `src/rag/rag_system.py`:
1. **Translation**: Detect Korean and translate to English if needed
2. **Classification**: Determine if query is AI/ML related
3. **Query Expansion**: Expand query with synonyms/related terms for better retrieval
4. **Hybrid Retrieval**: Vector search (embedding similarity) + BM25 (keyword matching)
5. **Reranking**: Cross-encoder or LLM-based reranking to improve relevance
6. **Metadata Boosting**: Boost papers with high upvotes or recency
7. **Answer Generation**: GPT-4o-mini generates answer from retrieved papers
8. **Web Fallback**: If no relevant papers found, uses Tavily for web search

### Database Models

**ChatProject**: Folders for organizing chats
- `user` (ForeignKey to User)
- `folder_name` (unique per user)
- `created_at`, `updated_at`

**ChatHistory**: Q&A records
- `user` (ForeignKey to User)
- `project_id` (0 = no project, >0 = belongs to ChatProject.id)
- `question`, `answer`, `sources` (JSON), `search_type`
- `created_at`

## Development Commands

### Initial Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
# Copy .env.example to .env and add OPENAI_API_KEY and TAVILY_API_KEY

# Initialize data (crawl papers and build vector DB)
python -m src.utils.data_init

# Run Django migrations
cd backend
python manage.py makemigrations
python manage.py migrate

# Create Django superuser
python manage.py createsuperuser
```

### Running Servers

**Development (Local)**
```bash
# Terminal 1: FastAPI RAG Server
python -m src.rag.api
# or
uvicorn src.rag.api:app --host 0.0.0.0 --port 8001 --reload

# Terminal 2: Django Web Server
cd backend
python manage.py runserver
```

**Production (Docker)**
```bash
# Build and start both servers
docker-compose up --build

# Run in background
docker-compose up -d

# Stop servers
docker-compose down

# View logs
docker-compose logs -f
```

### Django Management

```bash
cd backend

# Database migrations
python manage.py makemigrations
python manage.py migrate

# Create superuser
python manage.py createsuperuser

# Collect static files
python manage.py collectstatic

# Django shell (for debugging)
python manage.py shell

# Access Django admin at http://localhost:8000/admin
```

### Data Management

```bash
# Re-crawl papers and rebuild vector DB
python -m src.utils.data_init

# Test crawling only
python -m src.utils.crawling

# Test vector DB operations
python -m src.utils.vectordb
```

## Environment Variables

Required in `.env` file at project root:
```
OPENAI_API_KEY=sk-proj-...
TAVILY_API_KEY=tvly-dev-...
MODEL_NAME=text-embedding-3-small
```

## API Endpoints

### FastAPI (Port 8001)
- `GET /`: Server status
- `POST /api/chat`: Send question, get RAG answer
- `GET /api/stats`: Paper count and system status
- `GET /api/trending-keywords`: Get trending AI/ML keywords
- `GET /api/health`: Health check for Docker

### Django (Port 8000)
- `GET /`: Home page
- `GET /chatbot/`: Chat interface (requires login)
- `GET /projects/`: View all chats and projects (requires login)
- `POST /api/send/`: Proxy to FastAPI /api/chat
- `GET /api/history/`: Get user's chat history
- `POST /api/chat/<id>/delete/`: Delete a chat record
- `POST /projects/create/`: Create new project folder
- `DELETE /projects/<id>/delete/`: Delete project folder

## Key Files to Modify

**Add new RAG node**: Edit `src/rag/rag_system.py` (GraphState, node functions, graph construction)

**Change LLM prompts**: Edit `src/rag/prompts.py`

**Modify Django models**: Edit `backend/chatbot/models.py`, then run `makemigrations` and `migrate`

**Add Django views**: Edit `backend/chatbot/views.py` and `backend/chatbot/urls.py`

**Change UI**: Edit `backend/chatbot/templates/chatbot/*.html`

**Adjust crawling logic**: Edit `src/utils/crawling.py`

**Modify vector DB behavior**: Edit `src/utils/vectordb.py`

## Important Notes

- Django and FastAPI run as separate processes/containers
- Django proxies RAG requests to FastAPI (see `views.py:send_message()`)
- Vector DB must be initialized before starting FastAPI (run `data_init.py` first)
- ChromaDB data persists in `data/vector_db/` directory
- SQLite DB is at `backend/db.sqlite3`
- Korean text is auto-detected and translated to English for RAG processing
- FastAPI health check has 120s startup period for Docker (allows time for vector DB loading)
- Project folders (`ChatProject`) organize chats but don't affect RAG behavior
- The `project_id` field in `ChatHistory` links chats to projects (0 = uncategorized)
