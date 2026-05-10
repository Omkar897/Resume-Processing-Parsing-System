# AI Resume Intelligence System

An AI-powered resume processing and job matching platform using AMD-backed Fireworks API, vector embeddings, and semantic search for intelligent candidate analysis and job recommendations.

## What it does
- Upload a PDF resume
- Extract text locally using PyMuPDF and pdfplumber
- Use AMD-backed Fireworks LLM to infer:
  - Target role/category
  - Years of experience
  - Role family/seniority
  - Key skills/signals
- Scrape live jobs from Google Jobs via SerpAPI
- Rank jobs with semantic similarity (vector embeddings + cosine similarity)
- Apply Fireworks reranker as second-stage refinement
- Show ranked jobs with match scores and explanations
- ATS scoring for candidate evaluation

## Active provider stack
- **LLM extraction/classification** (AMD-backed Fireworks API):
  - Primary: `fireworks/minimax-m2p7` (AMD GPU-accelerated)
  - Fallback: `fireworks/deepseek-v3p2` (AMD GPU-accelerated)
- **Embeddings**: `fireworks/qwen3-embedding-8b` (AMD GPU-accelerated)
- **Reranker**: `fireworks/qwen3-reranker-8b` (AMD GPU-accelerated)
- **Job Search**: SerpAPI (Google Jobs)
- **Vector Storage**: ChromaDB

## Tech stack
- **Backend**: Flask, Gunicorn (production)
- **Resume parsing**: PyMuPDF, pdfplumber
- **AI/ML**: Fireworks API (AMD-backed), Sentence Transformers, ChromaDB
- **Job source**: SerpAPI (Google Jobs)
- **Frontend**: HTML/CSS/Vanilla JS
- **Deployment**: Railway (cloud platform)

## Project structure
- **Web app**: `web/app.py` - Flask application entry point
- **Core pipeline**: `src/jobs/enhanced_job_scraper.py` - Main job scraping and resume processing logic
- **AI integration**: `LLM/fireworks_resume_intelligence.py` - Fireworks API integration
- **RAG system**: `src/rag/resume_analyzer.py` - Resume analysis and ATS scoring

## Environment variables
### Required:
- `FIREWORKS_API_KEY` - Your Fireworks API key (AMD-backed)
- `SERPAPI_KEY` - Your SerpAPI key for job search

### Recommended defaults:
```bash
FIREWORKS_PRIMARY_CHAT_MODEL=fireworks/minimax-m2p7
FIREWORKS_FALLBACK_CHAT_MODEL=fireworks/deepseek-v3p2
FIREWORKS_EMBED_MODEL=fireworks/qwen3-embedding-8b
FIREWORKS_RERANK_MODEL=fireworks/qwen3-reranker-8b
USE_FIREWORKS_LLM=1
USE_LLM_QUERY_EXPANSION=1
USE_LLM_RERANK=1
LLM_QUERY_EXPANSIONS=2
LLM_RERANK_TOP_K=8
```

### Optional:
- `USE_LEGACY_MAIN_PARSER=0` - Set to `1` only for legacy `main.py` enrichment path
- `MAIL_USERNAME`, `MAIL_PASSWORD` - Email credentials (local SMTP mode only)
- `RAILWAY=1` - Set to `1` for Railway deployment
- `RAILWAY_EMAIL_DISABLED=1` - Email disabled on Railway (SMTP ports blocked)

## Local run
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python web/app.py
```
Open `http://localhost:5000`

## Railway deployment
This project is deployed on Railway using the Railway CLI.

### Deployment setup:
1. **Install Railway CLI**:
   ```bash
   npm install -g @railway/cli
   ```

2. **Login to Railway**:
   ```bash
   railway login
   ```

3. **Create service**:
   - Go to [railway.app](https://railway.app)
   - Create new project "glorious-insight"
   - Create empty service "alluring-wholeness"

4. **Link and deploy**:
   ```bash
   railway link
   railway up
   ```

5. **Set environment variables**:
   - Go to Railway dashboard
   - Add `FIREWORKS_API_KEY`, `SERPAPI_KEY`, and other required variables

### Health check:
- Health check endpoint: `/healthz`
- Railway URL: `https://alluring-wholeness-production-669c.up.railway.app`

### Railway notes:
- **Email functionality disabled** - Railway blocks SMTP ports for security
- **Results are ephemeral** - Use external storage for persistence if needed
- **AMD GPU acceleration** - Fireworks API uses AMD GPU infrastructure

## Current pipeline (runtime)
1. Upload PDF resume
2. Local PDF text extraction (PyMuPDF + pdfplumber fallback)
3. Fireworks LLM structured profile extraction (AMD GPU-accelerated)
4. Query generation + SerpAPI job fetch
5. Vector embedding similarity scoring
6. Fireworks reranker blend (AMD GPU-accelerated)
7. Return ranked jobs with match scores to UI

## Development notes
- Keep API keys in `.env` locally (never commit secrets)
- Fireworks API key should only be read from env (`FIREWORKS_API_KEY`)
- Email functionality is automatically disabled on Railway and Render platforms
- The system uses AMD-backed Fireworks API for cost-efficient AI processing
- Vector embeddings and semantic search provide intelligent job matching

## Tests
Run unit tests (if `pytest` is installed):
```bash
python -m pytest -q
```

## Live demo
- **Railway deployment**: https://alluring-wholeness-production-669c.up.railway.app
- **GitHub repository**: https://github.com/Omkar897/resume-processing-parsing-system

## Features
- ✅ AI-powered resume analysis using AMD-backed LLMs
- ✅ Semantic job matching with vector embeddings
- ✅ Real-time job search and ranking
- ✅ ATS scoring for candidate evaluation
- ✅ Production-ready deployment on Railway
- ✅ Platform-specific email handling
- ✅ Robust error handling and fallback mechanisms
- ✅ Cost-efficient AI processing using AMD GPU infrastructure

## Technologies Used
- **Fireworks API** (AMD-backed LLMs: minimax-m2p7, deepseek-v3p2)
- **ChromaDB** (Vector Database)
- **Flask** (Web Framework)
- **APILayer Resume Parser API** (Resume extraction)
- **SerpAPI** (Job Search)
- **PyMuPDF & pdfplumber** (PDF Processing)
- **Sentence Transformers** (Embeddings)
- **Railway** (Cloud Deployment)
- **Git & GitHub** (Version Control)
- **Gunicorn** (WSGI Server)
