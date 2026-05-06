# AI Resume Job Matcher

Fireworks-first resume-to-job matching web app.

## What it does
- Upload a PDF resume.
- Extract text locally.
- Use Fireworks LLM to infer:
  - target role/category
  - years of experience
  - role family/seniority
  - key skills/signals
- Scrape live jobs from Google Jobs via SerpAPI.
- Rank jobs with semantic similarity (embeddings + cosine).
- Apply Fireworks reranker as second-stage refinement.
- Show ranked jobs with match scores and explanations.

## Active provider stack
- LLM extraction/classification (cost-first routing):
  - Primary: `fireworks/minimax-m2p7`
  - Fallback: `fireworks/deepseek-v3p2` when primary output is weak/invalid
- Embeddings: `fireworks/qwen3-embedding-8b`
- Reranker: `fireworks/qwen3-reranker-8b`

Bedrock files are kept in repo for demo/interview discussion, but are not active by default.

## Tech stack
- Backend: Flask, requests
- Resume parsing: PyMuPDF (+ pdfplumber fallback)
- Retrieval & ranking: Fireworks embeddings/rerank + optional ChromaDB persistence
- Job source: SerpAPI (Google Jobs)
- Frontend: HTML/CSS/Vanilla JS

## Project entrypoint
- Web app: `web/app.py`
- Core pipeline: `src/jobs/enhanced_job_scraper.py`

## Environment variables
Required:
- `FIREWORKS_API_KEY`
- `SERPAPI_KEY`

Recommended defaults:
- `FIREWORKS_PRIMARY_CHAT_MODEL=fireworks/minimax-m2p7`
- `FIREWORKS_FALLBACK_CHAT_MODEL=fireworks/deepseek-v3p2`
- `FIREWORKS_EMBED_MODEL=fireworks/qwen3-embedding-8b`
- `FIREWORKS_RERANK_MODEL=fireworks/qwen3-reranker-8b`
- `USE_FIREWORKS_LLM=1`
- `USE_LLM_QUERY_EXPANSION=1`
- `USE_LLM_RERANK=1`
- `LLM_QUERY_EXPANSIONS=2`
- `LLM_RERANK_TOP_K=8`

Optional:
- `USE_LEGACY_MAIN_PARSER=0` (set `1` only if you want legacy `main.py` enrichment path)
- `MAIL_USERNAME`, `MAIL_PASSWORD` (local SMTP mode only)
- `RENDER_EMAIL_DISABLED=1` (auto-assumed on Render free)

## Local run
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python web/app.py
```
Open `http://localhost:5000`.

## Render deploy (native, non-Docker)
This repo includes `render.yaml`.

1. Push repo to GitHub.
2. In Render, create a new Blueprint/Web Service from this repo.
3. Set env vars in Render dashboard:
   - `FIREWORKS_API_KEY`
   - `SERPAPI_KEY`
4. Deploy.

Health check endpoint: `/healthz`

Notes for Render free:
- SMTP ports are blocked, so email sending is disabled by default in this deployment mode.
- Results, uploads, and caches are ephemeral unless externalized.

## Current pipeline (runtime)
1. Upload PDF
2. Local PDF text extraction
3. Fireworks LLM structured profile extraction
4. Query generation + SerpAPI job fetch
5. Embedding similarity scoring
6. Fireworks rerank blend
7. Return ranked jobs to UI

## Development notes
- Keep API keys in `.env` locally (never commit secrets).
- Fireworks API key should only be read from env (`FIREWORKS_API_KEY`).
- Legacy Bedrock module remains for reference but is inactive by default.

## Tests
Added:
- `tests/test_fireworks_resume_intelligence.py`

Run unit tests (if `pytest` is installed):
```bash
python -m pytest -q
```

## Optional Docker
Existing Docker assets remain in repo, but Render deployment is configured for native Python runtime.
