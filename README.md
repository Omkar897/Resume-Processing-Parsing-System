# AI Resume Intelligence System

An AI-powered resume processing and job matching platform. Upload a PDF resume to extract a candidate profile, find current roles through SerpAPI, and rank them with Fireworks embeddings and reranking.

## Features

- PDF resume upload and local text extraction with PyMuPDF and pdfplumber
- Fireworks LLM profile extraction: role, experience, seniority, and skills
- Google Jobs search through SerpAPI
- Semantic matching with Fireworks embeddings and reranking
- Resume insights and ATS scoring
- Vercel deployment with a Flask serverless function

## Tech stack

- Flask and Jinja templates
- Fireworks API for LLM, embeddings, and reranking
- SerpAPI for Google Jobs
- ChromaDB for the optional runtime vector store
- Vercel Functions for deployment

## Project structure

- `api/index.py` - Vercel Function entry point
- `web/app.py` - Flask application and API routes
- `src/jobs/enhanced_job_scraper.py` - Resume-processing and job-matching pipeline
- `src/rag/resume_analyzer.py` - Resume analysis and ATS scoring
- `LLM/fireworks_resume_intelligence.py` - Fireworks integration
- `vercel.json` - Vercel routing, bundled files, and function duration

## Environment variables

Create a local `.env` from [`.env.example`](.env.example). Never commit real credentials.

Required:

- `FIREWORKS_API_KEY` - Fireworks API key
- `SERPAPI_KEY` - SerpAPI key for Google Jobs search

Optional Fireworks settings:

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

Email is disabled by default on Vercel. Set `EMAIL_ENABLED=1` only if you have configured an email provider that is supported from the deployed function. For production email, an HTTP email API is preferable to direct SMTP.

## Run locally

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python web/app.py
```

Open `http://localhost:5000`.

## Deploy on Vercel

The repository is configured for Vercel. The Flask app is served through `api/index.py`; all application routes are rewritten to that function.

1. Import [Omkar897/Resume-Processing-Parsing-System](https://github.com/Omkar897/Resume-Processing-Parsing-System) into Vercel, or run the CLI from the project root:

   ```bash
   npm install -g vercel
   vercel login
   vercel
   ```

2. In **Vercel Project Settings → Environment Variables**, add `FIREWORKS_API_KEY` and `SERPAPI_KEY`. Add any optional variables from `.env.example` that you use locally.

3. Deploy to production:

   ```bash
   vercel --prod
   ```

4. Future pushes to the selected GitHub branch automatically trigger a Vercel deployment.

### Vercel runtime notes

- Resume uploads, the ChromaDB store, and embedding cache use `/tmp` on Vercel. They are temporary and are not shared between function instances.
- The function is configured for a 60-second maximum duration. Very large resumes or slow upstream APIs can still exceed this limit.
- The application uses Fireworks embeddings by default. The local Sentence Transformers fallback is deliberately not packaged for Vercel to keep the Python function within bundle limits.

## Tests

```bash
python -m pytest -q
```

## License

See [LICENSE](LICENSE).
