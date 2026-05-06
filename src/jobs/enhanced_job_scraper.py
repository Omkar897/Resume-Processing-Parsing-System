"""
Enhanced Job Scraper with RAG Integration
Adds semantic job matching and resume analysis.

FLOW (when you upload a resume on the web app):
1. Get category + experience: predict category with Fireworks LLM first, then enrich with main.py/APILayer experience data; legacy classifier is fallback only when LLM is unavailable.
2. Build base search query from category + years, then optionally expand queries with Fireworks.
3. Fetch jobs from SerpAPI (Google Jobs) across expanded queries and de-duplicate.
4. Rank with semantic embeddings; optionally apply Fireworks reranking on top candidates.
5. Save JSON; web app shows jobs with match scores, explanations, and resume insights.
"""

import subprocess
import sys
import os
import re
import requests
import json
import math
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import RAG components
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.rag.rag_engine import RAGEngine
from src.rag.resume_analyzer import ResumeAnalyzer

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"


class EnhancedJobScraper:
    """Job scraper with RAG-powered semantic matching"""

    def __init__(self, headless=False):
        self.serpapi_key = os.getenv("SERPAPI_KEY")
        if not self.serpapi_key:
            raise Exception("SERPAPI_KEY not found in .env file")

        self.headless = headless

        # Project root: prefer env (set by Flask) so paths match when run as subprocess
        self.project_root = os.path.abspath(
            os.environ.get("RESUME_PROJECT_ROOT")
            or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
        )

        self.rag_engine = None
        self.resume_analyzer = None
        self.llm_intelligence = None
        self.last_llm_prediction = None
        self.last_category_failure_reason = ""
        self.last_search_queries = []
        self.last_years_source = "unknown"
        self.llm_provider = "none"

        # RAG engine (local embeddings + ChromaDB)
        try:
            persist_dir = os.path.abspath(
                os.path.join(self.project_root, "data", "chromadb")
            )
            os.makedirs(persist_dir, exist_ok=True)
            # Keep HuggingFace/transformers cache inside project so subprocess has a valid path
            cache_dir = os.path.abspath(
                os.path.join(self.project_root, "data", ".embedding_cache")
            )
            os.makedirs(cache_dir, exist_ok=True)
            os.environ["HF_HOME"] = cache_dir
            os.environ["HF_HUB_CACHE"] = os.path.join(cache_dir, "hub")
            os.environ.pop("TRANSFORMERS_CACHE", None)
            if not self.headless:
                print("ðŸ§  Initializing RAG engine...")
            self.rag_engine = RAGEngine(persist_directory=persist_dir)
            if not self.headless:
                print("âœ… RAG engine ready")
        except Exception as e:
            # Always log a single line so web runs aren't silent
            print(f"âš ï¸ RAG engine disabled: {e}")
            self.rag_engine = None

        # Claude resume analyzer (optional)
        try:
            self.resume_analyzer = ResumeAnalyzer()
        except Exception as e:
            print(f"âš ï¸ Claude analyzer disabled: {e}")
            self.resume_analyzer = None

        # Fireworks category intelligence (optional, enabled by default when key is present)
        use_llm_category = self._should_use_llm_category()
        if use_llm_category:
            try:
                from LLM.fireworks_resume_intelligence import (
                    FireworksResumeIntelligence,
                )

                self.llm_intelligence = FireworksResumeIntelligence()
                self.llm_provider = "fireworks"
                if not self.headless:
                    print("✅ Fireworks resume intelligence ready")
            except Exception as e:
                print(f"⚠️ Fireworks resume intelligence disabled: {e}")
                self.llm_intelligence = None
                self.llm_provider = "none"

        # Optional LLM enhancements around retrieval/ranking.
        self.max_expansion_queries = self._safe_env_int(
            "LLM_QUERY_EXPANSIONS",
            default=self._safe_env_int(
                "BEDROCK_QUERY_EXPANSIONS", default=2, min_value=1, max_value=5
            ),
            min_value=1,
            max_value=5,
        )
        self.llm_rerank_top_k = self._safe_env_int(
            "LLM_RERANK_TOP_K",
            default=self._safe_env_int(
                "BEDROCK_RERANK_TOP_K", default=8, min_value=3, max_value=12
            ),
            min_value=3,
            max_value=12,
        )
        self.use_llm_query_expansion = bool(self.llm_intelligence) and (
            (
                os.getenv("USE_LLM_QUERY_EXPANSION")
                or os.getenv("USE_BEDROCK_QUERY_EXPANSION")  # legacy alias
                or "1"
            )
            .strip()
            .lower()
            in {"1", "true", "yes"}
        )
        self.use_llm_rerank = bool(self.llm_intelligence) and (
            (
                os.getenv("USE_LLM_RERANK") or os.getenv("USE_BEDROCK_RERANK") or "1"
            )  # legacy alias
            .strip()
            .lower()
            in {"1", "true", "yes"}
        )
        self.match_score_boost = self._safe_env_int(
            "MATCH_SCORE_BOOST", default=8, min_value=0, max_value=15
        )

    def _should_use_llm_category(self):
        """Enable Fireworks LLM path explicitly or when FIREWORKS_API_KEY is present."""
        explicit = (os.getenv("USE_FIREWORKS_LLM") or "").strip().lower()
        if explicit in {"1", "true", "yes"}:
            return True
        if explicit in {"0", "false", "no"}:
            return False

        return bool(os.getenv("FIREWORKS_API_KEY"))

    def _safe_env_int(self, key, default=0, min_value=0, max_value=100):
        """Parse bounded int from env with default fallback."""
        raw = os.getenv(key, str(default)).strip()
        try:
            value = int(raw)
        except Exception:
            return default
        return max(min_value, min(max_value, value))

    def parse_duration_to_months(self, duration_str):
        """Parse duration strings into total months."""
        try:
            if not duration_str:
                return 0
            month_map = {
                "jan": 1,
                "feb": 2,
                "mar": 3,
                "apr": 4,
                "may": 5,
                "jun": 6,
                "jul": 7,
                "aug": 8,
                "sep": 9,
                "oct": 10,
                "nov": 11,
                "dec": 12,
                "january": 1,
                "february": 2,
                "march": 3,
                "april": 4,
                "june": 6,
                "july": 7,
                "august": 8,
                "september": 9,
                "october": 10,
                "november": 11,
                "december": 12,
            }

            duration_lower = str(duration_str).lower().strip()
            duration_lower = duration_lower.replace("–", "-").replace("—", "-")

            # Pattern: "1 year 6 months", "2 yrs", "8 months"
            years_match = re.search(r"(\d+)\s*(year|years|yr|yrs)", duration_lower)
            months_match = re.search(r"(\d+)\s*(month|months|mo|mos)", duration_lower)
            if years_match or months_match:
                years = int(years_match.group(1)) if years_match else 0
                months = int(months_match.group(1)) if months_match else 0
                total = years * 12 + months
                return max(total, 1)

            # Pattern: "Jun 2023 - Aug 2024", "Jan 2024 - Present"
            if "-" in duration_lower:
                parts = [p.strip() for p in duration_lower.split("-", 1)]
                start_match = re.search(r"([a-zA-Z]+)\s*(\d{4})", parts[0])
                end_match = re.search(
                    r"([a-zA-Z]+)\s*(\d{4})|present|current|now", parts[1]
                )
                if start_match and end_match:
                    start_month_str = start_match.group(1).lower()[:3]
                    start_month = month_map.get(start_month_str, 1)
                    start_year = int(start_match.group(2))

                    if end_match.group(1) and end_match.group(2):
                        end_month_str = end_match.group(1).lower()[:3]
                        end_month = month_map.get(end_month_str, 12)
                        end_year = int(end_match.group(2))
                    else:
                        now = datetime.now()
                        end_month = now.month
                        end_year = now.year

                    months = (end_year - start_year) * 12 + (end_month - start_month) + 1
                    return max(months, 1)

            return 0
        except:
            return 0

    def _format_experience_label(self, years_experience):
        """Format float years as X years Y months."""
        try:
            total_months = max(0, int(round(float(years_experience or 0.0) * 12)))
        except Exception:
            total_months = 0
        years = total_months // 12
        months = total_months % 12

        if years == 0 and months == 0:
            return "0 months"
        if years == 0:
            return f"{months} month" if months == 1 else f"{months} months"
        if months == 0:
            return f"{years} year" if years == 1 else f"{years} years"
        year_label = "year" if years == 1 else "years"
        month_label = "month" if months == 1 else "months"
        return f"{years} {year_label} {months} {month_label}"

    def get_resume_data(self, resume_path):
        """Extract category, total experience, and structured extracted_data from resume.
        Default path: Fireworks LLM extraction/classification.
        Optional enrichment path: main.py --json / legacy parsing when enabled.
        Final fallback: local processor + classifier.
        Returns (category, total_years, extracted_data). extracted_data may be None.
        """
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        main_script_path = os.path.join(project_root, "main.py")
        llm_prediction = self._predict_category_with_llm(resume_path)

        llm_category = (
            llm_prediction.get("category") if isinstance(llm_prediction, dict) else None
        )

        # ---- 1) Default: LLM-only structured extraction ----
        if llm_category is not None:
            llm_years = 0.0
            try:
                llm_years = float(llm_prediction.get("years_experience") or 0.0)
            except Exception:
                llm_years = 0.0
            extracted_data = self._build_minimal_extracted_data(llm_category, llm_years)
            self._attach_llm_signals(extracted_data, llm_prediction)
            if not self.headless:
                conf = llm_prediction.get("confidence", 0.0)
                print(
                    f"✓ Found category (LLM): {llm_category} (confidence: {conf:.2f})"
                )
            return llm_category, llm_years, extracted_data

        strict_headless_llm = self.headless and (
            os.getenv("STRICT_LLM_CATEGORY_ONLY", "1").strip().lower()
            in {"1", "true", "yes"}
        )
        if strict_headless_llm:
            reason = self._format_llm_failure_reason(llm_prediction)
            self.last_category_failure_reason = (
                reason
                or "Automatic extraction/categorization failed in strict LLM mode."
            )
            print(
                f"⚠️ Category extraction failed in strict LLM mode: {self.last_category_failure_reason}"
            )
            return None, 0, None

        # ---- 2) Legacy/main.py enrichment path (opt-in) ----
        # Disabled by default for web/headless strict LLM behavior.
        use_legacy_main = os.getenv("USE_LEGACY_MAIN_PARSER", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if use_legacy_main:
            category, total_years, extracted_data = self._get_resume_data_via_main_json(
                main_script_path, resume_path, project_root
            )
            if category is not None:
                if extracted_data is None:
                    extracted_data = self._build_minimal_extracted_data(
                        category, total_years
                    )
                return category, total_years, extracted_data

            if not self.headless:
                print("⚠️ JSON path failed, trying legacy stdout parsing...")
            category, total_years = self._get_resume_data_via_main_stdout(
                main_script_path, resume_path, project_root
            )
            if category is not None:
                fallback_data = self._build_minimal_extracted_data(
                    category, total_years
                )
                return category, total_years, fallback_data

        # ---- 3) Final fallback: local PDF extraction + embedding classifier ----
        if not self.headless:
            print("⚠️ LLM path unavailable, trying local resume processor...")
        category, total_years, extracted_data = (
            self._get_resume_data_via_local_processor(resume_path, project_root)
        )
        if category is not None:
            return category, total_years, extracted_data

        if not self.headless:
            print("❌ All resume extraction paths failed")
        self.last_category_failure_reason = (
            self.last_category_failure_reason
            or self._format_llm_failure_reason(llm_prediction)
            or "All extraction/categorization paths failed."
        )
        return None, 0, None

    def _predict_category_with_llm(self, resume_path):
        """Run Fireworks classification once and cache the latest prediction."""
        self.last_llm_prediction = None
        if not self.llm_intelligence:
            self.last_category_failure_reason = "Fireworks resume intelligence is disabled."
            return {"success": False, "error": self.last_category_failure_reason}

        try:
            result = self.llm_intelligence.classify_resume_pdf(resume_path)
            if result.get("success") and result.get("category"):
                self.last_llm_prediction = result
                self.last_category_failure_reason = ""
                if not self.headless:
                    confidence = result.get("confidence", 0.0)
                    print(
                        f"✓ LLM category: {result['category']} (confidence: {confidence:.2f})"
                    )
                return result

            failure_reason = self._format_llm_failure_reason(result)
            self.last_category_failure_reason = failure_reason
            print(f"⚠️ LLM category failed: {failure_reason}")
            return result if isinstance(result, dict) else {"success": False, "error": failure_reason}
        except Exception as e:
            self.last_category_failure_reason = f"LLM category exception: {e}"
            print(f"⚠️ LLM category error: {e}")
            return {"success": False, "error": str(e)}

    def _format_llm_failure_reason(self, result):
        """Build a concise, actionable failure reason for logs and API errors."""
        if not isinstance(result, dict):
            return "Fireworks categorization returned no structured response."
        pieces = []
        if result.get("error"):
            pieces.append(str(result.get("error")).strip())
        if result.get("primary_error"):
            pieces.append(f"primary={str(result.get('primary_error')).strip()}")
        if result.get("fallback_error"):
            pieces.append(f"fallback={str(result.get('fallback_error')).strip()}")
        method = result.get("extraction_method")
        text_len = result.get("extracted_text_length")
        if method:
            pieces.append(f"pdf_extraction={method}")
        if text_len is not None:
            pieces.append(f"text_length={text_len}")
        if not pieces:
            return "Fireworks categorization returned an unusable result."
        return " | ".join(pieces)

    def _attach_llm_signals(self, extracted_data, prediction):
        """Attach LLM metadata for traceability in downstream flow."""
        if not isinstance(extracted_data, dict):
            return
        if not isinstance(prediction, dict):
            return
        if not prediction.get("success"):
            return

        extracted_data["llm_signals"] = prediction.get("key_signals", [])
        extracted_data["llm_category_reasoning"] = prediction.get("reasoning", "")
        extracted_data["llm_category_confidence"] = prediction.get("confidence", 0.0)
        extracted_data["llm_role_family"] = prediction.get("role_family", "General")
        extracted_data["llm_seniority"] = prediction.get("seniority", "Entry")
        extracted_data["llm_years_experience"] = prediction.get("years_experience", 0.0)
        extracted_data["llm_provider"] = self.llm_provider

        # Merge inferred LLM skills into extracted skills when available.
        llm_skills = prediction.get("skills", [])
        if isinstance(llm_skills, list) and llm_skills:
            existing = extracted_data.get("skills", [])
            if not isinstance(existing, list):
                existing = []
            seen = {str(s).strip().lower() for s in existing if str(s).strip()}
            for skill in llm_skills:
                skill_text = str(skill).strip()
                if not skill_text:
                    continue
                key = skill_text.lower()
                if key in seen:
                    continue
                seen.add(key)
                existing.append(skill_text)
            extracted_data["skills"] = existing

    def _extract_last_json_object(self, output_text):
        """Extract the last valid JSON object from noisy stdout/stderr."""
        if not output_text:
            return None

        output_text = output_text.strip()
        if not output_text:
            return None

        # Fast path: output is a clean JSON payload.
        try:
            return json.loads(output_text)
        except Exception:
            pass

        # Common path: JSON on the final line after logs.
        for line in reversed(output_text.splitlines()):
            line = line.strip()
            if not (line.startswith("{") and line.endswith("}")):
                continue
            try:
                return json.loads(line)
            except Exception:
                continue

        # Last resort: walk backwards to find a valid JSON object start.
        start = output_text.rfind("{")
        while start != -1:
            candidate = output_text[start:].strip()
            try:
                return json.loads(candidate)
            except Exception:
                start = output_text.rfind("{", 0, start)

        return None

    def _build_minimal_extracted_data(self, category, total_years):
        """Build a minimal extracted_data payload when only stdout parsing succeeds."""
        if total_years and total_years > 0:
            duration = f"Estimated {total_years:.1f} years"
        else:
            duration = "Duration not specified"

        return {
            "personal_info": {},
            "skills": [category] if category else [],
            "experience": (
                [
                    {
                        "role": category or "Not specified",
                        "company": "Not specified",
                        "duration": duration,
                        "description": "Generated from fallback category parsing.",
                    }
                ]
                if category
                else []
            ),
            "education": [],
            "projects": [],
        }

    def _get_resume_data_via_main_json(
        self, main_script_path, resume_path, project_root
    ):
        """Run main.py --json; return (category, total_years, extracted_data) or (None, 0, None)."""
        try:
            cmd = [sys.executable, main_script_path, "--file", resume_path, "--json"]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=env,
                errors="ignore",
                cwd=project_root,
            )
            output = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
            if not output:
                return None, 0, None
            data = self._extract_last_json_object(output)
            if not isinstance(data, dict):
                return None, 0, None
            if data.get("error"):
                return None, 0, None
            category = data.get("category")
            extracted_data = data.get("extracted_data")
            if not category:
                return None, 0, extracted_data
            total_months = 0
            if extracted_data and isinstance(extracted_data.get("experience"), list):
                for exp in extracted_data["experience"]:
                    duration_str = exp.get("duration") or ""
                    if duration_str and duration_str != "Duration not specified":
                        total_months += self.parse_duration_to_months(duration_str)
            total_years = total_months / 12.0
            if not self.headless:
                print(
                    f"âœ“ Found category: {category}"
                    + (" (full data for RAG)" if extracted_data else "")
                )
            return category, total_years, extracted_data
        except (json.JSONDecodeError, KeyError, TypeError):
            return None, 0, None
        except Exception:
            return None, 0, None

    def _get_resume_data_via_main_stdout(
        self, main_script_path, resume_path, project_root
    ):
        """Run main.py without --json; parse stdout for Predicted Category and Duration. Return (category, total_years) or (None, 0)."""
        try:
            cmd = [sys.executable, main_script_path, "--file", resume_path]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=env,
                errors="ignore",
                cwd=project_root,
            )
            output = result.stdout + result.stderr
            lines = output.split("\n")
            category = None
            total_months = 0
            in_experience_section = False
            for line in lines:
                if "Predicted Category:" in line:
                    m = re.search(r"Predicted Category:\s*(.+)", line)
                    if m:
                        category = m.group(1).strip()
                if "predicted category" in line.lower() and not category:
                    m = re.search(r"category[:\s]+([A-Za-z\s]+)", line, re.IGNORECASE)
                    if m:
                        category = m.group(1).strip()
                if "WORK EXPERIENCE" in line:
                    in_experience_section = True
                    continue
                if in_experience_section and (
                    "PROJECTS" in line or "EDUCATION" in line or "SKILLS" in line
                ):
                    in_experience_section = False
                if in_experience_section and "Duration:" in line:
                    m = re.search(r"Duration:\s*(.+)", line)
                    if m:
                        total_months += self.parse_duration_to_months(
                            m.group(1).strip()
                        )
            total_years = total_months / 12.0
            if category and not self.headless:
                print(f"âœ“ Found category (legacy): {category}")
            return category, total_years
        except Exception:
            return None, 0

    def _get_resume_data_via_local_processor(self, resume_path, project_root):
        """Use src/resume processor + classifier (local PDF + embedding model). No APILayer. Return (category, total_years, extracted_data) or (None, 0, None)."""
        try:
            old_cwd = os.getcwd()
            old_path = list(sys.path)
            sys.path.insert(0, project_root)
            os.chdir(project_root)
            try:
                from src.resume.processor import WorkingResumeProcessor

                processor = WorkingResumeProcessor()
                result = processor.process_resume(resume_path)
            finally:
                os.chdir(old_cwd)
                sys.path[:] = old_path
        except Exception as e:
            if not self.headless:
                print(f"âŒ Local processor error: {e}")
            return None, 0, None
        if not result or result.get("error"):
            return None, 0, None
        cl = result.get("classification") or {}
        cat = cl.get("category") or cl.get("predicted_category")
        if not cat:
            return None, 0, None
        category = str(cat).replace("_", " ")
        total_months = 0
        for exp in result.get("experience") or []:
            duration_str = exp.get("duration") or ""
            if duration_str:
                total_months += self.parse_duration_to_months(duration_str)
        total_years = total_months / 12.0
        extracted_data = None
        raw_skills = result.get("skills")
        if raw_skills is not None or result.get("experience"):
            if isinstance(raw_skills, list):
                skills_list = raw_skills
            elif isinstance(raw_skills, dict):
                skills_list = []
                for v in raw_skills.values():
                    (
                        skills_list.extend(v)
                        if isinstance(v, list)
                        else skills_list.append(v)
                    )
            else:
                skills_list = []
            extracted_data = {
                "personal_info": result.get("personal_info") or {},
                "skills": skills_list,
                "experience": result.get("experience") or [],
                "education": result.get("education") or [],
                "projects": result.get("projects", []) if "projects" in result else [],
            }
        if not self.headless:
            print(f"âœ“ Found category (local model): {category}")
        return category, total_years, extracted_data

    def build_search_query(self, category, years_experience):
        """Build appropriate search query based on experience"""
        rounded_years = math.ceil(years_experience)

        if rounded_years == 0:
            query = f"entry level {category}"
        elif rounded_years == 1:
            query = f"{category} 0-1 year experience"
        else:
            query = f"{category} {rounded_years} years experience"

        return query, rounded_years

    def _build_search_queries(
        self, category, years_experience, extracted_data, base_query
    ):
        """Build multi-query set using optional LLM query expansion."""
        queries = [base_query]

        if not self.use_llm_query_expansion or not self.llm_intelligence:
            return queries

        role_family = "General"
        seniority = "Entry"
        key_signals = []
        if isinstance(extracted_data, dict):
            role_family = extracted_data.get("llm_role_family", "General")
            seniority = extracted_data.get("llm_seniority", "Entry")
            key_signals = extracted_data.get("llm_signals", []) or []

        expansion = self.llm_intelligence.generate_search_queries(
            category=category,
            years_experience=years_experience,
            role_family=role_family,
            seniority=seniority,
            key_signals=key_signals,
            max_queries=self.max_expansion_queries,
        )
        if not expansion.get("success"):
            return queries

        seen = {q.lower().strip() for q in queries}
        for query in expansion.get("queries", []):
            q = str(query).strip()
            if not q:
                continue
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            queries.append(q)

        return queries

    def _job_dedupe_key(self, job):
        """Build stable dedupe key across queries."""
        apply_link = (job.get("apply_link") or "").strip().lower()
        if apply_link:
            return apply_link
        title = (job.get("title") or "").strip().lower()
        company = (job.get("company") or "").strip().lower()
        location = (job.get("location") or "").strip().lower()
        return f"{title}|{company}|{location}"

    def scrape_google_jobs_multi_query(self, queries, location="India", max_jobs=10):
        """Scrape jobs for multiple queries and de-duplicate results."""
        if not queries:
            return []

        unique_jobs = {}
        for query in queries:
            jobs = self.scrape_google_jobs_serpapi(
                query, location=location, max_jobs=max_jobs
            )
            for job in jobs:
                key = self._job_dedupe_key(job)
                existing = unique_jobs.get(key)
                if not existing:
                    unique_jobs[key] = job
                else:
                    # Prefer richer descriptions when duplicate appears.
                    existing_desc = existing.get("description", "") or ""
                    new_desc = job.get("description", "") or ""
                    if len(new_desc) > len(existing_desc):
                        unique_jobs[key] = job

        merged = list(unique_jobs.values())
        # Keep recent-first order approximation where posted info exists.
        merged.sort(
            key=lambda x: self.parse_time_posted(str(x.get("posted_at", ""))),
            reverse=True,
        )
        return merged[: max(max_jobs * 2, max_jobs)]

    def _resolve_experience_years(self, parsed_years, extracted_data):
        """Prefer LLM-estimated years when available; fallback to parsed years."""
        if isinstance(extracted_data, dict):
            llm_years = extracted_data.get("llm_years_experience", 0)
            try:
                llm_years = float(llm_years)
            except Exception:
                llm_years = 0
            if llm_years > 0:
                self.last_years_source = "llm_extracted"
                return llm_years
        if parsed_years and parsed_years > 0:
            self.last_years_source = "parsed_resume"
            return parsed_years
        self.last_years_source = "default_zero"
        return 0.0

    def parse_time_posted(self, posted_str):
        """Convert 'X hours ago', 'X days ago' to comparable timestamp"""
        try:
            posted_lower = posted_str.lower()
            now = datetime.now()

            if "just now" in posted_lower or "today" in posted_lower:
                return now

            numbers = re.findall(r"\d+", posted_str)
            if not numbers:
                return now - timedelta(days=365)

            value = int(numbers[0])

            if "minute" in posted_lower or "min" in posted_lower:
                return now - timedelta(minutes=value)
            elif "hour" in posted_lower:
                return now - timedelta(hours=value)
            elif "day" in posted_lower:
                return now - timedelta(days=value)
            elif "week" in posted_lower:
                return now - timedelta(weeks=value)
            elif "month" in posted_lower:
                return now - timedelta(days=value * 30)
            else:
                return now - timedelta(days=365)
        except:
            return datetime.now() - timedelta(days=365)

    def clean_job_data(self, job_data):
        """Remove empty, N/A, or 'Not specified' fields from job data"""
        cleaned = {}

        for key, value in job_data.items():
            if value in ["N/A", "Not specified", "", "Recently", None]:
                continue
            if isinstance(value, list) and len(value) == 0:
                continue
            if isinstance(value, bool) and value is False:
                continue
            if key == "posted_timestamp":
                continue

            cleaned[key] = value

        return cleaned

    def _flatten_job_highlights(self, job_highlights):
        """Flatten SerpAPI job highlights into plain text bullet-like string."""
        if not isinstance(job_highlights, list):
            return ""
        parts = []
        for block in job_highlights:
            if not isinstance(block, dict):
                continue
            title = str(block.get("title", "")).strip()
            items = block.get("items", [])
            if isinstance(items, list):
                item_text = ", ".join(
                    str(it).strip() for it in items if str(it).strip()
                )
            else:
                item_text = str(items).strip()
            merged = " - ".join(p for p in [title, item_text] if p)
            if merged:
                parts.append(merged)
        return " | ".join(parts)

    def _compose_job_text(self, job):
        """Create richer text representation for matching from existing scraped fields."""
        title = str(job.get("title", "") or "")
        description = str(job.get("description", "") or "")
        highlights = self._flatten_job_highlights(job.get("job_highlights", []))
        schedule = str(job.get("schedule_type", "") or "")
        via = str(job.get("via", "") or "")
        return " ".join(
            part for part in [title, description, highlights, schedule, via] if part
        )

    def scrape_google_jobs_serpapi(self, search_query, location="India", max_jobs=10):
        """Use SerpAPI to scrape Google Jobs with experience-based query"""
        jobs = []
        try:
            url = "https://serpapi.com/search"
            params = {
                "engine": "google_jobs",
                "q": search_query,
                "location": location,
                "api_key": self.serpapi_key,
                "hl": "en",
                "gl": "in",
            }

            response = requests.get(url, params=params, timeout=20)

            if response.status_code != 200:
                return jobs

            data = response.json()

            if "jobs_results" in data and isinstance(data["jobs_results"], list):
                for job in data["jobs_results"]:
                    extensions = job.get("detected_extensions", {})
                    apply_options = job.get("apply_options", [])
                    apply_link = (
                        apply_options[0].get("link")
                        if apply_options
                        else job.get("share_url", "")
                    )
                    related_links = job.get("related_links", [])

                    posted_at_str = extensions.get("posted_at", "")
                    highlight_text = self._flatten_job_highlights(
                        job.get("job_highlights", [])
                    )
                    raw_description = job.get("description", "") or ""
                    enriched_description = raw_description
                    if highlight_text:
                        enriched_description = (
                            f"{raw_description} Highlights: {highlight_text}"
                            if raw_description
                            else highlight_text
                        )

                    job_data = {
                        "title": job.get("title", ""),
                        "company": job.get("company_name", ""),
                        "location": job.get("location", ""),
                        "via": job.get("via", ""),
                        "posted_at": posted_at_str,
                        "posted_timestamp": self.parse_time_posted(posted_at_str),
                        "schedule_type": extensions.get("schedule_type", ""),
                        "salary": extensions.get("salary", ""),
                        "work_from_home": extensions.get("work_from_home", False),
                        "apply_link": apply_link,
                        "description": (
                            enriched_description[:650] + "..."
                            if enriched_description
                            else ""
                        ),
                        "job_highlights": job.get("job_highlights", []),
                        "related_links": [
                            {
                                "title": link.get("text", ""),
                                "link": link.get("link", ""),
                            }
                            for link in related_links[:3]
                            if link.get("link")
                        ],
                    }

                    cleaned_job = self.clean_job_data(job_data)
                    cleaned_job["posted_timestamp"] = job_data["posted_timestamp"]

                    if cleaned_job:
                        jobs.append(cleaned_job)

                jobs.sort(
                    key=lambda x: x.get("posted_timestamp", datetime.now()),
                    reverse=True,
                )

                for job in jobs:
                    if "posted_timestamp" in job:
                        del job["posted_timestamp"]

                jobs = jobs[:max_jobs]

        except Exception as e:
            if not self.headless:
                print(f"SerpAPI Error: {e}")

        return jobs

    def _normalize_score(self, value, min_value, max_value):
        """Normalize a value to 0-1 range; return 0.5 when range is flat."""
        span = max_value - min_value
        if abs(span) < 1e-9:
            return 0.5
        normalized = (value - min_value) / span
        return max(0.0, min(1.0, normalized))

    def _calculate_skill_overlap(self, resume_skills, job):
        """Estimate skill overlap ratio between resume skills and job text."""
        if not resume_skills:
            return 0.0, 0, 0

        unique_skills = []
        seen = set()
        for skill in resume_skills:
            skill_text = str(skill).strip().lower()
            if len(skill_text) < 2:
                continue
            if skill_text in seen:
                continue
            seen.add(skill_text)
            unique_skills.append(skill_text)

        if not unique_skills:
            return 0.0, 0, 0

        job_text = self._compose_job_text(job).lower()
        matched = sum(1 for skill in unique_skills if skill in job_text)
        total = len(unique_skills)
        return matched / total, matched, total

    def _tokenize_role(self, category):
        """Tokenize role/category into meaningful lowercased terms."""
        stopwords = {
            "and",
            "or",
            "the",
            "of",
            "to",
            "for",
            "in",
            "with",
            "engineer",
            "developer",
        }
        tokens = re.findall(r"[a-z0-9]+", str(category).lower())
        return [t for t in tokens if len(t) > 2 and t not in stopwords]

    def _role_title_alignment(self, category, job):
        """Compute alignment between inferred role and job title/description."""
        title = str(job.get("title", "") or "").lower()
        full_text = self._compose_job_text(job).lower()
        category_l = str(category or "").strip().lower()
        if not category_l:
            return 0.5

        if category_l in title:
            return 1.0

        role_tokens = self._tokenize_role(category_l)
        if not role_tokens:
            return 0.5

        title_hits = sum(1 for t in role_tokens if t in title)
        text_hits = sum(1 for t in role_tokens if t in full_text)

        title_ratio = title_hits / len(role_tokens)
        text_ratio = text_hits / len(role_tokens)
        return max(0.0, min(1.0, (title_ratio * 0.75) + (text_ratio * 0.35)))

    def _extract_year_range_from_text(self, text):
        """Parse likely years-of-experience requirement from job text."""
        if not text:
            return None, None

        text = text.lower().replace("–", "-").replace("—", "-")
        patterns = [
            r"(\d+)\s*-\s*(\d+)\s*(?:\+)?\s*(?:years|yrs)",
            r"(\d+)\s*(?:\+)\s*(?:years|yrs)",
            r"minimum\s+(\d+)\s*(?:years|yrs)",
            r"at\s+least\s+(\d+)\s*(?:years|yrs)",
            r"(\d+)\s*(?:years|yrs)\s*(?:of)?\s*experience",
        ]

        for pattern in patterns:
            m = re.search(pattern, text)
            if not m:
                continue
            if len(m.groups()) >= 2 and m.group(2):
                low = float(m.group(1))
                high = float(m.group(2))
                if high < low:
                    low, high = high, low
                return low, high
            val = float(m.group(1))
            if "+" in m.group(0) or "minimum" in m.group(0) or "at least" in m.group(0):
                return val, None
            return max(0.0, val - 1.0), val + 1.0

        return None, None

    def _title_level_expected_years(self, title):
        """Estimate expected experience range from title seniority keywords."""
        t = str(title or "").lower()
        if any(k in t for k in ["intern", "trainee"]):
            return 0.0, 1.0
        if any(k in t for k in ["entry", "fresher", "junior", "jr"]):
            return 0.0, 2.0
        if any(
            k in t
            for k in ["lead", "principal", "staff", "architect", "manager", "head"]
        ):
            return 5.0, None
        if "senior" in t or "sr" in t:
            return 3.0, None
        if any(k in t for k in ["mid", "associate"]):
            return 2.0, 5.0
        return None, None

    def _experience_alignment(self, years_experience, job):
        """Assess how candidate years align with job expected level."""
        try:
            years = float(years_experience or 0.0)
        except Exception:
            years = 0.0

        if years <= 0:
            return 0.55

        text = self._compose_job_text(job)
        low_text, high_text = self._extract_year_range_from_text(text)
        low_title, high_title = self._title_level_expected_years(job.get("title", ""))

        ranges = []
        if low_text is not None or high_text is not None:
            ranges.append((low_text, high_text))
        if low_title is not None or high_title is not None:
            ranges.append((low_title, high_title))
        if not ranges:
            return 0.6

        def score_for_range(low, high):
            if low is None and high is None:
                return 0.6
            if low is None:
                return max(0.0, min(1.0, 1.0 - max(0.0, years - high) / 6.0))
            if high is None:
                if years >= low:
                    return min(1.0, 0.7 + min(0.3, (years - low) / 8.0))
                return max(0.0, 1.0 - ((low - years) / 4.0))
            if low <= years <= high:
                return 1.0
            if years < low:
                return max(0.0, 1.0 - ((low - years) / 4.0))
            return max(0.0, 1.0 - ((years - high) / 6.0))

        scores = [score_for_range(low, high) for low, high in ranges]
        return max(0.0, min(1.0, sum(scores) / len(scores)))

    def _recency_component(self, posted_at):
        """Light recency preference from posted text."""
        text = str(posted_at or "").lower()
        if not text:
            return 0.55
        if "hour" in text or "today" in text or "just now" in text:
            return 1.0
        if "day" in text:
            m = re.search(r"(\d+)", text)
            days = int(m.group(1)) if m else 1
            if days <= 2:
                return 0.92
            if days <= 7:
                return 0.8
            return 0.68
        if "week" in text:
            return 0.6
        if "month" in text:
            return 0.45
        return 0.55

    def _calibrate_match_score(
        self,
        similarity,
        similarity_min,
        similarity_max,
        rank_index,
        total_jobs,
        overlap_ratio,
        category,
        years_experience,
        job,
    ):
        """Calibrate raw semantic similarity into a stable, user-friendly score."""
        semantic_component = self._normalize_score(
            similarity, similarity_min, similarity_max
        )
        title_component = self._role_title_alignment(category, job)
        experience_component = self._experience_alignment(years_experience, job)
        recency_component = self._recency_component(job.get("posted_at", ""))

        if total_jobs <= 1:
            rank_component = 1.0
        else:
            rank_component = 1.0 - (rank_index / (total_jobs - 1))

        calibrated = (
            18
            + (semantic_component * 34)
            + (rank_component * 14)
            + (overlap_ratio * 11)
            + (title_component * 16)
            + (experience_component * 11)
            + (recency_component * 8)
        )

        # Mild penalties for very weak semantic alignment.
        if similarity < 0.15:
            calibrated -= 6
        if similarity < 0.05:
            calibrated -= 6

        score = int(max(10, min(98, round(calibrated))))
        score = int(max(10, min(98, score + self.match_score_boost)))
        return score

    def rank_jobs_with_rag(
        self, resume_text, jobs, resume_skills, category=None, years_experience=0.0
    ):
        """Rank jobs using semantic similarity and calibrated scoring."""
        if not self.rag_engine:
            return [
                (job, 60, "Profile fit estimated with fallback ranking") for job in jobs
            ]

        try:
            if not self.headless:
                print("Calculating semantic match scores...")

            ranked_jobs = self.rag_engine.rank_jobs_by_similarity(resume_text, jobs)
            if not ranked_jobs:
                return []

            similarities = [similarity for _, similarity in ranked_jobs]
            similarity_min = min(similarities)
            similarity_max = max(similarities)
            total_jobs = len(ranked_jobs)

            enhanced_jobs = []
            for rank_index, (job, similarity) in enumerate(ranked_jobs):
                overlap_ratio, matched_skills, total_skills = (
                    self._calculate_skill_overlap(resume_skills, job)
                )

                match_score = self._calibrate_match_score(
                    similarity,
                    similarity_min,
                    similarity_max,
                    rank_index,
                    total_jobs,
                    overlap_ratio,
                    category,
                    years_experience,
                    job,
                )

                job_copy = job.copy()
                job_copy["semantic_similarity"] = round(max(0.0, similarity) * 100, 2)

                if self.resume_analyzer and match_score >= 78:
                    try:
                        explanation = self.resume_analyzer.generate_match_explanation(
                            job_copy.get("title", ""),
                            job_copy.get("description", ""),
                            resume_skills[:10],
                        )
                    except Exception:
                        explanation = "Strong semantic fit based on role context"
                else:
                    explanation = "Semantic fit based on resume context"

                enhanced_jobs.append((job_copy, match_score, explanation))

            # Optional second-stage rerank with LLM for better precision.
            enhanced_jobs = self._apply_llm_rerank(
                resume_text=resume_text, category=category, ranked_jobs=enhanced_jobs
            )

            return enhanced_jobs

        except Exception as e:
            if not self.headless:
                print(f"RAG ranking error: {e}")
            return [
                (job, 60, "Profile fit estimated with fallback ranking") for job in jobs
            ]

    def _apply_llm_rerank(self, resume_text, category, ranked_jobs):
        """Apply LLM reranking on top semantic candidates and blend scores."""
        if not self.use_llm_rerank or not self.llm_intelligence:
            return ranked_jobs
        if not ranked_jobs:
            return ranked_jobs

        top_k = min(self.llm_rerank_top_k, len(ranked_jobs))
        head = ranked_jobs[:top_k]
        tail = ranked_jobs[top_k:]
        candidate_jobs = [item[0] for item in head]

        rerank = self.llm_intelligence.rerank_jobs(
            resume_text=resume_text,
            category=category or "General",
            jobs=candidate_jobs,
            max_jobs=top_k,
        )
        if not rerank.get("success"):
            return ranked_jobs

        index_to_row = {idx: row for idx, row in enumerate(head)}
        reranked_head = []
        used = set()

        for item in rerank.get("ranking", []):
            if not isinstance(item, dict):
                continue
            idx = item.get("job_index")
            if not isinstance(idx, int):
                continue
            if idx in used or idx not in index_to_row:
                continue
            used.add(idx)

            job, semantic_score, semantic_reason = index_to_row[idx]
            llm_score = item.get("score", semantic_score)
            try:
                llm_score = float(llm_score)
            except Exception:
                llm_score = float(semantic_score)
            llm_score = max(0.0, min(100.0, llm_score))

            blended_score = int(round((semantic_score * 0.72) + (llm_score * 0.28)))
            blended_score += self.match_score_boost
            blended_score = max(10, min(98, blended_score))

            reason = (item.get("reason") or "").strip()
            if reason:
                explanation = f"{semantic_reason} | LLM rerank: {reason}"
            else:
                explanation = semantic_reason

            job_copy = job.copy()
            job_copy["llm_rerank_score"] = int(round(llm_score))
            reranked_head.append((job_copy, blended_score, explanation))

        # Append any candidates omitted by LLM output in original order.
        for idx, row in enumerate(head):
            if idx not in used:
                reranked_head.append(row)

        final_results = reranked_head + tail

        # Sort final results by blended score (highest first)
        final_results.sort(key=lambda x: x[1], reverse=True)

        return final_results

    def process_resume_and_scrape_jobs(
        self, resume_path, extracted_data=None, max_jobs=10
    ):
        """Main method with RAG integration. Gets extracted_data from main.py (--json)
        so Claude and RAG receive real resume content. extracted_data param is for
        backward compatibility only; when None, it is obtained from get_resume_data.
        """
        if not self.headless:
            print("ðŸ”„ Analyzing resume...")

        category, years_experience, extracted_from_main = self.get_resume_data(
            resume_path
        )
        if extracted_data is None:
            extracted_data = extracted_from_main

        if not category:
            if self.headless:
                reason = (
                    self.last_category_failure_reason
                    or "Automatic extraction/categorization failed."
                )
                raise Exception(
                    "Could not identify job category from resume. Automatic extraction or categorization failed.\n"
                    f"Reason: {reason}"
                )
            else:
                print("\nâŒ Failed to extract category automatically")
                category = input("Please enter job category manually: ").strip()
                if not category:
                    print("No category provided. Exiting.")
                    return {}, [], None, 0, 0

        years_experience = self._resolve_experience_years(
            years_experience, extracted_data
        )
        search_query, rounded_years = self.build_search_query(
            category, years_experience
        )
        search_queries = self._build_search_queries(
            category=category,
            years_experience=years_experience,
            extracted_data=extracted_data,
            base_query=search_query,
        )
        self.last_search_queries = search_queries

        if not self.headless:
            print(f"\nðŸ“Š Category: {category}")
            print(
                f"ðŸ’¼ Experience: {years_experience:.1f} years â†’ Rounded to {rounded_years} years"
            )
            print(f"ðŸ” Searching: '{search_query}'")
            if len(search_queries) > 1:
                print(f"ðŸ”Ž Expanded queries: {len(search_queries)}")
            print()

        # Scrape jobs (single or multi-query depending on LLM expansion).
        all_jobs = self.scrape_google_jobs_multi_query(
            search_queries, location="India", max_jobs=max_jobs
        )

        # Generate resume analysis only in interactive CLI mode.
        # Web flow runs this script headless and triggers analysis on-demand via /analyze-resume.
        resume_analysis = {}
        if self.resume_analyzer and extracted_data and not self.headless:
            try:
                print("Generating resume improvement tips...")
                resume_analysis = self.resume_analyzer.analyze_resume(
                    extracted_data, category
                )
            except Exception as e:
                print(f"Resume analysis error: {e}")

        # Rank jobs with RAG
        resume_text = ""
        resume_skills = []
        if extracted_data:
            # Build resume text for semantic matching
            resume_skills = extracted_data.get("skills", [])
            experiences = extracted_data.get("experience", [])
            projects = extracted_data.get("projects", [])

            resume_parts = [
                " ".join(resume_skills),
                " ".join([exp.get("role", "") for exp in experiences]),
                " ".join([proj.get("name", "") for proj in projects]),
            ]
            resume_text = " ".join(resume_parts)

        ranked_jobs = self.rank_jobs_with_rag(
            resume_text,
            all_jobs,
            resume_skills,
            category=category,
            years_experience=years_experience,
        )

        return (
            resume_analysis,
            ranked_jobs,
            category,
            years_experience,
            rounded_years,
            extracted_data,
        )

    def save_results(
        self,
        jobs,
        category,
        years_experience,
        rounded_years,
        resume_analysis=None,
        extracted_data=None,
    ):
        """Build results payload; optionally persist to disk for debugging."""
        today = datetime.now().strftime("%Y-%m-%d")

        # Format jobs with match scores
        formatted_jobs = []
        for job, match_score, explanation in jobs:
            job_copy = job.copy()
            job_copy["match_score"] = match_score
            job_copy["match_explanation"] = explanation
            formatted_jobs.append(job_copy)

        # Keep only the pieces we need for UI + Claude on-demand (avoid personal info leakage)
        resume_extracted_data = None
        if extracted_data:
            resume_extracted_data = {
                "skills": extracted_data.get("skills", []),
                "experience": extracted_data.get("experience", []),
                "education": extracted_data.get("education", []),
                "projects": extracted_data.get("projects", []),
            }

        results = {
            "search_category": category,
            "total_experience_years": round(years_experience, 1),
            "total_experience_months": int(round(max(0.0, years_experience) * 12)),
            "experience_display": self._format_experience_label(years_experience),
            "search_experience_years": rounded_years,
            "scrape_date": today,
            "scrape_time": datetime.now().strftime("%I:%M %p"),
            "total_jobs": len(formatted_jobs),
            "jobs": formatted_jobs,
            "resume_analysis": resume_analysis or {},
            "resume_extracted_data": resume_extracted_data or {},
            "rag_enabled": bool(self.rag_engine),
            "rag_embedding_provider": (
                getattr(self.rag_engine, "embedding_provider", "unknown")
                if self.rag_engine
                else "none"
            ),
            "claude_enabled": bool(self.resume_analyzer),
            "llm_provider": self.llm_provider,
            "llm_category_enabled": bool(self.llm_intelligence),
            "llm_category_used": bool(
                isinstance(self.last_llm_prediction, dict)
                and self.last_llm_prediction.get("success")
            ),
            "category_source": (
                "fireworks_llm"
                if (
                    isinstance(self.last_llm_prediction, dict)
                    and self.last_llm_prediction.get("success")
                )
                else "legacy_classifier"
            ),
            "category_confidence": (
                self.last_llm_prediction.get("confidence")
                if isinstance(self.last_llm_prediction, dict)
                else None
            ),
            "category_role_family": (
                self.last_llm_prediction.get("role_family")
                if isinstance(self.last_llm_prediction, dict)
                else None
            ),
            "category_seniority": (
                self.last_llm_prediction.get("seniority")
                if isinstance(self.last_llm_prediction, dict)
                else None
            ),
            "category_years_experience": (
                self.last_llm_prediction.get("years_experience")
                if isinstance(self.last_llm_prediction, dict)
                else None
            ),
            "category_model": (
                self.last_llm_prediction.get("model_used")
                if isinstance(self.last_llm_prediction, dict)
                else None
            ),
            "llm_model": (
                (
                    self.last_llm_prediction.get("model_used")
                    if isinstance(self.last_llm_prediction, dict)
                    else None
                )
                or os.getenv("FIREWORKS_PRIMARY_CHAT_MODEL")
                or os.getenv("FIREWORKS_CHAT_MODEL")
                or "fireworks/minimax-m2p7"
            ),
            # Backward compatibility keys retained for older consumers.
            "bedrock_category_enabled": False,
            "bedrock_category_used": False,
            "experience_years_source": self.last_years_source,
            "rerank_enabled": bool(self.use_llm_rerank),
            "search_queries_used": self.last_search_queries or [],
            "timestamp": datetime.now().isoformat(),
            "source": "Google Jobs via SerpAPI + RAG",
            "sorted_by": "Semantic match score",
        }

        should_save_file = os.getenv("SAVE_RESULTS_TO_FILE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if should_save_file:
            safe_category = re.sub(r"[^a-z0-9]+", "_", str(category).lower()).strip("_")
            if not safe_category:
                safe_category = "unknown_role"
            filename = f"{safe_category}_{rounded_years}yrs_{today}.json"
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            results["results_file"] = filename

        return results


def main():
    forced_headless = os.getenv("RESUME_HEADLESS", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    is_headless = forced_headless or (not sys.stdin.isatty())

    if len(sys.argv) != 2:
        print("Usage: python enhanced_job_scraper.py <resume_file_path>")
        return

    resume_path = sys.argv[1]

    if not os.path.exists(resume_path):
        print(f"âŒ File not found: {resume_path}")
        return

    scraper = EnhancedJobScraper(headless=is_headless)

    try:
        (
            resume_analysis,
            ranked_jobs,
            category,
            years_exp,
            rounded_years,
            extracted_data,
        ) = scraper.process_resume_and_scrape_jobs(resume_path)

        if ranked_jobs and category:
            results_payload = scraper.save_results(
                ranked_jobs,
                category,
                years_exp,
                rounded_years,
                resume_analysis,
                extracted_data,
            )
            print(f"Scraped {len(ranked_jobs)} jobs")
            print("RESULT_JSON:" + json.dumps(results_payload, ensure_ascii=False))

            # Show top matches
            if not is_headless:
                print("\nTop Matches:")
                for i, (job, score, explanation) in enumerate(ranked_jobs[:3], 1):
                    print(f"{i}. {job.get('title')} - {score}% match")
                    print(f"   {explanation}")
                if results_payload.get("results_file"):
                    print(f"Saved file: {results_payload['results_file']}")
        else:
            print("No matching jobs found")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
