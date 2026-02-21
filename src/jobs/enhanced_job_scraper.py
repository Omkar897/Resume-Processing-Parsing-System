"""
Enhanced Job Scraper with RAG Integration
Adds semantic job matching and resume analysis.

FLOW (when you upload a resume on the web app):
1. Get category + experience: try main.py --json (APILayer + rule-based) -> else parse main.py stdout -> else local PDF + embedding classifier.
2. Build search query from category + years (e.g. "Data Scientist 2 years experience").
3. Fetch up to 10 jobs from SerpAPI (Google Jobs).
4. If we have extracted_data: Claude analyzes resume (tips, ATS score); RAG ranks jobs by semantic similarity and adds match explanations.
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

        # RAG engine (local embeddings + ChromaDB)
        try:
            persist_dir = os.path.abspath(
                os.path.join(self.project_root, "data", "chromadb")
            )
            os.makedirs(persist_dir, exist_ok=True)
            # Keep HuggingFace/transformers cache inside project so subprocess has a valid path
            cache_dir = os.path.abspath(os.path.join(self.project_root, "data", ".embedding_cache"))
            os.makedirs(cache_dir, exist_ok=True)
            os.environ["TRANSFORMERS_CACHE"] = cache_dir
            os.environ["HF_HOME"] = cache_dir
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

    def parse_duration_to_months(self, duration_str):
        """Parse duration string like 'June-Aug 2024' or 'Jan-Feb 2024' to months"""
        try:
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

            duration_lower = duration_str.lower().strip()

            if "-" in duration_str:
                parts = duration_str.replace("â€“", "-").split("-")
                start_match = re.search(r"([a-zA-Z]+)\s*(\d{4})?", parts[0])
                end_match = re.search(r"([a-zA-Z]+)\s*(\d{4})", parts[1])

                if start_match and end_match:
                    start_month_str = start_match.group(1).lower()[:3]
                    end_month_str = end_match.group(1).lower()[:3]

                    start_month = month_map.get(start_month_str, 1)
                    end_month = month_map.get(end_month_str, 12)

                    months = end_month - start_month + 1
                    return max(months, 1)

            return 3
        except:
            return 3

    def get_resume_data(self, resume_path):
        """Extract category, total experience, and structured extracted_data from resume.
        Tries in order: (1) main.py --json for full data + RAG, (2) main.py stdout parsing,
        (3) local src/resume processor + classifier when APILayer is unavailable.
        Returns (category, total_years, extracted_data). extracted_data may be None.
        """
        project_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..")
        )
        main_script_path = os.path.join(project_root, "main.py")

        # ---- 1) Try main.py with --json (APILayer + rule-based classifier, full extracted_data for RAG) ----
        category, total_years, extracted_data = self._get_resume_data_via_main_json(
            main_script_path, resume_path, project_root
        )
        if category is not None:
            return category, total_years, extracted_data

        # ---- 2) Fallback: main.py without --json, parse human-readable stdout ----
        if not self.headless:
            print("âš ï¸ JSON path failed, trying legacy stdout parsing...")
        category, total_years = self._get_resume_data_via_main_stdout(
            main_script_path, resume_path, project_root
        )
        if category is not None:
            return (
                category,
                total_years,
                self._build_minimal_extracted_data(category, total_years),
            )

        # ---- 3) Fallback: local PDF extraction + embedding classifier (no APILayer) ----
        if not self.headless:
            print("âš ï¸ main.py failed, trying local resume processor...")
        category, total_years, extracted_data = self._get_resume_data_via_local_processor(
            resume_path, project_root
        )
        if category is not None:
            return category, total_years, extracted_data

        if not self.headless:
            print("âŒ All resume extraction paths failed")
        return None, 0, None


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
            "experience": [
                {
                    "role": category or "Not specified",
                    "company": "Not specified",
                    "duration": duration,
                    "description": "Generated from fallback category parsing.",
                }
            ]
            if category
            else [],
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
                print(f"âœ“ Found category: {category}" + (" (full data for RAG)" if extracted_data else ""))
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
                        total_months += self.parse_duration_to_months(m.group(1).strip())
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
                    skills_list.extend(v) if isinstance(v, list) else skills_list.append(v)
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
                            job.get("description", "")[:400] + "..."
                            if job.get("description")
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

    def rank_jobs_with_rag(self, resume_text, jobs, resume_skills):
        """Rank jobs using semantic similarity"""
        if not self.rag_engine:
            # Fallback: return jobs as-is
            return [
                (job, 75, "RAG unavailable (embedding engine not initialized)")
                for job in jobs
            ]

        try:
            if not self.headless:
                print("ðŸ§  Calculating semantic match scores...")

            ranked_jobs = self.rag_engine.rank_jobs_by_similarity(resume_text, jobs)

            # Add match explanations
            enhanced_jobs = []
            for job, similarity in ranked_jobs:
                # Convert similarity to percentage
                match_score = int(similarity * 100)

                # Generate explanation (only for top jobs to save API calls)
                if self.resume_analyzer and match_score >= 70:
                    try:
                        explanation = self.resume_analyzer.generate_match_explanation(
                            job.get("title", ""),
                            job.get("description", ""),
                            resume_skills[:10],
                        )
                    except:
                        explanation = f"Strong match based on your skills and experience"
                else:
                    explanation = f"Matches your {len(resume_skills)} skills"

                enhanced_jobs.append((job, match_score, explanation))

            return enhanced_jobs

        except Exception as e:
            if not self.headless:
                print(f"âš ï¸ RAG ranking error: {e}")
            # Fallback
            return [(job, 75, "Matches your profile") for job in jobs]

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
                raise Exception(
                    "Could not identify job category from resume. Please ensure your resume contains clear work experience, job titles, and skills sections."
                )
            else:
                print("\nâŒ Failed to extract category automatically")
                category = input("Please enter job category manually: ").strip()
                if not category:
                    print("No category provided. Exiting.")
                    return {}, [], None, 0, 0

        search_query, rounded_years = self.build_search_query(
            category, years_experience
        )

        if not self.headless:
            print(f"\nðŸ“Š Category: {category}")
            print(
                f"ðŸ’¼ Experience: {years_experience:.1f} years â†’ Rounded to {rounded_years} years"
            )
            print(f"ðŸ” Searching: '{search_query}'")
            print()

        # Scrape jobs
        all_jobs = self.scrape_google_jobs_serpapi(
            search_query, location="India", max_jobs=max_jobs
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

        ranked_jobs = self.rank_jobs_with_rag(resume_text, all_jobs, resume_skills)

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
        """Save results to JSON with RAG enhancements"""
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
            "search_experience_years": rounded_years,
            "scrape_date": today,
            "scrape_time": datetime.now().strftime("%I:%M %p"),
            "total_jobs": len(formatted_jobs),
            "jobs": formatted_jobs,
            "resume_analysis": resume_analysis or {},
            "resume_extracted_data": resume_extracted_data or {},
            "rag_enabled": bool(self.rag_engine),
            "claude_enabled": bool(self.resume_analyzer),
            "timestamp": datetime.now().isoformat(),
            "source": "Google Jobs via SerpAPI + RAG",
            "sorted_by": "Semantic match score",
        }

        filename = (
            f"{category.replace(' ', '_').lower()}_{rounded_years}yrs_{today}.json"
        )
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return filename


def main():
    forced_headless = os.getenv("RESUME_HEADLESS", "").strip().lower() in {"1", "true", "yes"}
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
            filename = scraper.save_results(
                ranked_jobs,
                category,
                years_exp,
                rounded_years,
                resume_analysis,
                extracted_data,
            )
            print(f"âœ… Scraped {len(ranked_jobs)} jobs â†’ {filename}")

            # Show top matches
            if not is_headless:
                print("\nðŸŽ¯ Top Matches:")
                for i, (job, score, explanation) in enumerate(ranked_jobs[:3], 1):
                    print(f"{i}. {job.get('title')} - {score}% match")
                    print(f"   {explanation}")
        else:
            print(f"âŒ No matching jobs found")
    except Exception as e:
        print(f"âŒ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
