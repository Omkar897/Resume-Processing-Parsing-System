"""
Resume Analyzer using Claude (Anthropic)
Generates AI-powered resume improvement suggestions
"""

import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None

try:
    from LLM.fireworks_client import FireworksClient
except Exception:
    FireworksClient = None

# Load environment variables
load_dotenv()


class ResumeAnalyzer:
    """AI-powered resume analysis using Claude"""

    def __init__(self):
        """Initialize analyzer in local mode by default; Claude API is optional."""
        self.model = "claude-3-haiku-20240307"
        self.client = None
        self.fireworks_client = None
        self.fireworks_model = (
            os.getenv("FIREWORKS_ANALYZER_MODEL")
            or os.getenv("FIREWORKS_PRIMARY_CHAT_MODEL")
            or os.getenv("FIREWORKS_CHAT_MODEL")
            or "fireworks/minimax-m2p7"
        )
        self.analysis_provider = "local"

        # Keep Claude integration logic, but disable remote calls unless explicitly enabled.
        use_claude_env = os.getenv("USE_CLAUDE_API", "0").strip().lower()
        self.use_claude_api = use_claude_env in {"1", "true", "yes"}
        use_fireworks_env = os.getenv("USE_FIREWORKS_RESUME_ANALYZER", "1").strip().lower()
        self.use_fireworks_api = use_fireworks_env in {"1", "true", "yes"}

        api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
        fireworks_key = os.getenv("FIREWORKS_API_KEY", "").strip()
        if self.use_fireworks_api and FireworksClient is not None and fireworks_key:
            try:
                self.fireworks_client = FireworksClient(api_key=fireworks_key)
                self.analysis_provider = "fireworks"
            except Exception:
                self.fireworks_client = None
                self.use_fireworks_api = False

        if self.use_claude_api and Anthropic is not None and api_key:
            self.client = Anthropic(api_key=api_key)
            if self.analysis_provider == "local":
                self.analysis_provider = "claude"
        elif self.use_claude_api:
            # Graceful downgrade when API is requested but unavailable/misconfigured.
            self.use_claude_api = False

    def analyze_resume(
        self, extracted_data: Dict, predicted_category: str
    ) -> Dict[str, Any]:
        """
        Analyze resume and generate improvement suggestions

        Args:
            extracted_data: Parsed resume data (from APILayer)
            predicted_category: Job category (e.g., "Data Scientist")

        Returns:
            Dictionary with strengths, weaknesses, suggestions, and ATS score
        """
        # Build resume summary for Claude
        resume_summary = self._build_resume_summary(extracted_data)

        if self.use_fireworks_api and self.fireworks_client is not None:
            analysis = self._analyze_with_fireworks(resume_summary, predicted_category)
            if analysis is not None:
                return self._normalize_analysis(
                    analysis, extracted_data, predicted_category, analysis_mode="fireworks"
                )

        if not self.use_claude_api or self.client is None:
            return self._create_dynamic_fallback_analysis(
                extracted_data,
                predicted_category,
                "Claude API disabled (local analysis mode).",
            )

        # Create prompt for Claude
        prompt = f"""You are an expert resume reviewer and career coach. Analyze this resume for a {predicted_category} position.

RESUME SUMMARY:
{resume_summary}

TARGET JOB CATEGORY: {predicted_category}

Please provide:
1. **Strengths** (2-3 key strengths)
2. **Missing Keywords** (3-5 important keywords/skills missing for this category)
3. **Improvement Suggestions** (3-4 actionable tips to improve the resume)
4. **ATS Compatibility Score** (0-100, based on formatting, keywords, and structure)

Format your response as JSON:
{{
  "strengths": ["strength1", "strength2", "strength3"],
  "missing_keywords": ["keyword1", "keyword2", "keyword3"],
  "suggestions": ["suggestion1", "suggestion2", "suggestion3"],
  "ats_score": 75,
  "ats_explanation": "Brief explanation of the score"
}}"""

        try:
            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )

            # Parse response
            response_text = response.content[0].text
            analysis = self._parse_json_response(response_text)

            if analysis is None:
                return self._create_dynamic_fallback_analysis(
                    extracted_data,
                    predicted_category,
                    "Claude response could not be parsed as JSON.",
                )

            return self._normalize_analysis(
                analysis, extracted_data, predicted_category, analysis_mode="claude"
            )

        except Exception as e:
            print(f"⚠️ Claude API error: {e}")
            return self._create_dynamic_fallback_analysis(
                extracted_data, predicted_category, str(e)
            )

    def _analyze_with_fireworks(
        self, resume_summary: str, predicted_category: str
    ) -> Optional[Dict[str, Any]]:
        """Run resume analysis through Fireworks chat completion."""
        if not self.fireworks_client:
            return None

        prompt = f"""You are an expert resume reviewer and career coach. Analyze this resume for a {predicted_category} position.

RESUME SUMMARY:
{resume_summary}

TARGET JOB CATEGORY: {predicted_category}

Return strict JSON only:
{{
  "strengths": ["strength1", "strength2", "strength3"],
  "missing_keywords": ["keyword1", "keyword2", "keyword3"],
  "suggestions": ["suggestion1", "suggestion2", "suggestion3"],
  "ats_score": 75,
  "ats_explanation": "Brief explanation of the score"
}}"""

        messages = [
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": prompt},
        ]

        try:
            parsed, _raw = self.fireworks_client.chat_json(
                model=self.fireworks_model,
                messages=messages,
                max_tokens=700,
                temperature=0.0,
            )
            if isinstance(parsed, dict):
                return parsed
            return None
        except Exception:
            return None

    def generate_match_explanation(
        self, job_title: str, job_description: str, resume_skills: List[str]
    ) -> str:
        """
        Generate explanation for why a job matches the resume

        Args:
            job_title: Job title
            job_description: Job description
            resume_skills: List of skills from resume

        Returns:
            Human-readable explanation
        """
        prompt = f"""Explain in 1-2 sentences why this job matches the candidate's resume.

JOB: {job_title}
DESCRIPTION: {job_description[:300]}...
CANDIDATE SKILLS: {', '.join(resume_skills[:10])}

Be specific and mention matching skills/experience. Keep it concise."""

        if not self.use_claude_api or self.client is None:
            return self._generate_local_match_explanation(
                job_title, job_description, resume_skills
            )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )

            explanation = response.content[0].text.strip()
            return explanation

        except Exception:
            return self._generate_local_match_explanation(
                job_title, job_description, resume_skills
            )

    def _generate_local_match_explanation(
        self, job_title: str, job_description: str, resume_skills: List[str]
    ) -> str:
        """Generate a local explanation without external API calls."""
        jd = f"{job_title} {job_description}".lower()
        normalized_skills = [str(s).strip() for s in (resume_skills or []) if str(s).strip()]
        matched = [s for s in normalized_skills if s.lower() in jd]

        if matched:
            sample = ", ".join(matched[:3])
            return f"Matches your profile based on overlapping skills: {sample}."

        if normalized_skills:
            return (
                f"Role alignment found with your {len(normalized_skills)} listed skills and experience context."
            )

        return "Role alignment found based on category and resume context."

    def _build_resume_summary(self, extracted_data: Dict) -> str:
        """Build a text summary of the resume for Claude"""
        summary_parts = []

        # Personal info
        personal = extracted_data.get("personal_info", {})
        if personal.get("name") != "Not specified":
            summary_parts.append(f"Name: {personal.get('name')}")

        # Skills
        skills = extracted_data.get("skills", [])
        if skills:
            summary_parts.append(f"Skills: {', '.join(skills[:15])}")

        # Experience
        experiences = extracted_data.get("experience", [])
        if experiences:
            summary_parts.append(f"\nExperience ({len(experiences)} entries):")
            for exp in experiences[:3]:  # Top 3 experiences
                summary_parts.append(
                    f"- {exp.get('role')} at {exp.get('company')} ({exp.get('duration')})"
                )

        # Projects
        projects = extracted_data.get("projects", [])
        if projects:
            summary_parts.append(f"\nProjects ({len(projects)} entries):")
            for proj in projects[:3]:  # Top 3 projects
                summary_parts.append(f"- {proj.get('name')} ({proj.get('type')})")

        # Education
        education = extracted_data.get("education", [])
        if education:
            summary_parts.append(f"\nEducation:")
            for edu in education[:2]:
                summary_parts.append(
                    f"- {edu.get('degree')} from {edu.get('institution')}"
                )

        return "\n".join(summary_parts)

    def _parse_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON object from Claude text response."""
        if not response_text:
            return None

        import json

        # Try full response first
        try:
            parsed = json.loads(response_text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        # Try to find JSON block
        start_idx = response_text.find("{")
        end_idx = response_text.rfind("}") + 1
        if start_idx == -1 or end_idx <= start_idx:
            return None

        try:
            parsed = json.loads(response_text[start_idx:end_idx])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _normalize_analysis(
        self,
        analysis: Dict[str, Any],
        extracted_data: Dict[str, Any],
        category: str,
        analysis_mode: str,
    ) -> Dict[str, Any]:
        """Normalize AI output and fill missing fields with data-driven fallback."""
        base = self._create_dynamic_fallback_analysis(extracted_data, category)

        strengths = analysis.get("strengths")
        if isinstance(strengths, list) and strengths:
            base["strengths"] = [str(s).strip() for s in strengths if str(s).strip()][:5]

        missing_keywords = analysis.get("missing_keywords")
        if isinstance(missing_keywords, list) and missing_keywords:
            base["missing_keywords"] = [
                str(k).strip() for k in missing_keywords if str(k).strip()
            ][:6]

        suggestions = analysis.get("suggestions")
        if isinstance(suggestions, list) and suggestions:
            base["suggestions"] = [
                str(s).strip() for s in suggestions if str(s).strip()
            ][:6]

        ats_score = analysis.get("ats_score")
        if isinstance(ats_score, (int, float)):
            base["ats_score"] = max(0, min(100, int(round(float(ats_score)))))

        ats_explanation = analysis.get("ats_explanation")
        if isinstance(ats_explanation, str) and ats_explanation.strip():
            base["ats_explanation"] = ats_explanation.strip()

        base["analysis_mode"] = analysis_mode
        return base

    def _category_keywords(self, category: str) -> List[str]:
        """Return keyword targets based on predicted category."""
        cat = (category or "").strip().lower()
        keywords_map = {
            "data scientist": [
                "python",
                "machine learning",
                "statistics",
                "pandas",
                "sql",
                "tensorflow",
                "pytorch",
                "feature engineering",
                "model deployment",
                "nlp",
            ],
            "data analyst": [
                "sql",
                "excel",
                "power bi",
                "tableau",
                "python",
                "data visualization",
                "statistics",
                "dashboard",
                "etl",
                "reporting",
            ],
            "software engineer": [
                "data structures",
                "algorithms",
                "system design",
                "api",
                "testing",
                "git",
                "oop",
                "debugging",
                "cloud",
                "ci/cd",
            ],
            "frontend": [
                "javascript",
                "typescript",
                "react",
                "css",
                "html",
                "accessibility",
                "redux",
                "next.js",
                "responsive design",
                "performance",
            ],
            "backend": [
                "api",
                "sql",
                "database",
                "microservices",
                "python",
                "java",
                "node.js",
                "redis",
                "docker",
                "testing",
            ],
            "devops": [
                "docker",
                "kubernetes",
                "aws",
                "ci/cd",
                "terraform",
                "linux",
                "monitoring",
                "prometheus",
                "ansible",
                "scripting",
            ],
            "cloud": [
                "aws",
                "azure",
                "gcp",
                "terraform",
                "kubernetes",
                "cloud security",
                "networking",
                "serverless",
                "monitoring",
                "cost optimization",
            ],
            "ai engineer": [
                "python",
                "machine learning",
                "llm",
                "rag",
                "pytorch",
                "tensorflow",
                "vector database",
                "prompt engineering",
                "mlops",
                "api",
            ],
            "fullstack": [
                "javascript",
                "react",
                "node.js",
                "api",
                "sql",
                "docker",
                "testing",
                "cloud",
                "system design",
                "authentication",
            ],
        }

        for known, values in keywords_map.items():
            if known in cat:
                return values

        return [
            "problem solving",
            "sql",
            "python",
            "communication",
            "testing",
            "api",
            "cloud",
            "git",
        ]

    def _extract_skill_strings(self, extracted_data: Dict[str, Any]) -> List[str]:
        """Flatten skills into a lowercased, de-duplicated list."""
        raw_skills = extracted_data.get("skills", []) if isinstance(extracted_data, dict) else []
        skills: List[str] = []

        if isinstance(raw_skills, list):
            skills = [str(s).strip() for s in raw_skills if str(s).strip()]
        elif isinstance(raw_skills, dict):
            for value in raw_skills.values():
                if isinstance(value, list):
                    skills.extend(str(s).strip() for s in value if str(s).strip())
                elif value is not None:
                    v = str(value).strip()
                    if v:
                        skills.append(v)

        # Preserve order while removing duplicates
        seen = set()
        normalized = []
        for skill in skills:
            k = skill.lower()
            if k not in seen:
                seen.add(k)
                normalized.append(skill)
        return normalized

    def _create_dynamic_fallback_analysis(
        self,
        extracted_data: Dict[str, Any],
        category: str,
        error_reason: str = "",
    ) -> Dict[str, Any]:
        """Create variable local analysis when Claude is unavailable."""
        extracted_data = extracted_data or {}
        skills = self._extract_skill_strings(extracted_data)
        skill_set = {s.lower() for s in skills}
        experiences = (
            extracted_data.get("experience", [])
            if isinstance(extracted_data.get("experience"), list)
            else []
        )
        projects = (
            extracted_data.get("projects", [])
            if isinstance(extracted_data.get("projects"), list)
            else []
        )
        education = (
            extracted_data.get("education", [])
            if isinstance(extracted_data.get("education"), list)
            else []
        )

        target_keywords = self._category_keywords(category)
        matched_keywords = [
            kw for kw in target_keywords if any(kw in skill for skill in skill_set)
        ]
        missing_keywords = [kw for kw in target_keywords if kw not in matched_keywords][:5]

        skills_count = len(skills)
        exp_count = len(experiences)
        project_count = len(projects)
        edu_count = len(education)
        match_count = len(matched_keywords)

        ats_score = 42
        ats_score += min(18, skills_count // 2)
        ats_score += min(12, exp_count * 6)
        ats_score += min(10, project_count * 4)
        ats_score += min(8, edu_count * 4)
        ats_score += min(10, match_count * 3)
        if skills_count < 5:
            ats_score -= 8
        if exp_count == 0:
            ats_score -= 6
        if project_count == 0:
            ats_score -= 3
        ats_score = max(35, min(92, int(ats_score)))

        strengths: List[str] = []
        if skills_count >= 15:
            strengths.append(f"Strong skill coverage with {skills_count} listed skills")
        elif skills_count > 0:
            strengths.append(f"Core technical profile is present with {skills_count} listed skills")

        if exp_count > 0:
            strengths.append(f"Work experience section includes {exp_count} role(s)")

        if project_count > 0:
            strengths.append(f"Project portfolio includes {project_count} project(s)")

        if match_count > 0:
            sample = ", ".join(matched_keywords[:3])
            strengths.append(f"Resume already includes category-aligned keywords ({sample})")

        if edu_count > 0:
            strengths.append("Education details are included")

        while len(strengths) < 3:
            strengths.append("Resume has a usable baseline structure for ATS parsing")

        suggestions: List[str] = []
        if missing_keywords:
            suggestions.append(
                f"Add category-targeted keywords such as: {', '.join(missing_keywords[:3])}"
            )

        if exp_count > 0:
            suggestions.append("Quantify impact in experience bullets using metrics and outcomes")
        else:
            suggestions.append("Add an experience section with internships, freelance, or practical work")

        if project_count == 0:
            suggestions.append("Add 1-2 relevant projects with tools used and measurable results")

        if skills_count < 10:
            suggestions.append("Expand technical stack details to include frameworks, tools, and platforms")

        suggestions.append("Use clean ATS-friendly formatting with consistent section headings")
        suggestions = suggestions[:4]

        explanation = (
            "This score is a local estimate based on skills coverage, experience depth, "
            "project evidence, and category keyword alignment."
        )
        if error_reason:
            explanation = (
                "Live Claude analysis is currently unavailable. "
                + explanation
            )

        return {
            "strengths": strengths[:5],
            "missing_keywords": missing_keywords,
            "suggestions": suggestions,
            "ats_score": ats_score,
            "ats_explanation": explanation,
            "analysis_mode": "local_fallback",
        }
