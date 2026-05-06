"""Fireworks-powered resume extraction, query expansion, and reranking."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from src.utils.pdf_extractor import PDFTextExtractor

from .fireworks_client import FireworksClient

load_dotenv()


class FireworksResumeIntelligence:
    """
    Structured resume intelligence with a cost-first model strategy:
    - Primary model: MiniMax M2.7
    - Fallback model: DeepSeek v3.2 on low-quality/invalid output
    """

    def __init__(
        self,
        primary_model: Optional[str] = None,
        fallback_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        rerank_model: Optional[str] = None,
        confidence_threshold: float = 0.58,
    ) -> None:
        self.primary_model = (
            primary_model
            or os.getenv("FIREWORKS_PRIMARY_CHAT_MODEL")
            or os.getenv("FIREWORKS_CHAT_MODEL")
            or "fireworks/minimax-m2p7"
        )
        self.fallback_model = (
            fallback_model
            or os.getenv("FIREWORKS_FALLBACK_CHAT_MODEL")
            or "fireworks/deepseek-v3p2"
        )
        self.embedding_model = (
            embedding_model
            or os.getenv("FIREWORKS_EMBED_MODEL")
            or "fireworks/qwen3-embedding-8b"
        )
        self.rerank_model = (
            rerank_model
            or os.getenv("FIREWORKS_RERANK_MODEL")
            or "fireworks/qwen3-reranker-8b"
        )
        self.confidence_threshold = float(
            os.getenv("FIREWORKS_CLASSIFY_CONF_THRESHOLD", confidence_threshold)
        )

        self.extractor = PDFTextExtractor()
        self.client = FireworksClient()

    def classify_resume_pdf(
        self, pdf_path: str, allowed_categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        extracted = self.extractor.extract_text(pdf_path)
        if "error" in extracted:
            return {"success": False, "error": extracted["error"]}

        text = extracted.get("text", "")
        result = self.classify_resume_text(text, allowed_categories=allowed_categories)
        result["extraction_method"] = extracted.get("method_used")
        result["extracted_text_length"] = extracted.get("text_length", len(text))
        return result

    def classify_resume_text(
        self, resume_text: str, allowed_categories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        cleaned = self._normalize_whitespace(resume_text)
        if not cleaned:
            return {"success": False, "error": "Resume text is empty after extraction"}

        # Keep context bounded and deterministic.
        trimmed = cleaned[:20000]

        primary = self._run_classification(
            model=self.primary_model,
            resume_text=trimmed,
            allowed_categories=allowed_categories,
        )
        if self._is_usable_classification(primary):
            primary["model_used"] = self.primary_model
            primary["fallback_used"] = False
            return primary
        primary_diag = self._diagnose_unusable_classification(primary, "primary")

        fallback = self._run_classification(
            model=self.fallback_model,
            resume_text=trimmed,
            allowed_categories=allowed_categories,
        )
        if self._is_usable_classification(fallback):
            fallback["model_used"] = self.fallback_model
            fallback["fallback_used"] = True
            fallback["primary_failure"] = primary.get(
                "error", "low_confidence_or_incomplete"
            )
            return fallback
        fallback_diag = self._diagnose_unusable_classification(fallback, "fallback")

        return {
            "success": False,
            "error": "Both primary and fallback Fireworks models failed classification",
            "primary_error": primary.get("error") or primary_diag,
            "fallback_error": fallback.get("error") or fallback_diag,
        }

    def generate_search_queries(
        self,
        category: str,
        years_experience: float,
        role_family: str = "General",
        seniority: str = "Entry",
        key_signals: Optional[List[str]] = None,
        max_queries: int = 2,
    ) -> Dict[str, Any]:
        key_signals = key_signals or []
        max_queries = max(1, min(5, int(max_queries)))
        prompt = (
            "Generate compact Google Jobs query variations for this candidate.\n"
            'Return JSON only with schema: {"queries": ["..."]}.\n'
            f"Category: {category}\n"
            f"Role family: {role_family}\n"
            f"Seniority: {seniority}\n"
            f"Years: {years_experience}\n"
            f"Signals: {', '.join(key_signals[:8]) or 'N/A'}\n"
            f"Rules: at most {max_queries} queries, each <= 8 words, no location, no duplicates."
        )
        messages = [
            {"role": "system", "content": "You output strict JSON only."},
            {"role": "user", "content": prompt},
        ]

        parsed, _raw = self.client.chat_json(
            model=self.primary_model,
            messages=messages,
            max_tokens=240,
            temperature=0.0,
        )
        if not isinstance(parsed, dict):
            return {"success": False, "error": "Query expansion returned invalid JSON"}

        raw_queries = parsed.get("queries", [])
        if not isinstance(raw_queries, list):
            return {"success": False, "error": "Query expansion response missing list"}

        seen = set()
        queries: List[str] = []
        for item in raw_queries:
            q = re.sub(r"\s+", " ", str(item).strip())
            if not q:
                continue
            key = q.lower()
            if key in seen:
                continue
            seen.add(key)
            queries.append(q)
            if len(queries) >= max_queries:
                break

        return {"success": True, "queries": queries}

    def rerank_jobs(
        self,
        resume_text: str,
        category: str,
        jobs: List[Dict[str, Any]],
        max_jobs: int = 8,
    ) -> Dict[str, Any]:
        if not jobs:
            return {"success": True, "ranking": []}

        max_jobs = max(1, min(20, int(max_jobs)))
        candidates = jobs[:max_jobs]
        query = f"{category}\n\n{self._normalize_whitespace(resume_text)[:2200]}"
        documents = [self._job_to_document(job) for job in candidates]

        rerank_task = (
            "Given a resume profile query, rank job descriptions by relevance and fit."
        )
        try:
            response = self.client.rerank(
                model=self.rerank_model,
                query=query,
                documents=documents,
                top_n=len(documents),
                return_documents=False,
                task=rerank_task,
            )
        except Exception as exc:
            return {"success": False, "error": f"Fireworks rerank error: {exc}"}

        items = response.get("data", []) if isinstance(response, dict) else []
        if not isinstance(items, list):
            return {"success": False, "error": "Rerank response malformed"}

        normalized = []
        seen = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            if not isinstance(idx, int):
                continue
            if idx < 0 or idx >= len(candidates) or idx in seen:
                continue
            seen.add(idx)
            relevance = item.get("relevance_score", 0.5)
            try:
                relevance = float(relevance)
            except Exception:
                relevance = 0.5
            relevance = max(0.0, min(1.0, relevance))
            normalized.append(
                {
                    "job_index": idx,
                    "score": int(round(relevance * 100)),
                    "reason": "Fireworks reranker relevance.",
                }
            )

        # Ensure deterministic coverage if API omitted entries.
        if len(normalized) < len(candidates):
            missing = [i for i in range(len(candidates)) if i not in seen]
            for idx in missing:
                normalized.append(
                    {
                        "job_index": idx,
                        "score": 50,
                        "reason": "Fallback relevance score.",
                    }
                )

        normalized.sort(key=lambda x: x["score"], reverse=True)
        return {"success": True, "ranking": normalized}

    def _run_classification(
        self, *, model: str, resume_text: str, allowed_categories: Optional[List[str]]
    ) -> Dict[str, Any]:
        taxonomy_hint = ""
        if allowed_categories:
            taxonomy_hint = (
                "Optional preferred taxonomy:\n"
                + "\n".join(f"- {category}" for category in allowed_categories)
                + "\n"
            )

        schema_hint = (
            "{"
            '"category":"specific role title",'
            '"role_family":"Data|Software Engineering|Product|Design|DevOps|QA|Security|Analytics|General",'
            '"seniority":"Intern|Entry|Junior|Mid|Senior|Lead|Manager",'
            '"years_experience":0.0,'
            '"confidence":0.0,'
            '"reasoning":"short evidence-based summary",'
            '"key_signals":["skill1","skill2","skill3"],'
            '"skills":["skillA","skillB"]'
            "}"
        )

        prompt = (
            "Analyze resume text and infer best target job role for search.\n"
            "Use evidence only. Do not invent experience.\n"
            f"{taxonomy_hint}\n"
            f"Return strict JSON only with this schema:\n{schema_hint}\n\n"
            "Rules:\n"
            "- years_experience: numeric 0..40\n"
            "- confidence: numeric 0..1\n"
            "- key_signals and skills: concise arrays\n\n"
            f"Resume text:\n{resume_text}"
        )
        messages = [
            {
                "role": "system",
                "content": "You are an expert resume role-mapping assistant.",
            },
            {"role": "user", "content": prompt},
        ]

        try:
            # Try JSON mode first, fallback to text parsing
            try:
                parsed, raw = self.client.chat_json(
                    model=model,
                    messages=messages,
                    max_tokens=700,
                    temperature=0.0,
                )
                if isinstance(parsed, dict) and parsed.get("category"):
                    return self._build_success_result(parsed, raw)
            except Exception:
                pass

            # Fallback: Use regular chat and parse text
            raw = self.client._post_json(
                "/chat/completions",
                {
                    "model": model,
                    "messages": messages,
                    "max_tokens": 700,
                    "temperature": 0.0,
                },
            )

            content = raw.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = self._parse_category_from_text(content)

            if not isinstance(parsed, dict):
                return {"success": False, "error": "Model response was not valid JSON"}

            category = self._normalize_category(
                parsed.get("category"), allowed_categories=allowed_categories
            )
            if not category:
                return {
                    "success": False,
                    "error": "Model did not return a usable category",
                    "raw_response": parsed,
                }

            confidence = self._normalize_confidence(parsed.get("confidence", 0.6))
            years = self._normalize_years(parsed.get("years_experience", 0.0))
            seniority = self._normalize_seniority(parsed.get("seniority"))
            role_family = self._normalize_role_family(parsed.get("role_family"))
            key_signals = self._normalize_list(parsed.get("key_signals"), max_items=10)
            skills = self._normalize_list(parsed.get("skills"), max_items=40)
            reasoning = str(parsed.get("reasoning", "")).strip()

            if not skills:
                # Fallback: derive few candidate skills from key signals when absent.
                skills = key_signals[:]

            return {
                "success": True,
                "category": category,
                "role_family": role_family,
                "seniority": seniority,
                "years_experience": years,
                "confidence": confidence,
                "reasoning": reasoning,
                "key_signals": key_signals,
                "skills": skills,
                "raw_model_response": raw,
            }
        except Exception as exc:
            return {"success": False, "error": f"Classification call failed: {exc}"}

    def _is_usable_classification(self, result: Dict[str, Any]) -> bool:
        if not isinstance(result, dict) or not result.get("success"):
            return False
        category = str(result.get("category", "")).strip()
        if not category:
            return False
        confidence = self._normalize_confidence(result.get("confidence", 0.55))
        key_signals = result.get("key_signals", []) or []
        skills = result.get("skills", []) or []
        min_accept_conf = float(
            os.getenv("FIREWORKS_CLASSIFY_MIN_ACCEPT_CONF", "0.15")
        )
        if confidence < min_accept_conf and not key_signals and not skills:
            return False
        return True

    def _diagnose_unusable_classification(self, result: Dict[str, Any], label: str) -> str:
        """Return compact reason when a model output exists but is rejected."""
        if not isinstance(result, dict):
            return f"{label}: no_structured_result"
        if not result.get("success"):
            return f"{label}: unsuccessful_result"
        category = str(result.get("category", "")).strip()
        confidence = self._normalize_confidence(result.get("confidence", 0.0))
        key_signals = result.get("key_signals", []) or []
        skills = result.get("skills", []) or []
        return (
            f"{label}: category={'yes' if bool(category) else 'no'}, "
            f"confidence={confidence:.2f}, key_signals={len(key_signals)}, skills={len(skills)}"
        )

    def _parse_category_from_text(self, text: str) -> Dict[str, Any]:
        """Parse category information from text response when JSON mode fails."""
        import re

        # Extract category using common patterns
        category_patterns = [
            r'["\']?category["\']?\s*[:=]\s*["\']([^"\']+)["\']',
            r'["\']?category["\']?\s*[:=]\s*([^\n,]+)',
            r'["\']?role["\']?\s*[:=]\s*["\']([^"\']+)["\']',
            r'i["\']?ll go with ["\']([^"\']+)["\']',
            r'best fit\s*(?:is|:)\s*["\']?([^"\',\n]+)["\']?',
            r'choosing\s*["\']([^"\']+)["\']',
        ]

        category = None
        for pattern in category_patterns:
            try:
                match = re.search(pattern, text, re.IGNORECASE)
            except re.error:
                continue
            if match:
                category = match.group(1).strip()
                break

        # Extract years of experience
        years_patterns = [
            r'["\']?years_experience["\']?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)',
            r'["\']?experience["\']?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)',
            r"([0-9]+(?:\.[0-9]+)?)\s*years?",
        ]

        years = 0.0
        for pattern in years_patterns:
            try:
                match = re.search(pattern, text, re.IGNORECASE)
            except re.error:
                continue
            if match:
                try:
                    years = float(match.group(1))
                    break
                except:
                    continue

        # Determine seniority
        seniority = "Entry"
        if any(word in text.lower() for word in ["intern", "student", "fresher"]):
            seniority = "Intern"
        elif any(word in text.lower() for word in ["junior", "entry", "0-1", "0 to 1"]):
            seniority = "Entry"
        elif any(word in text.lower() for word in ["mid", "2-5", "3-5"]):
            seniority = "Mid"
        elif any(word in text.lower() for word in ["senior", "lead", "5+", "manager"]):
            seniority = "Senior"

        # Determine role family
        role_family = "Software Engineering"
        if any(word in text.lower() for word in ["data", "analytics", "scientist"]):
            role_family = "Data"
        elif any(word in text.lower() for word in ["product", "design", "ui", "ux"]):
            role_family = "Product"
        elif any(
            word in text.lower() for word in ["devops", "cloud", "infrastructure"]
        ):
            role_family = "DevOps"

        # Extract key signals/skills
        skills = []
        skill_patterns = [
            r'["\']?skills["\']?\s*[:=]\s*\[([^\]]+)\]',
            r'key[\s_-]*signals["\']?\s*[:=]\s*\[([^\]]+)\]',
        ]

        for pattern in skill_patterns:
            try:
                match = re.search(pattern, text, re.IGNORECASE)
            except re.error:
                continue
            if match:
                skills_text = match.group(1)
                skills = [s.strip().strip("\"'") for s in skills_text.split(",")]
                break

        return {
            "category": category or "Software Engineer",
            "role_family": role_family,
            "seniority": seniority,
            "years_experience": years,
            "confidence": 0.7 if category else 0.5,
            "reasoning": "Parsed from text response",
            "key_signals": skills[:5],
            "skills": skills[:10],
        }

    def _build_success_result(
        self, parsed: Dict[str, Any], raw: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build success result from parsed JSON."""
        category = self._normalize_category(parsed.get("category"))
        confidence = self._normalize_confidence(parsed.get("confidence", 0.55))
        years = self._normalize_years(parsed.get("years_experience", 0.0))
        seniority = self._normalize_seniority(parsed.get("seniority"))
        role_family = self._normalize_role_family(parsed.get("role_family"))
        key_signals = self._normalize_list(parsed.get("key_signals"), max_items=10)
        skills = self._normalize_list(parsed.get("skills"), max_items=40)
        if not skills and key_signals:
            skills = key_signals[:]
        if not key_signals and skills:
            key_signals = skills[:5]
        if not skills and category:
            skills = [category]
        if not key_signals and category:
            key_signals = [category]

        return {
            "success": True,
            "category": category,
            "role_family": role_family,
            "seniority": seniority,
            "years_experience": years,
            "confidence": confidence,
            "reasoning": parsed.get("reasoning"),
            "key_signals": key_signals,
            "skills": skills,
            "raw_response": raw,
        }

    def _normalize_category(
        self, value: Any, allowed_categories: Optional[List[str]] = None
    ) -> Optional[str]:
        raw = str(value or "").strip()
        if not raw:
            return None
        clean = re.sub(r"\s+", " ", raw).strip(" .,:;-")
        if not clean:
            return None
        if not allowed_categories:
            return clean

        normalized = {c.lower(): c for c in allowed_categories}
        direct = normalized.get(clean.lower())
        if direct:
            return direct
        compact = re.sub(r"[^a-z0-9]+", "", clean.lower())
        for option in allowed_categories:
            if re.sub(r"[^a-z0-9]+", "", option.lower()) == compact:
                return option
        for option in allowed_categories:
            opt_l = option.lower()
            clean_l = clean.lower()
            if opt_l in clean_l or clean_l in opt_l:
                return option
        return clean

    def _normalize_confidence(self, value: Any) -> float:
        try:
            conf = float(value)
        except Exception:
            conf = 0.55
        if conf > 1.0:
            conf = conf / 100.0
        return max(0.0, min(1.0, conf))

    def _normalize_years(self, value: Any) -> float:
        try:
            years = float(value)
        except Exception:
            years = 0.0
        return max(0.0, min(40.0, round(years, 1)))

    def _normalize_seniority(self, value: Any) -> str:
        raw = str(value or "").strip().lower()
        mapping = {
            "intern": "Intern",
            "entry": "Entry",
            "junior": "Junior",
            "mid": "Mid",
            "senior": "Senior",
            "lead": "Lead",
            "manager": "Manager",
        }
        for key, norm in mapping.items():
            if key in raw:
                return norm
        return "Entry"

    def _normalize_role_family(self, value: Any) -> str:
        raw = str(value or "").strip()
        if not raw:
            return "General"
        return re.sub(r"\s+", " ", raw).title()

    def _normalize_list(self, value: Any, max_items: int = 10) -> List[str]:
        if not isinstance(value, list):
            return []
        items: List[str] = []
        seen = set()
        for item in value:
            text = re.sub(r"\s+", " ", str(item or "").strip())
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            items.append(text)
            if len(items) >= max_items:
                break
        return items

    def _normalize_whitespace(self, text: str) -> str:
        text = str(text or "").replace("\x00", " ")
        return re.sub(r"\s+", " ", text).strip()

    def _job_to_document(self, job: Dict[str, Any]) -> str:
        title = str(job.get("title", "") or "")
        company = str(job.get("company", "") or "")
        location = str(job.get("location", "") or "")
        description = str(job.get("description", "") or "")
        schedule = str(job.get("schedule_type", "") or "")
        return " | ".join(
            part for part in [title, company, location, schedule, description] if part
        )
