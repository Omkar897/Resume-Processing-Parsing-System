"""Bedrock-based resume category classifier."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv

from src.utils.pdf_extractor import PDFTextExtractor

load_dotenv()


class BedrockResumeCategoryClassifier:
    """Classify resumes into open-ended roles using Amazon Bedrock."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        region_name: Optional[str] = None,
        max_chars: int = 18000,
    ) -> None:
        # Support Bedrock API-key auth via bearer token env.
        # If user stored the Bedrock API key under AWS_ACCESS_KEY_ID (common local setup),
        # mirror it to AWS_BEARER_TOKEN_BEDROCK for boto3 Bedrock runtime calls.
        bearer = (os.getenv("AWS_BEARER_TOKEN_BEDROCK") or "").strip()
        access_key_id = (os.getenv("AWS_ACCESS_KEY_ID") or "").strip()
        if not bearer and (
            access_key_id.startswith("bedrock-api-key-")
            or access_key_id.startswith("ABSK")
        ):
            os.environ["AWS_BEARER_TOKEN_BEDROCK"] = access_key_id

        self.model_id = model_id or os.getenv(
            "BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"
        )
        self.region_name = (
            region_name
            or os.getenv("BEDROCK_REGION")
            or os.getenv("AWS_REGION")
            or os.getenv("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        self.max_chars = max_chars
        self.extractor = PDFTextExtractor()
        self.client = boto3.client("bedrock-runtime", region_name=self.region_name)

    def classify_resume_pdf(
        self,
        pdf_path: str,
        allowed_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Extract text from PDF and classify the resume category."""
        extracted = self.extractor.extract_text(pdf_path)
        if "error" in extracted:
            return {"success": False, "error": extracted["error"]}

        text = extracted.get("text", "")
        result = self.classify_resume_text(text, allowed_categories=allowed_categories)
        result["extraction_method"] = extracted.get("method_used")
        result["extracted_text_length"] = extracted.get("text_length", len(text))
        return result

    def classify_resume_text(
        self,
        resume_text: str,
        allowed_categories: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Classify resume text via Bedrock and return structured output."""
        cleaned_text = self._normalize_whitespace(resume_text)
        trimmed_text = cleaned_text[: self.max_chars]

        if not trimmed_text.strip():
            return {"success": False, "error": "Resume text is empty after extraction"}

        system_prompt = (
            "You are an expert resume-to-job-role mapper. "
            "Infer the most likely target role and seniority from resume evidence only. "
            "Return JSON only."
        )
        user_prompt = self._build_user_prompt(trimmed_text, allowed_categories)

        try:
            response_text = self._invoke_bedrock(system_prompt, user_prompt)
            parsed = self._parse_llm_json(response_text)
            if parsed is None:
                return {
                    "success": False,
                    "error": "Could not parse model response as JSON",
                    "raw_response": response_text[:1000],
                }

            normalized_category = self._normalize_category(
                parsed.get("category"), allowed_categories
            )
            if not normalized_category:
                return {
                    "success": False,
                    "error": "Model did not return a usable category",
                    "raw_category": parsed.get("category"),
                }

            confidence = self._normalize_confidence(parsed.get("confidence", 0.6))
            reasoning = str(parsed.get("reasoning", "")).strip()
            key_signals = parsed.get("key_signals", [])
            if not isinstance(key_signals, list):
                key_signals = []
            years_experience = self._normalize_years(parsed.get("years_experience"))
            seniority = self._normalize_seniority(parsed.get("seniority"))
            role_family = self._normalize_role_family(parsed.get("role_family"))

            return {
                "success": True,
                "category": normalized_category,
                "role_family": role_family,
                "seniority": seniority,
                "years_experience": years_experience,
                "confidence": confidence,
                "reasoning": reasoning,
                "key_signals": [str(s).strip() for s in key_signals[:8] if str(s).strip()],
                "used_chars": len(trimmed_text),
                "total_chars": len(cleaned_text),
                "truncated": len(cleaned_text) > len(trimmed_text),
                "model_id": self.model_id,
                "region": self.region_name,
            }
        except (ClientError, BotoCoreError) as exc:
            return {"success": False, "error": f"Bedrock client error: {exc}"}
        except Exception as exc:
            return {"success": False, "error": f"Bedrock classification error: {exc}"}

    def generate_search_queries(
        self,
        category: str,
        years_experience: float,
        role_family: str = "General",
        seniority: str = "Entry",
        key_signals: Optional[List[str]] = None,
        max_queries: int = 2,
    ) -> Dict[str, Any]:
        """Generate extra job-search query variants from profile signals."""
        key_signals = key_signals or []
        max_queries = max(1, min(5, int(max_queries)))

        base_query = self._build_base_query(category, years_experience)
        signals_text = ", ".join(str(s).strip() for s in key_signals[:8] if str(s).strip())

        system_prompt = (
            "You generate compact Google Jobs query variations. "
            "Keep queries realistic, concise, and ATS/job-board friendly. "
            "Return JSON only."
        )
        user_prompt = (
            "Generate additional query variations for job search.\n"
            "Focus on synonyms, adjacent titles, and skills emphasis.\n"
            "Do not include location in queries.\n\n"
            f"Primary role: {category}\n"
            f"Role family: {role_family}\n"
            f"Seniority: {seniority}\n"
            f"Years of experience: {years_experience}\n"
            f"Key resume signals: {signals_text or 'N/A'}\n"
            f"Base query: {base_query}\n\n"
            "Return strict JSON with schema:\n"
            "{\n"
            '  "queries": ["query 1", "query 2"]\n'
            "}\n\n"
            f"Rules:\n"
            f"- Return up to {max_queries} queries.\n"
            "- Keep each query under 8 words.\n"
            "- Avoid duplicates.\n"
            "- Output JSON only.\n"
        )

        try:
            response_text = self._invoke_bedrock(
                system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=220
            )
            parsed = self._parse_llm_json(response_text)
            raw_queries = parsed.get("queries", []) if isinstance(parsed, dict) else []
            if not isinstance(raw_queries, list):
                raw_queries = []

            queries = []
            seen = set()
            for q in raw_queries:
                q_str = re.sub(r"\s+", " ", str(q).strip())
                if not q_str:
                    continue
                key = q_str.lower()
                if key in seen:
                    continue
                seen.add(key)
                queries.append(q_str)
                if len(queries) >= max_queries:
                    break

            return {"success": True, "queries": queries}
        except (ClientError, BotoCoreError) as exc:
            return {"success": False, "error": f"Bedrock query expansion error: {exc}"}
        except Exception as exc:
            return {"success": False, "error": f"Query expansion failure: {exc}"}

    def rerank_jobs(
        self,
        resume_text: str,
        category: str,
        jobs: List[Dict[str, Any]],
        max_jobs: int = 8,
    ) -> Dict[str, Any]:
        """Rerank a small candidate set with LLM judgment."""
        if not jobs:
            return {"success": True, "ranking": []}

        max_jobs = max(1, min(12, int(max_jobs)))
        candidates = jobs[:max_jobs]
        resume_summary = self._normalize_whitespace(resume_text)[:2600]

        jobs_payload = []
        for idx, job in enumerate(candidates):
            jobs_payload.append(
                {
                    "job_index": idx,
                    "title": str(job.get("title", ""))[:120],
                    "company": str(job.get("company", ""))[:80],
                    "description": str(job.get("description", ""))[:280],
                    "location": str(job.get("location", ""))[:80],
                }
            )

        system_prompt = (
            "You are a hiring relevance reranker. "
            "Score each job by semantic fit to the resume for the target role. "
            "Use evidence, not keyword stuffing. Return JSON only."
        )
        user_prompt = (
            f"Target role: {category}\n"
            "Resume summary:\n"
            f"{resume_summary}\n\n"
            "Jobs:\n"
            f"{json.dumps(jobs_payload, ensure_ascii=False)}\n\n"
            "Return strict JSON schema:\n"
            "{\n"
            '  "ranking": [\n'
            '    {"job_index": 0, "score": 0, "reason": "short reason"}\n'
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "- Include each provided job_index exactly once.\n"
            "- score must be integer 0..100.\n"
            "- Sort by best score first.\n"
            "- Keep reason under 18 words.\n"
            "- Output JSON only.\n"
        )

        try:
            response_text = self._invoke_bedrock(
                system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=500
            )
            parsed = self._parse_llm_json(response_text)
            ranking = parsed.get("ranking", []) if isinstance(parsed, dict) else []
            if not isinstance(ranking, list):
                ranking = []

            normalized = []
            seen = set()
            for item in ranking:
                if not isinstance(item, dict):
                    continue
                try:
                    idx = int(item.get("job_index"))
                    score = int(round(float(item.get("score", 0))))
                except Exception:
                    continue
                if idx < 0 or idx >= len(candidates) or idx in seen:
                    continue
                seen.add(idx)
                score = max(0, min(100, score))
                reason = str(item.get("reason", "")).strip()
                normalized.append(
                    {
                        "job_index": idx,
                        "score": score,
                        "reason": reason[:200],
                    }
                )

            # Ensure every candidate appears exactly once.
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
        except (ClientError, BotoCoreError) as exc:
            return {"success": False, "error": f"Bedrock rerank error: {exc}"}
        except Exception as exc:
            return {"success": False, "error": f"Rerank failure: {exc}"}

    def _build_user_prompt(
        self, resume_text: str, categories: Optional[List[str]] = None
    ) -> str:
        taxonomy_hint = ""
        if categories:
            taxonomy_hint = (
                "Optional taxonomy hint (prefer closest fit but do not force if poor fit):\n"
                + "\n".join(f"- {category}" for category in categories)
                + "\n\n"
            )

        return (
            "Analyze the resume and infer the best target role for job search.\n"
            "Use only evidence from the resume. Do not invent experience.\n\n"
            f"{taxonomy_hint}"
            "Return strict JSON with this schema:\n"
            "{\n"
            '  "category": "specific target role title (e.g., Data Scientist, ML Engineer, Backend Developer)",\n'
            '  "role_family": "broad family (e.g., Data, Software Engineering, Product, Design, DevOps, QA, Security, Analytics)",\n'
            '  "seniority": "Intern | Entry | Junior | Mid | Senior | Lead | Manager",\n'
            '  "years_experience": 0.0,\n'
            '  "confidence": 0.0,\n'
            '  "reasoning": "1-2 sentence explanation grounded in resume evidence",\n'
            '  "key_signals": ["keyword1", "keyword2", "keyword3"]\n'
            "}\n\n"
            "Rules:\n"
            "- years_experience must be numeric (0 to 40).\n"
            "- confidence must be numeric (0 to 1).\n"
            "- Output JSON only.\n\n"
            "Resume text:\n"
            f"{resume_text}"
        )

    def _invoke_bedrock(
        self, system_prompt: str, user_prompt: str, max_tokens: int = 280
    ) -> str:
        body = json.dumps(
            {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_prompt}],
                    }
                ],
            }
        )

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )
        payload = json.loads(response["body"].read())
        content_blocks = payload.get("content", [])
        text_parts = [
            block.get("text", "")
            for block in content_blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(text_parts).strip()

    def _build_base_query(self, category: str, years_experience: float) -> str:
        rounded_years = max(0, int(round(years_experience)))
        if rounded_years == 0:
            return f"entry level {category}"
        if rounded_years == 1:
            return f"{category} 0-1 year experience"
        return f"{category} {rounded_years} years experience"

    def _parse_llm_json(self, response_text: str) -> Optional[Dict[str, Any]]:
        if not response_text:
            return None

        try:
            parsed = json.loads(response_text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        start = response_text.find("{")
        end = response_text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        candidate = response_text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None

    def _normalize_category(
        self, predicted_category: Any, allowed_categories: Optional[List[str]] = None
    ) -> Optional[str]:
        if predicted_category is None:
            return None

        raw = str(predicted_category).strip()
        if not raw:
            return None

        cleaned = re.sub(r"\s+", " ", raw).strip(" .,:;-")
        if not cleaned:
            return None

        # Open-ended mode: accept model role title as-is (preserve acronyms).
        if not allowed_categories:
            return cleaned

        normalized_map = {category.lower(): category for category in allowed_categories}
        direct = normalized_map.get(raw.lower())
        if direct:
            return direct

        compact = re.sub(r"[^a-z0-9]+", "", raw.lower())
        for category in allowed_categories:
            if re.sub(r"[^a-z0-9]+", "", category.lower()) == compact:
                return category

        for category in allowed_categories:
            category_l = category.lower()
            raw_l = raw.lower()
            if category_l in raw_l or raw_l in category_l:
                return category

        return None

    def _normalize_confidence(self, confidence: Any) -> float:
        try:
            value = float(confidence)
        except Exception:
            return 0.6

        if value > 1.0:
            value = value / 100.0

        return max(0.0, min(1.0, value))

    def _normalize_years(self, years: Any) -> float:
        try:
            value = float(years)
        except Exception:
            return 0.0
        return max(0.0, min(40.0, round(value, 1)))

    def _normalize_seniority(self, seniority: Any) -> str:
        raw = str(seniority or "").strip().lower()
        mapping = {
            "intern": "Intern",
            "entry": "Entry",
            "junior": "Junior",
            "mid": "Mid",
            "senior": "Senior",
            "lead": "Lead",
            "manager": "Manager",
        }
        for k, v in mapping.items():
            if k in raw:
                return v
        return "Entry"

    def _normalize_role_family(self, role_family: Any) -> str:
        value = str(role_family or "").strip()
        if not value:
            return "General"
        return re.sub(r"\s+", " ", value).strip().title()

    def _normalize_whitespace(self, text: str) -> str:
        text = str(text or "")
        text = text.replace("\x00", " ")
        return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classify resume with Bedrock")
    parser.add_argument("--file", required=True, help="Path to resume PDF")
    args = parser.parse_args()

    clf = BedrockResumeCategoryClassifier()
    output = clf.classify_resume_pdf(args.file)
    print(json.dumps(output, indent=2, ensure_ascii=False))
