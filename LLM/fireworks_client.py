"""Lightweight Fireworks REST client wrappers for chat, embeddings, and rerank."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import requests
from dotenv import load_dotenv

load_dotenv()


class FireworksClient:
    """Minimal Fireworks API client using direct HTTP requests."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout_seconds: int = 40,
    ) -> None:
        self.api_key = (api_key or os.getenv("FIREWORKS_API_KEY") or "").strip()
        if not self.api_key:
            raise ValueError("FIREWORKS_API_KEY not found in environment")

        self.base_url = (
            base_url
            or os.getenv("FIREWORKS_BASE_URL")
            or "https://api.fireworks.ai/inference/v1"
        ).rstrip("/")
        self.timeout_seconds = int(timeout_seconds)

        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between requests (2 requests/second)

    def chat_json(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 700,
        temperature: float = 0.0,
        extra_payload: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        """
        Call chat completions in JSON mode and parse content as JSON.

        Returns:
            (parsed_json, raw_response_json)
        """
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "context_length_exceeded_behavior": "truncate",
            "response_format": {"type": "json_object"},
        }
        if extra_payload:
            payload.update(extra_payload)

        raw = self._post_json("/chat/completions", payload)
        content = (
            raw.get("choices", [{}])[0].get("message", {}).get("content", "")
            if isinstance(raw, dict)
            else ""
        )
        parsed = self._parse_json_text(content)
        return parsed, raw

    def create_embeddings(
        self,
        *,
        model: str,
        inputs: List[str],
        dimensions: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"model": model, "input": inputs}
        if dimensions is not None:
            payload["dimensions"] = int(dimensions)
        return self._post_json("/embeddings", payload)

    def rerank(
        self,
        *,
        model: str,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None,
        return_documents: bool = False,
        task: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "query": query,
            "documents": documents,
            "return_documents": bool(return_documents),
        }
        if top_n is not None:
            payload["top_n"] = int(top_n)
        if task:
            payload["task"] = task
        return self._post_json("/rerank", payload)

    def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Low-level POST to Fireworks API with rate limiting and error handling."""
        # Rate limiting: wait between requests
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            f"{self.base_url}{path}",
            headers=headers,
            json=payload,
            timeout=self.timeout_seconds,
        )

        # Update last request time
        self.last_request_time = time.time()

        if response.status_code >= 400:
            body = response.text[:1200]
            raise RuntimeError(
                f"Fireworks API error {response.status_code} at {path}: {body}"
            )
        try:
            return response.json()
        except Exception as exc:
            snippet = response.text[:1200]
            raise RuntimeError(
                f"Fireworks API returned non-JSON at {path}: {snippet}"
            ) from exc

    def _parse_json_text(self, text: str) -> Optional[Dict[str, Any]]:
        if not text:
            return None
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            pass

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except Exception:
            return None
