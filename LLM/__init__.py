"""LLM helpers for resume classification."""

try:
    from .bedrock_resume_classifier import BedrockResumeCategoryClassifier
except Exception:  # pragma: no cover - optional legacy dependency path
    BedrockResumeCategoryClassifier = None

from .fireworks_resume_intelligence import FireworksResumeIntelligence

__all__ = ["BedrockResumeCategoryClassifier", "FireworksResumeIntelligence"]
