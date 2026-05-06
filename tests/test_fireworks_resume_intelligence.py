import os
import unittest

from LLM.fireworks_resume_intelligence import FireworksResumeIntelligence


class TestFireworksResumeIntelligence(unittest.TestCase):
    def setUp(self):
        os.environ["FIREWORKS_API_KEY"] = "test-key"

    def test_fallback_used_when_primary_is_weak(self):
        helper = FireworksResumeIntelligence()

        responses = [
            {
                "success": True,
                "category": "Data Scientist",
                "role_family": "Data",
                "seniority": "Entry",
                "years_experience": 0.8,
                "confidence": 0.31,
                "reasoning": "Low confidence parse.",
                "key_signals": ["python"],
                "skills": ["python"],
            },
            {
                "success": True,
                "category": "Data Scientist",
                "role_family": "Data",
                "seniority": "Entry",
                "years_experience": 0.8,
                "confidence": 0.82,
                "reasoning": "Good parse.",
                "key_signals": ["python", "pandas", "ml"],
                "skills": ["python", "pandas", "scikit-learn"],
            },
        ]

        def fake_run_classification(**_kwargs):
            return responses.pop(0)

        helper._run_classification = fake_run_classification
        result = helper.classify_resume_text("sample resume text")

        self.assertTrue(result.get("success"))
        self.assertEqual(result.get("category"), "Data Scientist")
        self.assertTrue(result.get("fallback_used"))

    def test_rerank_normalizes_output(self):
        helper = FireworksResumeIntelligence()

        helper.client.rerank = lambda **_kwargs: {
            "data": [
                {"index": 1, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.4},
            ]
        }
        jobs = [
            {"title": "Job A", "description": "desc", "company": "A"},
            {"title": "Job B", "description": "desc", "company": "B"},
        ]
        ranked = helper.rerank_jobs("resume", "Data Scientist", jobs, max_jobs=2)

        self.assertTrue(ranked.get("success"))
        self.assertEqual(ranked["ranking"][0]["job_index"], 1)
        self.assertGreaterEqual(ranked["ranking"][0]["score"], ranked["ranking"][1]["score"])


if __name__ == "__main__":
    unittest.main()
