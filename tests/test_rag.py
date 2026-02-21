"""
Quick test script for RAG components
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rag.rag_engine import RAGEngine
from src.rag.resume_analyzer import ResumeAnalyzer


def test_rag_engine():
    """Test RAG engine initialization and embedding generation"""
    print("[TEST] Testing RAG Engine...")

    try:
        engine = RAGEngine()
        print("[SUCCESS] RAG Engine initialized")

        # Test embedding generation
        test_text = "Python developer with 3 years of experience in machine learning"
        embedding = engine.generate_embedding(test_text)
        print(f"[SUCCESS] Generated embedding (dimension: {len(embedding)})")

        # Test similarity calculation
        resume_text = "Python developer with ML experience"
        job_text = "Looking for Python engineer with machine learning skills"
        similarity = engine.calculate_semantic_similarity(resume_text, job_text)
        print(f"[SUCCESS] Similarity score: {similarity:.2f}")

        return True
    except Exception as e:
        print(f"[ERROR] RAG Engine test failed: {e}")
        return False


def test_resume_analyzer():
    """Test resume analyzer with Claude"""
    print("\n[TEST] Testing Resume Analyzer...")

    try:
        analyzer = ResumeAnalyzer()
        print("[SUCCESS] Resume Analyzer initialized")

        # Test with sample data
        sample_data = {
            "skills": ["Python", "Machine Learning", "TensorFlow"],
            "experience": [
                {"role": "Data Scientist", "company": "TechCorp", "duration": "2 years"}
            ],
            "projects": [
                {"name": "ML Model", "type": "Machine Learning", "date": "2024"}
            ],
        }

        print("[AI] Calling Claude API for analysis...")
        analysis = analyzer.analyze_resume(sample_data, "Data Scientist")

        print("[SUCCESS] Resume analysis completed:")
        print(f"   - Strengths: {len(analysis.get('strengths', []))}")
        print(f"   - Missing keywords: {len(analysis.get('missing_keywords', []))}")
        print(f"   - Suggestions: {len(analysis.get('suggestions', []))}")
        print(f"   - ATS Score: {analysis.get('ats_score', 0)}")

        return True
    except Exception as e:
        print(f"[ERROR] Resume Analyzer test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("RAG COMPONENTS TEST SUITE")
    print("=" * 60)

    # Test RAG Engine
    rag_success = test_rag_engine()

    # Test Resume Analyzer
    analyzer_success = test_resume_analyzer()

    print("\n" + "=" * 60)
    print("TEST RESULTS:")
    print(f"  RAG Engine: {'✅ PASS' if rag_success else '❌ FAIL'}")
    print(f"  Resume Analyzer: {'✅ PASS' if analyzer_success else '❌ FAIL'}")
    print("=" * 60)

    if rag_success and analyzer_success:
        print("\n🎉 All tests passed! RAG system is ready.")
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed. Check the errors above.")
        sys.exit(1)
