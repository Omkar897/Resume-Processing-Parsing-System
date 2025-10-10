# -*- coding: utf-8 -*-
import argparse
import json
import sys
import os
from pathlib import Path
import re

# SUPPRESS TENSORFLOW WARNINGS
import logging
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

import sys

sys.path.append("src")

from src.utils.pdf_extractor import PDFTextExtractor
from src.resume.classifier import ResumeClassifier
from src.resume.spacy_parser import SpacyResumeParser


class WorkingProcessor:
    def __init__(self):
        self.extractor = PDFTextExtractor()
        self.classifier = ResumeClassifier()
        self.spacy_parser = SpacyResumeParser()
        self.classifier._load_model()

    def process_resume(self, pdf_path):
        """Process resume with comprehensive spaCy extraction"""

        # Step 1: Extract text
        text_result = self.extractor.extract_text(pdf_path)

        if "error" in text_result:
            return {"error": f'Text extraction failed: {text_result["error"]}'}

        text = text_result["text"]

        # Step 2: Parse all info using spaCy
        parsed_info = self._parse_all_info(text)

        # Step 3: Classify
        classification = self.classifier.classify_resume(text)

        # Step 4: Enhanced classification
        enhanced_text = self._create_enhanced_text(parsed_info, text)
        enhanced_classification = self.classifier.classify_resume(enhanced_text)

        return {
            "filename": os.path.basename(pdf_path),
            "personal_info": parsed_info["personal_info"],
            "education": parsed_info["education"],
            "experience": parsed_info["experience"],
            "skills": parsed_info["skills"],
            "certifications": parsed_info.get("certifications", []),
            "projects": parsed_info.get("projects", []),
            "achievements": parsed_info.get("achievements", []),
            "languages": parsed_info.get("languages", []),
            "summary": parsed_info.get("summary", ""),
            "classification": {
                "category": enhanced_classification["predicted_category"],
                "confidence": float(enhanced_classification["confidence"]),
                "confidence_level": self._get_confidence_level(
                    enhanced_classification["confidence"]
                ),
                "improvement": float(enhanced_classification["confidence"])
                - float(classification["confidence"]),
            },
            "stats": {
                "text_length": text_result["text_length"],
                "method": text_result["method_used"],
                "total_skills": sum(
                    len(skills) for skills in parsed_info["skills"].values()
                ),
                "education_entries": len(parsed_info["education"]),
                "experience_entries": len(parsed_info["experience"]),
                "certification_entries": len(parsed_info.get("certifications", [])),
                "project_entries": len(parsed_info.get("projects", [])),
                "achievement_entries": len(parsed_info.get("achievements", [])),
                "language_entries": len(parsed_info.get("languages", [])),
            },
            "processing_status": "success",
        }

    def _parse_all_info(self, text):
        """Parse all resume information using comprehensive spaCy extraction"""
        return self.spacy_parser.extract_all_resume_data(text)

    def _create_enhanced_text(self, parsed_info, original_text):
        """Create enhanced text for better classification"""
        enhanced_parts = []

        # Boost skills for classification
        for category, skills in parsed_info["skills"].items():
            if "ml_ai" in category or "web" in category or "programming" in category:
                enhanced_parts.extend(skills * 3)
            else:
                enhanced_parts.extend(skills * 2)

        # Boost education information
        for edu in parsed_info["education"]:
            if edu["degree"]:
                enhanced_parts.append(edu["degree"] * 2)

        # Boost experience roles
        for exp in parsed_info["experience"]:
            if exp["role"]:
                enhanced_parts.append(exp["role"] * 2)

        # Boost certifications
        for cert in parsed_info.get("certifications", []):
            enhanced_parts.append(cert["name"] * 2)

        # Boost project technologies
        for proj in parsed_info.get("projects", []):
            enhanced_parts.extend(proj.get("technologies", []) * 2)

        return " ".join(enhanced_parts) + " " + original_text

    def _get_confidence_level(self, confidence):
        """Convert confidence to level"""
        if confidence > 0.7:
            return "HIGH"
        elif confidence > 0.5:
            return "MEDIUM"
        else:
            return "LOW"


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Resume Processing System with spaCy"
    )
    parser.add_argument("--file", help="PDF file to process")
    parser.add_argument("--test", action="store_true", help="Use default test resume")

    args = parser.parse_args()

    # Smart file selection
    if args.test:
        pdf_path = r"data\resumes\Test Resumes\RESUME AI ATS.pdf"
    elif args.file:
        pdf_path = args.file
    else:
        pdf_path = r"data\resumes\Test Resumes\RESUME AI ATS.pdf"

    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        return

    try:
        processor = WorkingProcessor()
        result = processor.process_resume(pdf_path)

        if "error" in result:
            print(f"Error: {result['error']}")
            return

        print(f"\n" + "=" * 80)
        print(f"COMPREHENSIVE RESUME PROCESSING RESULTS")
        print(f"=" * 80)

        # Personal Information
        print(f"\n📋 PERSONAL INFORMATION:")
        for key, value in result["personal_info"].items():
            if value:
                print(f"   {key.title()}: {value}")

        # Summary
        if result.get("summary"):
            print(f"\n📝 PROFESSIONAL SUMMARY:")
            print(f"   {result['summary']}")

        # Education
        print(f"\n🎓 EDUCATION ({result['stats']['education_entries']} entries):")
        if result["education"]:
            for i, edu in enumerate(result["education"], 1):
                print(f"   {i}. {edu['degree']}")
                if edu["institution"]:
                    print(f"      🏫 Institution: {edu['institution']}")
                if edu["year"]:
                    print(f"      📅 Year: {edu['year']}")
                if edu.get("cgpa"):
                    print(f"      📊 CGPA: {edu['cgpa']}")
                if edu.get("location"):
                    print(f"      📍 Location: {edu['location']}")
        else:
            print("   No education entries found")

        # Experience
        print(
            f"\n💼 WORK EXPERIENCE ({result['stats']['experience_entries']} entries):"
        )
        if result["experience"]:
            for i, exp in enumerate(result["experience"], 1):
                print(f"   {i}. {exp['role']}")
                if exp.get("company"):
                    print(f"      🏢 Company: {exp['company']}")
                print(f"      📅 Duration: {exp['duration']}")
                if exp.get("location"):
                    print(f"      📍 Location: {exp['location']}")
        else:
            print("   No experience entries found")

        # Skills
        print(f"\n🛠️  TECHNICAL SKILLS ({result['stats']['total_skills']} total):")
        for category, skills in result["skills"].items():
            print(f"   {category.replace('_', ' ').title()}: {', '.join(skills)}")

        # Certifications
        if result.get("certifications"):
            print(
                f"\n🏆 CERTIFICATIONS ({result['stats']['certification_entries']} entries):"
            )
            for i, cert in enumerate(result["certifications"], 1):
                print(f"   {i}. {cert['name']}")
                if cert.get("issuer"):
                    print(f"      🏢 Issuer: {cert['issuer']}")
                if cert.get("date"):
                    print(f"      📅 Date: {cert['date']}")

        # Projects
        if result.get("projects"):
            print(f"\n🚀 PROJECTS ({result['stats']['project_entries']} entries):")
            for i, proj in enumerate(result["projects"], 1):
                print(f"   {i}. {proj['title']}")
                if proj.get("technologies"):
                    print(f"      💻 Technologies: {', '.join(proj['technologies'])}")
                if proj.get("duration"):
                    print(f"      📅 Duration: {proj['duration']}")
                if proj.get("description"):
                    desc = (
                        proj["description"][:100] + "..."
                        if len(proj["description"]) > 100
                        else proj["description"]
                    )
                    print(f"      📝 Description: {desc}")

        # Achievements
        if result.get("achievements"):
            print(
                f"\n🏅 ACHIEVEMENTS ({result['stats']['achievement_entries']} entries):"
            )
            for i, achievement in enumerate(result["achievements"], 1):
                print(f"   {i}. {achievement['title']}")
                if achievement.get("organization"):
                    print(f"      🏢 Organization: {achievement['organization']}")
                if achievement.get("date"):
                    print(f"      📅 Date: {achievement['date']}")

        # Languages
        if result.get("languages"):
            print(f"\n🌐 LANGUAGES ({result['stats']['language_entries']} entries):")
            for i, lang in enumerate(result["languages"], 1):
                proficiency = (
                    f" ({lang['proficiency']})" if lang.get("proficiency") else ""
                )
                print(f"   {i}. {lang['language']}{proficiency}")

        # Classification Results
        print(f"\n🎯 AI CLASSIFICATION RESULTS:")
        clf = result["classification"]
        print(f"   Category: {clf['category']}")
        print(f"   Confidence: {clf['confidence']:.3f} [{clf['confidence_level']}]")
        if clf["improvement"] > 0:
            print(f"   Enhancement Boost: +{clf['improvement']:.3f}")

        # Processing Stats
        # Processing Stats
        print(f"\n📊 PROCESSING STATISTICS:")
        stats = result["stats"]
        print(f"   Text Length: {stats['text_length']} characters")
        print(f"   Extraction Method: {stats['method']}")

        # Calculate total sections extracted
        total_sections = (
            stats["education_entries"]
            + stats["experience_entries"]
            + stats["certification_entries"]
            + stats["project_entries"]
            + stats["achievement_entries"]
            + stats["language_entries"]
        )

        print(f"   Total Sections Extracted: {total_sections}")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
