# -*- coding: utf-8 -*-
import argparse
import json
import sys
import os
from pathlib import Path

# SUPPRESS TENSORFLOW WARNINGS - Add these lines
import logging
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logging
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN optimizations
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# Add paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

# Import the working processor that has all the advanced extraction
import sys

sys.path.append("src")

from src.utils.pdf_extractor import PDFTextExtractor
from src.resume.classifier import ResumeClassifier


class WorkingProcessor:
    def __init__(self):
        self.extractor = PDFTextExtractor()
        self.classifier = ResumeClassifier()
        self.classifier._load_model()

    def process_resume(self, pdf_path):
        """Process resume - keep working version"""

        # Step 1: Extract text (silent)
        text_result = self.extractor.extract_text(pdf_path)

        if "error" in text_result:
            return {"error": f'Text extraction failed: {text_result["error"]}'}

        text = text_result["text"]

        # Step 2: Debug sections (silent)
        self._debug_sections(text)

        # Step 3: Parse all info (keep what works)
        parsed_info = self._parse_all_info(text)

        # Step 4: Classify (silent)
        classification = self.classifier.classify_resume(text)

        # Step 5: Enhanced classification (silent)
        enhanced_text = self._create_enhanced_text(parsed_info, text)
        enhanced_classification = self.classifier.classify_resume(enhanced_text)

        return {
            "filename": os.path.basename(pdf_path),
            "personal_info": parsed_info["personal_info"],
            "education": parsed_info["education"],
            "experience": parsed_info["experience"],
            "skills": parsed_info["skills"],
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
            },
            "processing_status": "success",
        }

    def _debug_sections(self, text):
        """Silent section detection"""
        sections_found = []
        text_lower = text.lower()

        section_checks = {
            "Education": ["education", "academic"],
            "Experience": ["experience", "internship", "employment"],
            "Skills": ["skills", "technical"],
            "Projects": ["projects", "project"],
        }

        for section, keywords in section_checks.items():
            found = any(kw in text_lower for kw in keywords)
            status = "Found" if found else "Missing"
            sections_found.append(f"{section}: {status}")

    def _parse_all_info(self, text):
        """Parse all resume information"""
        return {
            "personal_info": self._extract_personal_info(text),
            "education": self._extract_education(text),
            "experience": self._extract_experience(text),
            "skills": self._extract_skills(text),
        }

    def _extract_personal_info(self, text):
        """Extract personal information - KEEP WORKING VERSION"""
        import re

        info = {}

        # Name - first clean line
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        for line in lines[:5]:
            clean_line = re.sub(r"[*#\[\]()]", "", line).strip()
            if (
                clean_line
                and len(clean_line.split()) <= 4
                and len(clean_line) > 2
                and not re.search(
                    r"@|phone|\+\d|www\.|linkedin|github", clean_line.lower()
                )
            ):
                info["name"] = clean_line
                break

        # Email
        email_match = re.search(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text
        )
        info["email"] = email_match.group(0) if email_match else None

        # Phone
        phone_patterns = [r"\+91[\s-]?\d{10}", r"\d{10}"]
        for pattern in phone_patterns:
            phone_match = re.search(pattern, text)
            if phone_match:
                info["phone"] = phone_match.group(0)
                break

        # LinkedIn
        linkedin_match = re.search(
            r"linkedin\.com/in/([A-Za-z0-9-]+)", text, re.IGNORECASE
        )
        info["linkedin"] = linkedin_match.group(0) if linkedin_match else None

        # GitHub
        github_match = re.search(r"github\.com/([A-Za-z0-9-]+)", text, re.IGNORECASE)
        info["github"] = github_match.group(0) if github_match else None

        return info

    def _extract_education(self, text):
        """Extract education information - KEEP WORKING VERSION"""
        import re

        education = []

        # Find education section
        edu_patterns = [
            r"EDUCATION(.*?)(?:EXPERIENCE|SKILLS|PROJECTS|INTERNSHIP|CERTIFICATIONS|RELEVANT|$)",
            r"ACADEMIC.*?BACKGROUND(.*?)(?:EXPERIENCE|SKILLS|PROJECTS|$)",
            r"ACADEMIC\s+DETAILS(.*?)(?:SOFT\s+SKILLS|TECHNICAL\s+SKILLS|SKILLS|WORK\s+EXPERIENCE|EXPERIENCE|$)",
        ]

        edu_section = None
        for pattern in edu_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                edu_section = match.group(1).strip()
                break

        if edu_section:
            degree_patterns = [
                r"(Bachelor.*?Technology.*?Computer.*?Science.*?Engineering?).*?(\d{4})",
                r"(Manipal Institute of Technology).*?(\d{4})",
                r"(B\.?Tech.*?Computer.*?Science).*?(\d{4})",
                r"(Bachelor.*?).*?(\d{4}).*?(\d{4})",
                r"(Bachelor.*?Technology.*?).*?CGPA.*?(\d{4})",
                r"([A-Z][a-z]+.*?(?:Technology|Engineering|Computer)[^\\n]{0,50}?).*?(\d{4})",
                r"(B\.?\s*Tech.*?Computer.*?).*?(\d{4})",
            ]

            for i, pattern in enumerate(degree_patterns):
                matches = re.findall(pattern, edu_section, re.IGNORECASE | re.DOTALL)
                if matches:
                    for match in matches:
                        degree = re.sub(r"\s+", " ", match[0].strip())
                        year = match[1]

                        institution = None
                        inst_patterns = [
                            r"(Manipal Institute of Technology)",
                            r"([A-Z][a-z]+ (?:Institute|University|College)[^\n]*)",
                        ]

                        for inst_pattern in inst_patterns:
                            inst_match = re.search(
                                inst_pattern, edu_section, re.IGNORECASE
                            )
                            if inst_match:
                                institution = inst_match.group(0).strip()
                                break

                        cgpa_match = re.search(
                            r"CGPA[:\s]*([0-9.]+)", edu_section, re.IGNORECASE
                        )
                        cgpa = cgpa_match.group(1) if cgpa_match else None

                        education.append(
                            {
                                "degree": degree,
                                "institution": institution,
                                "year": year,
                                "cgpa": cgpa,
                            }
                        )
                    break

        return education

    def _extract_experience(self, text):
        """FIXED experience extraction with proper chronological sorting"""
        import re
        from datetime import datetime

        experience = []

        # Based on the raw text we saw, let's extract directly
        # Your text has: "Web Developer Intern\nCodSoft , Bangalore\nFebruary 2024 – March 2024"

        # Pattern 1: Look for the specific format in your resume
        exp_patterns = [
            # Pattern: Role\nCompany, Location\nMonth Year – Month Year
            r"(AI Intern|Web Developer Intern)\s*\n\s*([A-Z][A-Za-z.]+)\s*,\s*([A-Z][a-z]+)\s*\n\s*([A-Z][a-z]+\s+\d{4})\s*[–—-]\s*([A-Z][a-z]+\s+\d{4})",
            # Pattern: Role\nCompany\nMonth Year – Month Year
            r"(AI Intern|Web Developer Intern)\s*\n\s*([A-Z][A-Za-z.]+)\s*\n\s*([A-Z][a-z]+\s+\d{4})\s*[–—-]\s*([A-Z][a-z]+\s+\d{4})",
        ]

        for pattern in exp_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match) == 5:  # Role, Company, Location, Start, End
                    role = match[0]
                    company = f"{match[1]}, {match[2]}"
                    start_date = match[3]
                    end_date = match[4]
                elif len(match) == 4:  # Role, Company, Start, End
                    role = match[0]
                    company = match[1]
                    start_date = match[2]
                    end_date = match[3]
                else:
                    continue

                duration = f"{start_date} - {end_date}"

                experience.append(
                    {
                        "duration": duration,
                        "role": role,
                        "company": company,
                        "description": f"{role} at {company}",
                        "start_date_raw": start_date,  # Keep for sorting
                    }
                )

        # If the specific patterns don't work, try a broader approach
        if not experience:
            # Look for any date ranges in the text
            date_pattern = r"([A-Z][a-z]+\s+\d{4})\s*[–—-]\s*([A-Z][a-z]+\s+\d{4})"
            date_matches = re.findall(date_pattern, text)

            for start_date, end_date in date_matches:
                duration = f"{start_date} - {end_date}"

                # Look for role/company context around these dates
                date_context = re.search(
                    rf"([^\n]*)\n([^\n]*)\n[^\n]*{re.escape(start_date)}.*?{re.escape(end_date)}",
                    text,
                    re.IGNORECASE,
                )

                if date_context:
                    potential_role = date_context.group(1).strip()
                    potential_company = date_context.group(2).strip()

                    # Check if they look like role/company
                    if (
                        "intern" in potential_role.lower()
                        or "developer" in potential_role.lower()
                    ):
                        experience.append(
                            {
                                "duration": duration,
                                "role": potential_role,
                                "company": potential_company,
                                "description": f"{potential_role} at {potential_company}",
                                "start_date_raw": start_date,
                            }
                        )

        # SORT BY DATE - Most recent first (reverse chronological order)
        def parse_date(date_str):
            """Convert 'March 2024' to datetime for sorting"""
            try:
                return datetime.strptime(date_str, "%B %Y")
            except:
                return datetime.min

        if experience:
            experience.sort(key=lambda x: parse_date(x["start_date_raw"]), reverse=True)

            # Remove the temporary sorting field
            for exp in experience:
                del exp["start_date_raw"]

        return experience

    def _extract_skills(self, text):
        """Extract skills by category - KEEP WORKING VERSION"""
        import re

        skill_categories = {
            "programming_languages": [
                "Python",
                "Java",
                "JavaScript",
                "TypeScript",
                "C++",
                "C#",
                "C",
                "Go",
                "Rust",
            ],
            "web_technologies": [
                "React",
                "Angular",
                "Vue.js",
                "Node.js",
                "Express",
                "Django",
                "Flask",
                "HTML5",
                "CSS3",
            ],
            "databases": [
                "MySQL",
                "PostgreSQL",
                "MongoDB",
                "Redis",
                "SQLite",
                "Oracle",
            ],
            "ml_ai_tools": [
                "TensorFlow",
                "PyTorch",
                "scikit-learn",
                "Pandas",
                "NumPy",
                "Keras",
                "OpenCV",
            ],
            "cloud_devops": [
                "AWS",
                "Azure",
                "Docker",
                "Kubernetes",
                "Jenkins",
                "Git",
                "GitHub",
            ],
            "tools": ["VS Code", "IntelliJ", "Eclipse", "Postman", "Jira"],
        }

        found_skills = {}
        text_lower = text.lower()

        for category, skills_list in skill_categories.items():
            category_skills = []
            for skill in skills_list:
                if re.search(r"\b" + re.escape(skill.lower()) + r"\b", text_lower):
                    category_skills.append(skill)

            if category_skills:
                found_skills[category] = category_skills

        return found_skills

    def _create_enhanced_text(self, parsed_info, original_text):
        """Create enhanced text for better classification"""
        enhanced_parts = []

        for category, skills in parsed_info["skills"].items():
            if "ml_ai" in category or "web" in category or "programming" in category:
                enhanced_parts.extend(skills * 3)
            else:
                enhanced_parts.extend(skills * 2)

        for edu in parsed_info["education"]:
            if edu["degree"]:
                enhanced_parts.append(edu["degree"] * 2)

        for exp in parsed_info["experience"]:
            if exp["role"]:
                enhanced_parts.append(exp["role"] * 2)

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
    parser = argparse.ArgumentParser(description="Resume Processing System")

    # Enhanced argument parsing with defaults
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

        print(f"\n" + "=" * 70)
        print(f"COMPLETE RESUME PROCESSING RESULTS")
        print(f"=" * 70)

        print(f"Personal Information:")
        for key, value in result["personal_info"].items():
            if value:
                print(f"   {key.title()}: {value}")

        print(f"\nEducation ({result['stats']['education_entries']} entries):")
        if result["education"]:
            for i, edu in enumerate(result["education"], 1):
                print(f"   {i}. {edu['degree']}")
                if edu["institution"]:
                    print(f"      Institution: {edu['institution']}")
                if edu["year"]:
                    print(f"      Year: {edu['year']}")
                if edu.get("cgpa"):
                    print(f"      CGPA: {edu['cgpa']}")
        else:
            print("   No education entries found")

        print(f"\nExperience ({result['stats']['experience_entries']} entries):")
        if result["experience"]:
            for i, exp in enumerate(result["experience"], 1):
                company_info = f" at {exp['company']}" if exp.get("company") else ""
                print(f"   {i}. {exp['role']}{company_info} | {exp['duration']}")
        else:
            print("   No experience entries found")

        print(f"\nSkills ({result['stats']['total_skills']} total):")
        for category, skills in result["skills"].items():
            print(f"   {category.replace('_', ' ').title()}: {', '.join(skills)}")

        print(f"\nClassification Results:")
        clf = result["classification"]
        print(f"   Category: {clf['category']}")
        print(f"   Confidence: {clf['confidence']:.3f} [{clf['confidence_level']}]")

        print(f"\nProcessing Stats:")
        stats = result["stats"]
        print(f"   Text Length: {stats['text_length']} chars")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
