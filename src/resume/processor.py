# working_resume_processor.py (ENHANCED VERSION)
import re
import os
from ..utils.pdf_extractor import PDFTextExtractor  
from .classifier import ResumeClassifier

class WorkingResumeProcessor:
    def __init__(self):
        self.extractor = PDFTextExtractor()
        self.classifier = ResumeClassifier()
        self.classifier._load_model()
    
    def process_resume(self, pdf_path):
        """Enhanced resume processing with education & experience"""
        
        print(f"🚀 Processing: {os.path.basename(pdf_path)}")
        
        # Step 1: Extract text
        print("📄 Extracting text...")
        text_result = self.extractor.extract_text(pdf_path)
        
        if 'error' in text_result:
            return {'error': f'Text extraction failed: {text_result["error"]}'}
        
        text = text_result['text']
        print(f"✅ Text extracted: {text_result['text_length']} characters")
        
        # Debug: Show what sections we find
        self._debug_sections(text)
        
        # Step 2: Parse all information
        print("🔍 Parsing information...")
        parsed_info = self._parse_all_info(text)
        
        # Step 3: Classify with enhanced text
        print("🎯 Classifying resume...")
        original_classification = self.classifier.classify_resume(text)
        
        # Create enhanced text for better classification
        enhanced_text = self._create_enhanced_text(parsed_info, text)
        enhanced_classification = self.classifier.classify_resume(enhanced_text)
        
        # Step 4: Final result
        result = {
            'filename': os.path.basename(pdf_path),
            'personal_info': parsed_info['personal_info'],
            'education': parsed_info['education'],
            'experience': parsed_info['experience'],
            'skills': parsed_info['skills'],
            'classification': {
                'category': enhanced_classification['predicted_category'],
                'confidence': float(enhanced_classification['confidence']),
                'confidence_level': self._get_confidence_level(enhanced_classification['confidence']),
                'improvement': float(enhanced_classification['confidence']) - float(original_classification['confidence'])
            },
            'stats': {
                'text_length': text_result['text_length'],
                'method': text_result['method_used'],
                'total_skills': sum(len(skills) for skills in parsed_info['skills'].values()),
                'education_entries': len(parsed_info['education']),
                'experience_entries': len(parsed_info['experience'])
            },
            'processing_status': 'success'
        }
        
        return result
    
    def _debug_sections(self, text):
        """Show what sections we detect"""
        print("🔍 Section detection:")
        
        sections_found = []
        text_lower = text.lower()
        
        section_checks = {
            'Education': ['education', 'academic'],
            'Experience': ['experience', 'internship', 'employment'],
            'Skills': ['skills', 'technical'],
            'Projects': ['projects', 'project']
        }
        
        for section, keywords in section_checks.items():
            found = any(kw in text_lower for kw in keywords)
            status = "✅" if found else "❌"
            sections_found.append(f"{status} {section}")
        
        print(f"   {' | '.join(sections_found)}")
    
    def _parse_all_info(self, text):
        """Parse all resume information"""
        return {
            'personal_info': self._extract_personal_info(text),
            'education': self._extract_education(text),
            'experience': self._extract_experience(text),
            'skills': self._extract_skills(text)
        }
    
    def _extract_personal_info(self, text):
        """Extract personal information"""
        info = {}
        
        # Name - first clean line
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines[:5]:
            clean_line = re.sub(r'[*#\[\]()]', '', line).strip()
            if (clean_line and len(clean_line.split()) <= 4 and len(clean_line) > 2 
                and not re.search(r'@|phone|\+\d|www\.|linkedin|github', clean_line.lower())):
                info['name'] = clean_line
                break
        
        # Email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        info['email'] = email_match.group(0) if email_match else None
        
        # Phone
        phone_patterns = [r'\+91[\s-]?\d{10}', r'\d{10}']
        for pattern in phone_patterns:
            phone_match = re.search(pattern, text)
            if phone_match:
                info['phone'] = phone_match.group(0)
                break
        
        # LinkedIn
        linkedin_match = re.search(r'linkedin\.com/in/([A-Za-z0-9-]+)', text, re.IGNORECASE)
        info['linkedin'] = linkedin_match.group(0) if linkedin_match else None
        
        # GitHub
        github_match = re.search(r'github\.com/([A-Za-z0-9-]+)', text, re.IGNORECASE)
        info['github'] = github_match.group(0) if github_match else None
        
        return info
    
    # Replace the _extract_education method in working_resume_processor.py with this:

    def _extract_education(self, text):
        """Enhanced education extraction"""
        education = []
        
        # Find education section with more flexible patterns
        edu_patterns = [
            r'EDUCATION(.*?)(?:EXPERIENCE|SKILLS|PROJECTS|INTERNSHIP|CERTIFICATIONS|RELEVANT|$)',
            r'ACADEMIC.*?BACKGROUND(.*?)(?:EXPERIENCE|SKILLS|PROJECTS|$)',
            r'QUALIFICATIONS?(.*?)(?:EXPERIENCE|SKILLS|PROJECTS|$)'
        ]
        
        edu_section = None
        for pattern in edu_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                edu_section = match.group(1).strip()
                print(f"   📚 Education section found: {len(edu_section)} characters")
                break
        
        if edu_section:
            print(f"   📝 Education content preview: {edu_section[:100]}...")
            
            # More flexible degree patterns
            degree_patterns = [
                r'(Bachelor.*?Technology.*?Computer Science.*?Engineering?).*?(\d{4})',
                r'(B\.?Tech.*?Computer.*?Science.*?).*?(\d{4})',  
                r'(Bachelor.*?Computer.*?).*?(\d{4})',
                r'(B\.?E\.?.*?Computer.*?).*?(\d{4})',
                # Look for any Bachelor/BTech followed by year
                r'(Bachelor[^\n]*?).*?(\d{4})',
                r'(B\.?Tech[^\n]*?).*?(\d{4})',
                # Even more flexible - any degree pattern
                r'([A-Z][a-z]+ [A-Z][a-z]+ [A-Z][a-z]+.*?)(\d{4})'
            ]
            
            for i, pattern in enumerate(degree_patterns):
                matches = re.findall(pattern, edu_section, re.IGNORECASE | re.DOTALL)
                if matches:
                    print(f"   ✅ Pattern {i+1} matched: {len(matches)} entries")
                    for match in matches:
                        degree = re.sub(r'\s+', ' ', match[0].strip())
                        year = match[1]
                        
                        # Look for institution in the same section
                        institution_patterns = [
                            r'(Manipal Institute of Technology)',
                            r'([A-Z][a-z]+ (?:Institute|University|College)[^\n]*)',
                            r'(IIT|NIT|BITS)\s+([A-Za-z]+)'
                        ]
                        
                        institution = None
                        for inst_pattern in institution_patterns:
                            inst_match = re.search(inst_pattern, edu_section, re.IGNORECASE)
                            if inst_match:
                                institution = inst_match.group(0).strip()
                                break
                        
                        # Look for CGPA/GPA
                        cgpa_match = re.search(r'CGPA[:\s]*([0-9.]+)', edu_section, re.IGNORECASE)
                        cgpa = cgpa_match.group(1) if cgpa_match else None
                        
                        education.append({
                            'degree': degree,
                            'institution': institution,
                            'year': year,
                            'cgpa': cgpa
                        })
                    break  # Stop after first successful pattern
            
            if not education:
                print("   ❌ No degree patterns matched")
                # Show first few lines of education section for debugging
                lines = [line.strip() for line in edu_section.split('\n')[:5] if line.strip()]
                for i, line in enumerate(lines):
                    print(f"      Line {i+1}: {line}")
        
        return education

    
    def _extract_experience(self, text):
        """Extract experience/internship information"""
        experience = []
        
        # Find experience sections
        exp_patterns = [
            r'(?:WORK\s+)?EXPERIENCE(.*?)(?:EDUCATION|SKILLS|PROJECTS|CERTIFICATIONS|$)',
            r'INTERNSHIPS?(.*?)(?:EDUCATION|SKILLS|PROJECTS|$)',
            r'EMPLOYMENT(.*?)(?:EDUCATION|SKILLS|PROJECTS|$)'
        ]
        
        exp_section = None
        for pattern in exp_patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                exp_section = match.group(1).strip()
                break
        
        if exp_section:
            # Look for date ranges
            date_patterns = [
                r'(\w+\s+\d{4})\s*[-–—]\s*(\w+\s+\d{4}|present|current)',
                r'(\d{4})\s*[-–—]\s*(\d{4}|present|current)'
            ]
            
            for pattern in date_patterns:
                matches = list(re.finditer(pattern, exp_section, re.IGNORECASE))
                for match in matches:
                    duration = f"{match.group(1)} - {match.group(2)}"
                    
                    # Get surrounding context
                    start = max(0, match.start() - 100)
                    end = min(len(exp_section), match.end() + 200)
                    context = exp_section[start:end].strip()
                    
                    # Extract role/company
                    role_matches = re.findall(r'(Software.*?Engineer|Developer|Intern|Analyst)', context, re.IGNORECASE)
                    role = role_matches[0] if role_matches else "Position"
                    
                    experience.append({
                        'duration': duration,
                        'role': role,
                        'description': context[:150] + "..." if len(context) > 150 else context
                    })
        
        return experience
    
    def _extract_skills(self, text):
        """Extract skills by category"""
        skill_categories = {
            'programming_languages': [
                'Python', 'Java', 'JavaScript', 'TypeScript', 'C++', 'C#', 'C', 'Go', 'Rust', 'PHP'
            ],
            'web_technologies': [
                'React', 'Angular', 'Vue.js', 'Node.js', 'Express', 'Django', 'Flask', 'HTML5', 'CSS3'
            ],
            'databases': [
                'MySQL', 'PostgreSQL', 'MongoDB', 'Redis', 'SQLite', 'Oracle'
            ],
            'ml_ai_tools': [
                'TensorFlow', 'PyTorch', 'scikit-learn', 'Pandas', 'NumPy', 'Keras', 'OpenCV', 'Matplotlib'
            ],
            'cloud_devops': [
                'AWS', 'Azure', 'Google Cloud', 'Docker', 'Kubernetes', 'Jenkins', 'Git', 'GitHub'
            ],
            'tools': [
                'VS Code', 'IntelliJ', 'Eclipse', 'Postman', 'Jira', 'Figma'
            ]
        }
        
        found_skills = {}
        text_lower = text.lower()
        
        for category, skills_list in skill_categories.items():
            category_skills = []
            for skill in skills_list:
                if re.search(r'\b' + re.escape(skill.lower()) + r'\b', text_lower):
                    category_skills.append(skill)
            
            if category_skills:
                found_skills[category] = category_skills
        
        return found_skills
    
    def _create_enhanced_text(self, parsed_info, original_text):
        """Create enhanced text for better classification"""
        enhanced_parts = []
        
        # Emphasize skills (repeat important ones)
        for category, skills in parsed_info['skills'].items():
            if 'ml_ai' in category or 'web' in category or 'programming' in category:
                enhanced_parts.extend(skills * 3)  # Triple emphasis
            else:
                enhanced_parts.extend(skills * 2)  # Double emphasis
        
        # Add education context
        for edu in parsed_info['education']:
            if edu['degree']:
                enhanced_parts.append(edu['degree'] * 2)  # Repeat degree
        
        # Add experience context
        for exp in parsed_info['experience']:
            if exp['role']:
                enhanced_parts.append(exp['role'] * 2)  # Repeat role
        
        return ' '.join(enhanced_parts) + ' ' + original_text
    
    def _get_confidence_level(self, confidence):
        """Convert confidence to level"""
        if confidence > 0.7:
            return 'HIGH'
        elif confidence > 0.5:
            return 'MEDIUM'
        else:
            return 'LOW'

def test_working_processor():
    """Test the enhanced processor"""
    
    processor = WorkingResumeProcessor()
    result = processor.process_resume("sample_resume.pdf")
    
    if 'error' in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"\n" + "="*70)
    print(f"📊 COMPLETE RESUME PROCESSING RESULTS")
    print(f"="*70)
    
    # Personal Info
    print(f"👤 Personal Information:")
    for key, value in result['personal_info'].items():
        if value:
            print(f"   {key.title()}: {value}")
    
    # Education
    # Update the education display part to:
    print(f"\n🎓 Education ({result['stats']['education_entries']} entries):")
    if result['education']:
        for i, edu in enumerate(result['education'], 1):
            print(f"   {i}. {edu['degree']}")
            if edu['institution']:
                print(f"      📍 {edu['institution']}")
            if edu['year']:
                print(f"      📅 Year: {edu['year']}")
            if edu.get('cgpa'):
                print(f"      📊 CGPA: {edu['cgpa']}")
    else:
        print("   ❌ No education entries extracted - check debug info above")

    
    # Experience
    print(f"\n💼 Experience ({result['stats']['experience_entries']} entries):")
    for i, exp in enumerate(result['experience'], 1):
        print(f"   {i}. {exp['role']} | {exp['duration']}")
    
    # Skills
    print(f"\n💻 Skills ({result['stats']['total_skills']} total):")
    for category, skills in result['skills'].items():
        print(f"   {category.replace('_', ' ').title()}: {', '.join(skills)}")
    
    # Classification
    print(f"\n🎯 Classification Results:")
    clf = result['classification']
    print(f"   Category: {clf['category']}")
    print(f"   Confidence: {clf['confidence']:.3f} [{clf['confidence_level']}]")
    print(f"   Improvement: {clf['improvement']:+.3f}")
    
    # Stats
    print(f"\n📊 Processing Stats:")
    stats = result['stats']
    print(f"   Text Length: {stats['text_length']} chars")
    print(f"   Extraction Method: {stats['method']}")

if __name__ == "__main__":
    test_working_processor()
