import subprocess
import sys
import os
import re
import requests
import json
import math
from datetime import datetime, timedelta

env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"


class IntegratedJobScraper:
    def __init__(self):
        self.serpapi_key = (
            "206fd79ebb2efca9f63d527a53171d31ef42d9351da6af8219b4f3ed20c96a11"
        )

    def parse_duration_to_months(self, duration_str):
        """Parse duration string like 'June-Aug 2024' or 'Jan-Feb 2024' to months"""
        try:
            month_map = {
                "jan": 1,
                "feb": 2,
                "mar": 3,
                "apr": 4,
                "may": 5,
                "jun": 6,
                "jul": 7,
                "aug": 8,
                "sep": 9,
                "oct": 10,
                "nov": 11,
                "dec": 12,
                "january": 1,
                "february": 2,
                "march": 3,
                "april": 4,
                "june": 6,
                "july": 7,
                "august": 8,
                "september": 9,
                "october": 10,
                "november": 11,
                "december": 12,
            }

            duration_lower = duration_str.lower().strip()

            if "-" in duration_str:
                parts = duration_str.replace("–", "-").split("-")
                start_match = re.search(r"([a-zA-Z]+)\s*(\d{4})?", parts[0])
                end_match = re.search(r"([a-zA-Z]+)\s*(\d{4})", parts[1])

                if start_match and end_match:
                    start_month_str = start_match.group(1).lower()[:3]
                    end_month_str = end_match.group(1).lower()[:3]

                    start_month = month_map.get(start_month_str, 1)
                    end_month = month_map.get(end_month_str, 12)

                    months = end_month - start_month + 1
                    return max(months, 1)

            return 3

        except:
            return 3

    def get_resume_data(self, resume_path):
        """Extract category and calculate total experience from resume"""
        try:
            main_script_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "api_only_main.py"
            )
            cmd = [sys.executable, main_script_path, "--file", resume_path]

            # Run with error handling
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=env,
                errors="ignore",  # Ignore encoding errors
            )

            # Get both stdout and stderr
            output = result.stdout + result.stderr
            output_lines = output.split("\n")

            category = None
            total_months = 0
            in_experience_section = False

            for i, line in enumerate(output_lines):
                # Try multiple patterns to find category
                # Pattern 1: With emoji
                if "Predicted Category:" in line:
                    category_match = re.search(r"Predicted Category:\s*(.+)", line)
                    if category_match:
                        category = category_match.group(1).strip()
                        print(f"✓ Found category: {category}")

                # Pattern 2: Without emoji (if encoding failed)
                if "predicted category" in line.lower() and not category:
                    category_match = re.search(
                        r"category[:\s]+([A-Za-z\s]+)", line, re.IGNORECASE
                    )
                    if category_match:
                        category = category_match.group(1).strip()
                        print(f"✓ Found category (alt): {category}")

                # Detect experience section
                if "WORK EXPERIENCE" in line:
                    in_experience_section = True
                    continue

                if in_experience_section and (
                    "PROJECTS" in line or "EDUCATION" in line or "SKILLS" in line
                ):
                    in_experience_section = False

                # Parse duration
                if in_experience_section and "Duration:" in line:
                    duration_match = re.search(r"Duration:\s*(.+)", line)
                    if duration_match:
                        duration_str = duration_match.group(1).strip()
                        months = self.parse_duration_to_months(duration_str)
                        total_months += months
                        print(f"✓ Found duration: {duration_str} = {months} months")

            total_years = total_months / 12.0

            if not category:
                print("⚠️  Could not extract category from resume output")
                print("First 20 lines of output:")
                for line in output_lines[:20]:
                    print(f"  {line}")

            return category, total_years

        except Exception as e:
            print(f"❌ Error in get_resume_data: {e}")
            import traceback

            traceback.print_exc()
            return None, 0

    def build_search_query(self, category, years_experience):
        """Build appropriate search query based on experience"""
        rounded_years = math.ceil(years_experience)

        if rounded_years == 0:
            query = f"entry level {category}"
        elif rounded_years == 1:
            query = f"{category} 0-1 year experience"
        else:
            query = f"{category} {rounded_years} years experience"

        return query, rounded_years

    def parse_time_posted(self, posted_str):
        """Convert 'X hours ago', 'X days ago' to comparable timestamp"""
        try:
            posted_lower = posted_str.lower()
            now = datetime.now()

            if "just now" in posted_lower or "today" in posted_lower:
                return now

            numbers = re.findall(r"\d+", posted_str)
            if not numbers:
                return now - timedelta(days=365)

            value = int(numbers[0])

            if "minute" in posted_lower or "min" in posted_lower:
                return now - timedelta(minutes=value)
            elif "hour" in posted_lower:
                return now - timedelta(hours=value)
            elif "day" in posted_lower:
                return now - timedelta(days=value)
            elif "week" in posted_lower:
                return now - timedelta(weeks=value)
            elif "month" in posted_lower:
                return now - timedelta(days=value * 30)
            else:
                return now - timedelta(days=365)

        except:
            return datetime.now() - timedelta(days=365)

    def clean_job_data(self, job_data):
        """Remove empty, N/A, or 'Not specified' fields from job data"""
        cleaned = {}

        for key, value in job_data.items():
            if value in ["N/A", "Not specified", "", "Recently", None]:
                continue
            if isinstance(value, list) and len(value) == 0:
                continue
            if isinstance(value, bool) and value is False:
                continue
            if key == "posted_timestamp":
                continue

            cleaned[key] = value

        return cleaned

    def scrape_google_jobs_serpapi(self, search_query, location="India", max_jobs=10):
        """Use SerpAPI to scrape Google Jobs with experience-based query"""
        jobs = []
        try:
            url = "https://serpapi.com/search"
            params = {
                "engine": "google_jobs",
                "q": search_query,
                "location": location,
                "api_key": self.serpapi_key,
                "hl": "en",
                "gl": "in",
            }

            response = requests.get(url, params=params, timeout=20)

            if response.status_code != 200:
                return jobs

            data = response.json()

            if "jobs_results" in data and isinstance(data["jobs_results"], list):
                for job in data["jobs_results"]:
                    extensions = job.get("detected_extensions", {})
                    apply_options = job.get("apply_options", [])
                    apply_link = (
                        apply_options[0].get("link")
                        if apply_options
                        else job.get("share_url", "")
                    )
                    related_links = job.get("related_links", [])

                    posted_at_str = extensions.get("posted_at", "")

                    job_data = {
                        "title": job.get("title", ""),
                        "company": job.get("company_name", ""),
                        "location": job.get("location", ""),
                        "via": job.get("via", ""),
                        "posted_at": posted_at_str,
                        "posted_timestamp": self.parse_time_posted(posted_at_str),
                        "schedule_type": extensions.get("schedule_type", ""),
                        "salary": extensions.get("salary", ""),
                        "work_from_home": extensions.get("work_from_home", False),
                        "apply_link": apply_link,
                        "description": (
                            job.get("description", "")[:400] + "..."
                            if job.get("description")
                            else ""
                        ),
                        "job_highlights": job.get("job_highlights", []),
                        "related_links": [
                            {
                                "title": link.get("text", ""),
                                "link": link.get("link", ""),
                            }
                            for link in related_links[:3]
                            if link.get("link")
                        ],
                    }

                    cleaned_job = self.clean_job_data(job_data)
                    cleaned_job["posted_timestamp"] = job_data["posted_timestamp"]

                    if cleaned_job:
                        jobs.append(cleaned_job)

                jobs.sort(
                    key=lambda x: x.get("posted_timestamp", datetime.now()),
                    reverse=True,
                )

                for job in jobs:
                    if "posted_timestamp" in job:
                        del job["posted_timestamp"]

                jobs = jobs[:max_jobs]

        except Exception as e:
            print(f"SerpAPI Error: {e}")

        return jobs

    def process_resume_and_scrape_jobs(self, resume_path, max_jobs=10):
        """Main method - calculate experience and search accordingly"""
        print("🔄 Analyzing resume...")
        category, years_experience = self.get_resume_data(resume_path)

        if not category:
            print("\n❌ Failed to extract category automatically")
            category = input("Please enter job category manually: ").strip()
            if not category:
                print("No category provided. Exiting.")
                return [], None, 0, 0

        search_query, rounded_years = self.build_search_query(
            category, years_experience
        )

        print(f"\n📊 Category: {category}")
        print(
            f"💼 Experience: {years_experience:.1f} years → Rounded to {rounded_years} years"
        )
        print(f"🔍 Searching: '{search_query}'")
        print()

        all_jobs = self.scrape_google_jobs_serpapi(
            search_query, location="India", max_jobs=max_jobs
        )

        return all_jobs, category, years_experience, rounded_years

    def save_results(self, jobs, category, years_experience, rounded_years):
        """Save results to JSON with date and experience in filename"""
        today = datetime.now().strftime("%Y-%m-%d")

        results = {
            "search_category": category,
            "total_experience_years": round(years_experience, 1),
            "search_experience_years": rounded_years,
            "scrape_date": today,
            "scrape_time": datetime.now().strftime("%I:%M %p"),
            "total_jobs": len(jobs),
            "jobs": jobs,
            "timestamp": datetime.now().isoformat(),
            "source": "Google Jobs via SerpAPI",
            "sorted_by": "Most recent first",
        }

        filename = (
            f"{category.replace(' ', '_').lower()}_{rounded_years}yrs_{today}.json"
        )
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return filename


def main():
    if len(sys.argv) != 2:
        print("Usage: python job_scraper.py <resume_file_path>")
        return

    resume_path = sys.argv[1]

    if not os.path.exists(resume_path):
        print(f"❌ File not found: {resume_path}")
        return

    scraper = IntegratedJobScraper()
    jobs, category, years_exp, rounded_years = scraper.process_resume_and_scrape_jobs(
        resume_path
    )

    if jobs and category:
        filename = scraper.save_results(jobs, category, years_exp, rounded_years)
        print(f"✅ Scraped {len(jobs)} jobs → {filename}")
    else:
        print(f"❌ No matching jobs found")


if __name__ == "__main__":
    main()
