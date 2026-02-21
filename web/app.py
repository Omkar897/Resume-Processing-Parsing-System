from flask import Flask, render_template, request, jsonify
import os
import sys
import subprocess
import json
import re
from datetime import datetime
from werkzeug.utils import secure_filename
from flask_mail import Mail, Message
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(
    os.path.dirname(__file__), "static", "uploads"
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"pdf"}

# Email configuration
app.config["MAIL_SERVER"] = "smtp.gmail.com"
app.config["MAIL_PORT"] = 587
app.config["MAIL_USE_TLS"] = True
app.config["MAIL_USERNAME"] = os.getenv("MAIL_USERNAME")
app.config["MAIL_PASSWORD"] = os.getenv("MAIL_PASSWORD")
app.config["MAIL_DEFAULT_SENDER"] = os.getenv("MAIL_USERNAME")

mail = Mail(app)

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_resume():
    try:
        # Check if file exists in request
        if "resume" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["resume"]

        # Check if filename is empty
        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Check file type
        if not allowed_file(file.filename):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        # Check file size (additional check beyond MAX_CONTENT_LENGTH)
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)

        if file_size < 1000:  # Less than 1KB
            return (
                jsonify(
                    {"error": "File is too small. Please upload a valid resume PDF."}
                ),
                400,
            )

        if file_size > 16 * 1024 * 1024:  # More than 16MB
            return jsonify({"error": "File is too large. Maximum size is 16MB."}), 400

        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(filepath)

        abs_filepath = os.path.abspath(filepath)
        print(f"✅ File saved: {abs_filepath}")

        # Get paths
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scraper_path = os.path.join(
            project_root, "src", "jobs", "enhanced_job_scraper.py"
        )
        jobs_folder = os.path.join(project_root, "src", "jobs")

        # Prepare environment (run from project root so RAG/data paths and model cache resolve correctly)
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        env["RESUME_PROJECT_ROOT"] = project_root
        env["RESUME_HEADLESS"] = "1"
        cache_dir = os.path.join(project_root, "data", ".embedding_cache")
        os.makedirs(cache_dir, exist_ok=True)
        env["TRANSFORMERS_CACHE"] = os.path.abspath(cache_dir)
        env["HF_HOME"] = os.path.abspath(cache_dir)

        print(f"🚀 Running RAG-enhanced scraper...")
        result = subprocess.run(
            [sys.executable, scraper_path, abs_filepath],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            cwd=project_root,
            stdin=subprocess.DEVNULL,
            timeout=120,  # Increased timeout for RAG processing
        )

        # Log scraper output so we can see RAG/Claude init warnings (stdout is captured, not shown)
        scraper_out = (result.stdout or "") + (result.stderr or "")
        for line in scraper_out.splitlines():
            if "⚠️" in line or "RAG engine disabled" in line or "Claude analyzer disabled" in line:
                print(f"[Scraper] {line}")

        # Enhanced error handling
        if result.returncode != 0:
            error_output = result.stdout + result.stderr

            # Check for specific errors and provide friendly messages
            if (
                "Failed to extract category" in error_output
                or "Could not identify" in error_output
            ):
                user_msg = """Unable to identify your job category from the resume. 

Please ensure your resume includes:
• Clear job titles (e.g., Software Engineer, Data Scientist)
• Work Experience section with company names
• Skills section
• Education details

Try uploading a more detailed resume or a different format."""

            elif "No module named" in error_output:
                user_msg = "System error: Missing required dependencies. Please contact support."

            elif "FileNotFoundError" in error_output or "No such file" in error_output:
                user_msg = "Error reading the PDF file. Please ensure it's a valid PDF and try again."

            elif "PDF" in error_output and "error" in error_output.lower():
                user_msg = "Unable to extract text from this PDF. The file might be corrupted or image-based. Please try a different resume."

            elif "timeout" in error_output.lower():
                user_msg = "Processing is taking too long. Please try with a smaller or simpler resume."

            else:
                # Generic error with first 200 chars
                user_msg = f"Unable to process your resume. Please try again with a different file.\n\nDetails: {error_output[:200]}"

            print(f"❌ Full error: {error_output}")

            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass

            return jsonify({"error": user_msg}), 500

        # Parse output to find JSON filename (scraper writes it to cwd = project_root when run from web)
        json_file = None
        for line in result.stdout.split("\n"):
            if "Scraped" in line and ".json" in line:
                # Handles both clean UTF-8 and mojibake output variants.
                match = re.search(r"([^\s]+\.json)", line)
                if match:
                    json_file = match.group(1).strip()
                    break

        # Fallback: scan full stdout if line parsing missed it.
        if not json_file:
            match = re.search(r"([^\s]+\.json)", result.stdout or "")
            if match:
                json_file = match.group(1).strip()

        if not json_file:
            # Clean up
            try:
                os.remove(filepath)
            except:
                pass
            return (
                jsonify(
                    {
                        "error": "Processing completed but no jobs were found. This might happen if:\n• The resume category is unclear\n• No matching jobs are currently available\n\nPlease try again or upload a different resume."
                    }
                ),
                500,
            )

        # Read JSON (scraper runs with cwd=project_root, so file is there)
        json_path = os.path.join(project_root, json_file)

        if not os.path.exists(json_path):
            try:
                os.remove(filepath)
            except:
                pass
            return (
                jsonify(
                    {
                        "error": f"Results file not found. Please try uploading your resume again."
                    }
                ),
                500,
            )

        with open(json_path, "r", encoding="utf-8") as f:
            jobs_data = json.load(f)

        # Check if we actually got jobs
        if not jobs_data.get("jobs") or len(jobs_data.get("jobs", [])) == 0:
            try:
                os.remove(filepath)
            except:
                pass
            return (
                jsonify(
                    {
                        "error": f"No jobs found for your profile ({jobs_data.get('search_category', 'Unknown category')}). This could mean:\n• No current openings match your experience level\n• Try again later as new jobs are posted daily\n• Consider broadening your job search"
                    }
                ),
                404,
            )

        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass

        print(f"✅ Successfully processed resume")

        return jsonify(
            {
                "success": True,
                "category": jobs_data.get("search_category"),
                "experience": jobs_data.get("total_experience_years"),
                "search_experience": jobs_data.get("search_experience_years"),
                "total_jobs": jobs_data.get("total_jobs"),
                "jobs": jobs_data.get("jobs", []),
                "resume_analysis": jobs_data.get("resume_analysis", {}),
                "resume_extracted_data": jobs_data.get("resume_extracted_data", {}),
                "rag_enabled": jobs_data.get("rag_enabled", False),
                "claude_enabled": jobs_data.get("claude_enabled", False),
                "scrape_date": jobs_data.get("scrape_date"),
                "scrape_time": jobs_data.get("scrape_time"),
            }
        )

    except subprocess.TimeoutExpired:
        # Clean up
        try:
            os.remove(filepath)
        except:
            pass
        return (
            jsonify(
                {
                    "error": "Processing timeout. Your resume is taking too long to analyze. Please try with a simpler PDF."
                }
            ),
            500,
        )

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        print(f"❌ Exception: {error_trace}")

        # Clean up uploaded file
        try:
            if "filepath" in locals():
                os.remove(filepath)
        except:
            pass

        # User-friendly error message
        return (
            jsonify(
                {
                    "error": f"An unexpected error occurred while processing your resume. Please try again.\n\nError: {str(e)}"
                }
            ),
            500,
        )


@app.route("/analyze-resume", methods=["POST"])
def analyze_resume():
    """Run Claude resume analysis on-demand from the results page."""
    data = request.get_json(silent=True) or {}
    extracted_data = data.get("extracted_data") or data.get("resume_extracted_data")
    category = data.get("category")

    if not extracted_data or not category:
        return (
            jsonify(
                {
                    "success": False,
                    "error": "Missing required fields: extracted_data and category",
                }
            ),
            400,
        )

    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from src.rag.resume_analyzer import ResumeAnalyzer

        analyzer = ResumeAnalyzer()
        analysis = analyzer.analyze_resume(extracted_data, category)
        return jsonify({"success": True, "resume_analysis": analysis})
    except Exception as e:
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(e),
                }
            ),
            500,
        )


@app.route("/send-email", methods=["POST"])
def send_email():
    """Send job results via email"""
    try:
        data = request.json
        user_email = data.get("email")
        jobs = data.get("jobs", [])
        category = data.get("category", "Jobs")
        experience = data.get("experience", 0)

        if not user_email:
            return jsonify({"error": "Email address required"}), 400

        if not jobs:
            return jsonify({"error": "No jobs to send"}), 400

        # Create message
        msg = Message(
            subject=f"Your Personalized {category} Jobs - {len(jobs)} Matches",
            recipients=[user_email],
        )

        msg.html = generate_email_html(jobs, category, experience)

        # Send email
        mail.send(msg)

        print(f"✅ Email sent to {user_email}")
        return jsonify({"success": True, "message": "Email sent successfully!"})

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        print(f"❌ Email error: {error_trace}")

        # Specific email errors
        if "Authentication" in str(e) or "Username and Password not accepted" in str(e):
            error_msg = (
                "Email authentication failed. Please check your email configuration."
            )
        elif "Connection" in str(e):
            error_msg = "Unable to connect to email server. Please check your internet connection."
        elif "Invalid" in str(e) and "address" in str(e):
            error_msg = "Invalid email address. Please check and try again."
        else:
            error_msg = f"Failed to send email. Please try again. Error: {str(e)[:100]}"

        return jsonify({"error": error_msg}), 500


def generate_email_html(jobs, category, experience):
    """Generate beautiful HTML email with job listings"""

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                background-color: #f5f5f5;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                max-width: 600px;
                margin: 0 auto;
                background: white;
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }}
            .header h1 {{
                margin: 0;
                font-size: 24px;
            }}
            .header p {{
                margin: 10px 0 0 0;
                opacity: 0.9;
            }}
            .content {{
                padding: 30px;
            }}
            .job-card {{
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
            }}
            .job-title {{
                font-size: 18px;
                color: #333;
                font-weight: bold;
                margin: 0 0 5px 0;
            }}
            .job-company {{
                color: #667eea;
                font-size: 16px;
                font-weight: 600;
                margin: 0 0 10px 0;
            }}
            .job-meta {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
                margin: 10px 0;
            }}
            .meta-tag {{
                background: #f0f0f0;
                padding: 5px 12px;
                border-radius: 15px;
                font-size: 13px;
                color: #666;
            }}
            .job-description {{
                color: #666;
                line-height: 1.6;
                margin: 10px 0;
                font-size: 14px;
            }}
            .apply-btn {{
                display: inline-block;
                background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                color: white;
                padding: 12px 25px;
                border-radius: 25px;
                text-decoration: none;
                font-weight: bold;
                margin-top: 10px;
            }}
            .footer {{
                background: #f9f9f9;
                padding: 20px;
                text-align: center;
                color: #666;
                font-size: 14px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎯 Your Personalized Job Matches</h1>
                <p>{category} • {experience} years experience • {len(jobs)} jobs found</p>
            </div>
            
            <div class="content">
    """

    for i, job in enumerate(jobs, 1):
        html += f"""
                <div class="job-card">
                    <h2 class="job-title">{job.get('title', 'N/A')}</h2>
                    <p class="job-company">🏢 {job.get('company', 'N/A')}</p>
                    
                    <div class="job-meta">
        """

        if job.get("location"):
            html += f'<span class="meta-tag">📍 {job["location"]}</span>'
        if job.get("posted_at"):
            html += f'<span class="meta-tag">⏰ {job["posted_at"]}</span>'
        if job.get("schedule_type"):
            html += f'<span class="meta-tag">💼 {job["schedule_type"]}</span>'
        if job.get("salary"):
            html += f'<span class="meta-tag">💰 {job["salary"]}</span>'
        if job.get("via"):
            html += f'<span class="meta-tag">📢 via {job["via"]}</span>'

        html += """
                    </div>
        """

        if job.get("description"):
            html += f'<p class="job-description">{job["description"]}</p>'

        apply_link = job.get("apply_link", "#")
        html += f"""
                    <a href="{apply_link}" class="apply-btn" target="_blank">Apply Now →</a>
                </div>
        """

    html += f"""
            </div>
            
            <div class="footer">
                <p>📧 These jobs were found on {datetime.now().strftime('%B %d, %Y')}</p>
                <p>Good luck with your job search! 🚀</p>
                <p style="font-size: 12px; margin-top: 15px;">
                    Generated by AI Resume Job Matcher
                </p>
            </div>
        </div>
    </body>
    </html>
    """

    return html


@app.route("/results")
def results():
    return render_template("results.html")


if __name__ == "__main__":
    print("🌐 Flask server starting...")
    print(f"📂 Upload folder: {app.config['UPLOAD_FOLDER']}")
    app.run(debug=True, port=5000, host="0.0.0.0")
