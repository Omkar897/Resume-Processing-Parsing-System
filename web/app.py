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


def is_render_environment():
    return os.getenv("RENDER", "").strip().lower() in {"1", "true", "yes"}


def is_email_enabled():
    explicit = (os.getenv("RENDER_EMAIL_DISABLED") or "").strip().lower()
    if explicit in {"1", "true", "yes"}:
        return False
    if explicit in {"0", "false", "no"}:
        return True
    # Render free services block SMTP ports (25/465/587).
    if is_render_environment():
        return False
    return bool(app.config.get("MAIL_USERNAME") and app.config.get("MAIL_PASSWORD"))


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _clean_subprocess_error(raw_output):
    """Filter noisy warnings and return the most relevant error lines."""
    if not raw_output:
        return ""

    lines = [line.strip() for line in raw_output.splitlines() if line.strip()]
    if not lines:
        return ""

    noisy_markers = [
        "FutureWarning",
        "transformers/utils/hub.py",
        "Using 'TRANSFORMERS_CACHE' is deprecated",
    ]
    filtered = [
        line for line in lines if not any(marker in line for marker in noisy_markers)
    ]

    chosen = filtered if filtered else lines
    return "\n".join(chosen[-10:])


def _extract_last_json_object(output_text):
    """Extract the last valid JSON object from noisy subprocess output."""
    if not output_text:
        return None
    text = output_text.strip()
    if not text:
        return None

    for line in reversed(text.splitlines()):
        line = line.strip()
        if not (line.startswith("{") and line.endswith("}")):
            continue
        try:
            return json.loads(line)
        except Exception:
            continue

    start = text.rfind("{")
    while start != -1:
        candidate = text[start:].strip()
        try:
            return json.loads(candidate)
        except Exception:
            start = text.rfind("{", 0, start)
    return None


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
        print(f"[Upload] File saved: {abs_filepath}")

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
        env["RESUME_HEADLESS"] = "1"  # Disable debug output for production

        # Explicitly pass Fireworks environment variables
        env["FIREWORKS_API_KEY"] = os.getenv("FIREWORKS_API_KEY", "")
        env["FIREWORKS_PRIMARY_CHAT_MODEL"] = os.getenv(
            "FIREWORKS_PRIMARY_CHAT_MODEL", "fireworks/minimax-m2p7"
        )
        env["FIREWORKS_FALLBACK_CHAT_MODEL"] = os.getenv(
            "FIREWORKS_FALLBACK_CHAT_MODEL", "fireworks/deepseek-v3p2"
        )
        env["FIREWORKS_EMBED_MODEL"] = os.getenv(
            "FIREWORKS_EMBED_MODEL", "fireworks/qwen3-embedding-8b"
        )
        env["FIREWORKS_RERANK_MODEL"] = os.getenv(
            "FIREWORKS_RERANK_MODEL", "fireworks/qwen3-reranker-8b"
        )
        env["USE_FIREWORKS_LLM"] = os.getenv("USE_FIREWORKS_LLM", "1")
        env["USE_LLM_QUERY_EXPANSION"] = os.getenv("USE_LLM_QUERY_EXPANSION", "1")
        env["USE_LLM_RERANK"] = os.getenv("USE_LLM_RERANK", "1")

        # Debug: Check if API key is being passed
        print(
            f"DEBUG: FIREWORKS_API_KEY being passed: {bool(env.get('FIREWORKS_API_KEY'))}"
        )
        print(
            f"DEBUG: FIREWORKS_API_KEY length: {len(env.get('FIREWORKS_API_KEY', ''))}"
        )

        cache_dir = os.path.join(project_root, "data", ".embedding_cache")
        os.makedirs(cache_dir, exist_ok=True)
        env["HF_HOME"] = os.path.abspath(cache_dir)
        env["HF_HUB_CACHE"] = os.path.abspath(os.path.join(cache_dir, "hub"))
        env.pop("TRANSFORMERS_CACHE", None)

        print("[Upload] Running RAG-enhanced scraper...")
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
            if (
                "⚠️" in line
                or "[RAG]" in line
                or "RAG engine disabled" in line
                or "Claude analyzer disabled" in line
                or "Fireworks resume intelligence disabled" in line
            ):
                print(f"[Scraper] {line}")

        # Enhanced error handling
        if result.returncode != 0:
            error_output = result.stdout + result.stderr

            # Check for specific errors and provide friendly messages
            if (
                "Failed to extract category" in error_output
                or "Could not identify" in error_output
            ):
                reason_match = re.search(r"Reason:\s*(.+)", error_output)
                reason_text = (
                    reason_match.group(1).strip()
                    if reason_match
                    else "Automatic extraction/categorization failed."
                )
                user_msg = (
                    "Unable to identify your job category from the resume.\n\n"
                    "Automatic extraction or categorization failed.\n"
                    f"Reason: {reason_text}\n\n"
                    "Please try again after checking Fireworks API connectivity and your resume text quality."
                )

            elif "No module named" in error_output:
                user_msg = "System error: Missing required dependencies. Please contact support."

            elif "FileNotFoundError" in error_output or "No such file" in error_output:
                user_msg = "Error reading the PDF file. Please ensure it's a valid PDF and try again."

            elif "PDF" in error_output and "error" in error_output.lower():
                user_msg = "Unable to extract text from this PDF. The file might be corrupted or image-based. Please try a different resume."

            elif "timeout" in error_output.lower():
                user_msg = "Processing is taking too long. Please try with a smaller or simpler resume."

            else:
                cleaned_details = _clean_subprocess_error(error_output)
                if not cleaned_details:
                    cleaned_details = "Unknown processing error."
                user_msg = (
                    "Unable to process your resume. Please try again with a different file.\n\n"
                    f"Details: {cleaned_details[:1200]}"
                )

            print(f"[Upload][Error] Full error: {error_output}")

            # Clean up uploaded file
            try:
                os.remove(filepath)
            except:
                pass

            return jsonify({"error": user_msg}), 500

        jobs_data = None
        for line in reversed((result.stdout or "").splitlines()):
            if line.startswith("RESULT_JSON:"):
                payload = line[len("RESULT_JSON:") :].strip()
                try:
                    jobs_data = json.loads(payload)
                except Exception:
                    jobs_data = None
                break

        if jobs_data is None:
            jobs_data = _extract_last_json_object(result.stdout or "")

        if not isinstance(jobs_data, dict):
            try:
                os.remove(filepath)
            except:
                pass
            return (
                jsonify(
                    {
                        "error": "Processing completed but no structured results were returned. Please try again."
                    }
                ),
                500,
            )

        # Check if we actually got jobs
        if not jobs_data.get("jobs") or len(jobs_data.get("jobs", [])) == 0:
            try:
                os.remove(filepath)
            except:
                pass
            return (
                jsonify(
                    {
                        "error": f"No jobs found for your profile ({jobs_data.get('search_category', 'Unknown category')}). This could mean:\n- No current openings match your experience level\n- Try again later as new jobs are posted daily\n- Consider broadening your job search"
                    }
                ),
                404,
            )

        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass

        print("[Upload] Successfully processed resume")

        return jsonify(
            {
                "success": True,
                "category": jobs_data.get("search_category"),
                "experience": jobs_data.get("total_experience_years"),
                "experience_months": jobs_data.get("total_experience_months"),
                "experience_display": jobs_data.get("experience_display"),
                "search_experience": jobs_data.get("search_experience_years"),
                "total_jobs": jobs_data.get("total_jobs"),
                "jobs": jobs_data.get("jobs", []),
                "resume_analysis": jobs_data.get("resume_analysis", {}),
                "resume_extracted_data": jobs_data.get("resume_extracted_data", {}),
                "rag_enabled": jobs_data.get("rag_enabled", False),
                "claude_enabled": jobs_data.get("claude_enabled", False),
                "llm_provider": jobs_data.get("llm_provider", "unknown"),
                "llm_model": jobs_data.get("llm_model") or jobs_data.get("category_model"),
                "scrape_date": jobs_data.get("scrape_date"),
                "scrape_time": jobs_data.get("scrape_time"),
                "email_enabled": is_email_enabled(),
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
        print(f"[Upload][Error] Exception: {error_trace}")

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
    """Run resume analysis on-demand from the results page."""
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
        if not is_email_enabled():
            return (
                jsonify(
                    {
                        "error": (
                            "Email delivery is disabled in this deployment environment. "
                            "Use local run or enable an HTTP email provider."
                        )
                    }
                ),
                503,
            )

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

        print(f"[Email] Sent to {user_email}")
        return jsonify({"success": True, "message": "Email sent successfully!"})

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        print(f"[Email][Error] {error_trace}")

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
    """Generate HTML email with job listings using the app's current design language."""

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {{
                margin: 0;
                padding: 24px;
                background: radial-gradient(1000px 500px at 0% 0%, #1e5f74 0%, transparent 58%),
                            radial-gradient(800px 420px at 100% 100%, #1d6257 0%, transparent 55%),
                            linear-gradient(145deg, #07131d, #102a35 48%, #163f4f);
                font-family: 'Segoe UI', Arial, sans-serif;
                color: #e9fbff;
            }}
            .container {{
                max-width: 680px;
                margin: 0 auto;
                background: rgba(11, 22, 30, 0.78);
                border: 1px solid rgba(165, 235, 255, 0.16);
                border-radius: 18px;
                overflow: hidden;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
            }}
            .header {{
                padding: 28px 30px;
                border-bottom: 1px solid rgba(165, 235, 255, 0.14);
                background: linear-gradient(120deg, rgba(69, 240, 255, 0.12), rgba(107, 255, 203, 0.08));
            }}
            .header h1 {{
                margin: 0;
                font-size: 24px;
                letter-spacing: -0.02em;
                color: #f2feff;
            }}
            .header p {{
                margin: 10px 0 0 0;
                color: #a9d2dd;
                font-size: 14px;
            }}
            .content {{
                padding: 24px;
            }}
            .job-card {{
                border: 1px solid rgba(165, 235, 255, 0.18);
                border-radius: 14px;
                background: rgba(10, 21, 29, 0.68);
                padding: 18px;
                margin-bottom: 14px;
            }}
            .job-title {{
                margin: 0;
                color: #eafcff;
                font-size: 18px;
                line-height: 1.35;
            }}
            .job-company {{
                margin: 6px 0 10px 0;
                color: #6bffcb;
                font-size: 14px;
                font-weight: 600;
            }}
            .job-meta {{
                margin: 8px 0 10px 0;
            }}
            .meta-tag {{
                display: inline-block;
                margin: 4px 6px 0 0;
                padding: 5px 10px;
                border-radius: 999px;
                border: 1px solid rgba(165, 235, 255, 0.22);
                background: rgba(15, 32, 42, 0.76);
                color: #b9dfe8;
                font-size: 12px;
            }}
            .job-description {{
                color: #a9d2dd;
                line-height: 1.6;
                font-size: 13px;
                margin: 10px 0 0 0;
            }}
            .apply-btn {{
                display: inline-block;
                margin-top: 14px;
                text-decoration: none;
                color: #05212b;
                font-weight: 700;
                font-size: 13px;
                padding: 10px 16px;
                border-radius: 999px;
                background: linear-gradient(110deg, #45f0ff, #7dfff1, #6bffcb);
            }}
            .footer {{
                padding: 18px 24px;
                border-top: 1px solid rgba(165, 235, 255, 0.12);
                color: #95c4d0;
                font-size: 12px;
                background: rgba(8, 18, 25, 0.65);
            }}
            .footer p {{
                margin: 4px 0;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Your Personalized Job Matches</h1>
                <p>{category} | {experience} years experience | {len(jobs)} jobs found</p>
            </div>
            <div class="content">
    """

    for job in jobs:
        html += f"""
                <div class="job-card">
                    <h2 class="job-title">{job.get('title', 'N/A')}</h2>
                    <p class="job-company">{job.get('company', 'N/A')}</p>
                    <div class="job-meta">
        """

        if job.get("location"):
            html += f'<span class="meta-tag">Location: {job["location"]}</span>'
        if job.get("posted_at"):
            html += f'<span class="meta-tag">Posted: {job["posted_at"]}</span>'
        if job.get("schedule_type"):
            html += f'<span class="meta-tag">Type: {job["schedule_type"]}</span>'
        if job.get("salary"):
            html += f'<span class="meta-tag">Salary: {job["salary"]}</span>'
        if job.get("via"):
            html += f'<span class="meta-tag">Source: {job["via"]}</span>'

        html += """
                    </div>
        """

        if job.get("description"):
            html += f'<p class="job-description">{job["description"]}</p>'

        apply_link = job.get("apply_link", "#")
        html += f"""
                    <a href="{apply_link}" class="apply-btn" target="_blank">Open Application</a>
                </div>
        """

    html += f"""
            </div>
            <div class="footer">
                <p>Generated on {datetime.now().strftime('%B %d, %Y')}</p>
                <p>Sent by AI Resume Job Matcher</p>
            </div>
        </div>
    </body>
    </html>
    """

    return html


@app.route("/healthz")
def healthz():
    return jsonify({"status": "ok"})


@app.route("/results")
def results():
    return render_template("results.html")


if __name__ == "__main__":
    print("Flask server starting...")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    debug = os.getenv("FLASK_DEBUG", "0").strip().lower() in {"1", "true", "yes"}
    port = int(os.getenv("PORT", "5000"))
    app.run(debug=debug, port=port, host="0.0.0.0")
