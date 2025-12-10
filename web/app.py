from flask import Flask, render_template, request, jsonify
import os
import sys
import subprocess
import json
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
        if "resume" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["resume"]

        if file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        if not allowed_file(file.filename):
            return jsonify({"error": "Only PDF files are allowed"}), 400

        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(filepath)

        abs_filepath = os.path.abspath(filepath)
        print(f"✅ File saved: {abs_filepath}")

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        scraper_path = os.path.join(project_root, "src", "jobs", "job_scraper.py")
        jobs_folder = os.path.join(project_root, "src", "jobs")

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        print(f"🚀 Running scraper...")
        result = subprocess.run(
            [sys.executable, scraper_path, abs_filepath],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            cwd=jobs_folder,
            stdin=subprocess.DEVNULL,
        )

        if result.returncode != 0:
            error_msg = (
                f"Scraper failed.\nOutput: {result.stdout}\nError: {result.stderr}"
            )
            print(f"❌ {error_msg}")
            return jsonify({"error": error_msg}), 500

        json_file = None
        for line in result.stdout.split("\n"):
            if "✅ Scraped" in line and ".json" in line:
                parts = line.split("→")
                if len(parts) > 1:
                    json_file = parts[-1].strip()
                    break

        if not json_file:
            return (
                jsonify({"error": f"No JSON file created. Output: {result.stdout}"}),
                500,
            )

        json_path = os.path.join(jobs_folder, json_file)

        if not os.path.exists(json_path):
            return jsonify({"error": f"JSON not found: {json_path}"}), 500

        with open(json_path, "r", encoding="utf-8") as f:
            jobs_data = json.load(f)

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
                "scrape_date": jobs_data.get("scrape_date"),
                "scrape_time": jobs_data.get("scrape_time"),
            }
        )

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        print(f"❌ Exception: {error_trace}")
        return jsonify({"error": str(e)}), 500


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

        msg = Message(
            subject=f"Your Personalized {category} Jobs - {len(jobs)} Matches",
            recipients=[user_email],
        )

        msg.html = generate_email_html(jobs, category, experience)

        mail.send(msg)

        print(f"✅ Email sent to {user_email}")
        return jsonify({"success": True, "message": "Email sent successfully!"})

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        print(f"❌ Email error: {error_trace}")
        return jsonify({"error": f"Failed to send email: {str(e)}"}), 500


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
