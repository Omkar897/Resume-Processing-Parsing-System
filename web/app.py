from flask import Flask, render_template, request, jsonify
import os
import sys
import subprocess
import json
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(
    os.path.dirname(__file__), "static", "uploads"
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"pdf"}

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
        scraper_path = os.path.join(project_root, "src", "jobs", "job_scraper.py")
        jobs_folder = os.path.join(project_root, "src", "jobs")

        print(f"📂 Scraper: {scraper_path}")

        # Prepare environment
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        # Run scraper in headless mode
        print(f"🚀 Running scraper in headless mode...")
        result = subprocess.run(
            [sys.executable, scraper_path, abs_filepath],
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            cwd=jobs_folder,
            stdin=subprocess.DEVNULL,  # Force headless mode
        )

        print(f"📊 Return code: {result.returncode}")

        if result.returncode != 0:
            error_msg = (
                f"Scraper failed.\nOutput: {result.stdout}\nError: {result.stderr}"
            )
            print(f"❌ {error_msg}")
            return jsonify({"error": error_msg}), 500

        # Parse output to find JSON filename
        json_file = None
        for line in result.stdout.split("\n"):
            if "✅ Scraped" in line and ".json" in line:
                parts = line.split("→")
                if len(parts) > 1:
                    json_file = parts[-1].strip()
                    print(f"📁 Found JSON: {json_file}")
                    break

        if not json_file:
            return (
                jsonify({"error": f"No JSON file created. Output: {result.stdout}"}),
                500,
            )

        # Read JSON
        json_path = os.path.join(jobs_folder, json_file)

        if not os.path.exists(json_path):
            return jsonify({"error": f"JSON not found: {json_path}"}), 500

        with open(json_path, "r", encoding="utf-8") as f:
            jobs_data = json.load(f)

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
                "scrape_date": jobs_data.get("scrape_date"),
                "scrape_time": jobs_data.get("scrape_time"),
            }
        )

    except Exception as e:
        import traceback

        error_trace = traceback.format_exc()
        print(f"❌ Exception: {error_trace}")
        return jsonify({"error": str(e)}), 500


@app.route("/results")
def results():
    return render_template("results.html")


if __name__ == "__main__":
    print("🌐 Flask server starting...")
    print(f"📂 Upload folder: {app.config['UPLOAD_FOLDER']}")
    app.run(debug=True, port=5000, host="0.0.0.0")
