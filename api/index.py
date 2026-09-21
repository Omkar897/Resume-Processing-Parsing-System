"""Vercel Function entry point for the Flask application."""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Vercel Functions can only write to /tmp. The Flask app and scraper share this.
# Keep local imports usable on Windows, where /tmp is not a writable directory.
if os.name != "nt":
    os.environ.setdefault("RESUME_RUNTIME_DIR", "/tmp/resume-processing")

from web.app import app  # noqa: E402
