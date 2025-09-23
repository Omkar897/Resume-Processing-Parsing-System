# Configuration settings for the job alert system

# Email configuration
EMAIL_SETTINGS = {
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'sender_email': 'oshinde897@gmail.com',  # Replace with your email
    'sender_password': 'gdocjhhrzaateuha',  # Use app password for Gmail
    'recipient_email': 'oshinde897@gmail.com'  # Replace with your email
}

# Scraping configuration
SCRAPING_CONFIG = {
    'delay_between_requests': 2,  # seconds between requests
    'max_retries': 3,
    'timeout': 30,
    'headers': {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
}

# AI/ML configuration
AI_CONFIG = {
    'similarity_threshold': 0.7,  # minimum similarity score for job matching
    'max_jobs_per_alert': 10,
    'model_name': 'sentence-transformers/all-MiniLM-L6-v2'  # for embeddings
}

# Company URLs to scrape (we'll start with a few)
COMPANY_URLS = {
    'indeed': 'https://www.indeed.com/jobs?q=software+engineer&l=',
    'linkedin': 'https://www.linkedin.com/jobs/search/?keywords=software%20engineer',
    # We'll add more specific company URLs later
}

# File paths
DATA_PATHS = {
    'jobs_data': 'data/jobs/',
    'resumes': 'data/resumes/',
    'logs': 'data/logs/'
}

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'data/logs/app.log'
}
