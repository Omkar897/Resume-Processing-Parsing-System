# 🎯 AI Resume Job Matcher

An intelligent web application that automatically analyzes resumes using machine learning, extracts job categories and experience levels, and fetches personalized job recommendations from Google Jobs in real-time.

## ✨ Features

- 🤖 **AI-Powered Resume Analysis** - Machine learning classifier identifies job categories (Software Engineer, Data Scientist, etc.)
- 💼 **Smart Experience Detection** - Automatically calculates total work experience from resume
- 🎯 **Personalized Job Matching** - Fetches jobs tailored to your exact experience level (entry-level, 2 years, 5+ years, etc.)
- 🔍 **Real-Time Job Search** - Scrapes latest job postings from Google Jobs via SerpAPI
- 📊 **Beautiful Web Interface** - Modern drag-and-drop UI with gradient design
- ⚡ **Instant Results** - Get 10 perfectly matched jobs sorted by posting date

## 🛠️ Tech Stack

**Backend:**
- Python 3.9+
- Flask (Web Framework)
- scikit-learn (ML Classification)
- SerpAPI (Job Scraping)
- PyPDF2 (Resume Parsing)

**Frontend:**
- HTML5/CSS3
- Vanilla JavaScript
- Responsive Design

**ML Model:**
- TF-IDF Vectorization
- Trained Resume Classifier
- Experience Extraction Algorithm

## 📋 Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Git
- SerpAPI account (free tier available)
- APILayer account (optional, for enhanced parsing)

## 🚀 Installation & Setup

### 1. Clone the Repository

git clone https://github.com/yourusername/resume-job-matcher.git
cd resume-job-matcher

text

### 2. Create Virtual Environment

**Windows:**
python -m venv venv
venv\Scripts\activate

text

**Mac/Linux:**
python3 -m venv venv
source venv/bin/activate

text

### 3. Install Dependencies

pip install -r requirements.txt

text

### 4. Configure API Keys

**Step 4.1:** Copy the example environment file
Windows
copy .env.example .env

Mac/Linux
cp .env.example .env

text

**Step 4.2:** Get your API keys

- **SerpAPI Key:**
  1. Sign up at https://serpapi.com/
  2. Go to https://serpapi.com/manage-api-key
  3. Copy your API key (Free tier: 100 searches/month)

- **APILayer Key (Optional):**
  1. Sign up at https://apilayer.com/
  2. Subscribe to Resume Parser API
  3. Copy your API key

**Step 4.3:** Update `.env` file with your actual keys
SERPAPI_KEY=paste_your_actual_serpapi_key_here
APILAYER_KEY=paste_your_actual_apilayer_key_here

text

### 5. Run the Application

cd web
python app.py

text

The server will start at: [**http://localhost:5000**](http://localhost:5000)

## 📖 Usage

1. **Open your browser** and navigate to `http://localhost:5000`
2. **Upload your resume** (PDF format) via drag-and-drop or file picker
3. **Wait for analysis** - The AI will classify your resume and calculate experience
4. **View matched jobs** - Get 10 personalized job listings with:
   - Job title and company
   - Location and salary (when available)
   - Posted date
   - Direct apply links



## 🔑 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SERPAPI_KEY` | SerpAPI key for Google Jobs scraping | Yes |
| `APILAYER_KEY` | APILayer Resume Parser API key | Optional |

## 🤖 How It Works

1. **Resume Upload** → User uploads PDF resume
2. **Text Extraction** → PyPDF2 extracts text from PDF
3. **ML Classification** → Trained model predicts job category (Software Engineer, Data Scientist, etc.)
4. **Experience Calculation** → Parses work history durations and calculates total years
5. **Query Building** → Creates experience-specific search query (e.g., "Software Engineer 2 years experience")
6. **Job Scraping** → SerpAPI fetches 10 latest jobs from Google Jobs
7. **Results Display** → Beautiful cards with job details and apply buttons

## 🎨 Features in Detail

### Smart Experience Matching
- **0 years** → Searches for "entry level" positions
- **1 year** → "0-1 year experience" jobs
- **2+ years** → Rounded up (1.4 years → 2 years, 8.9 years → 9 years)

### Job Information Displayed
- Job title
- Company name
- Location
- Salary range (when available)
- Posted time (e.g., "2 hours ago")
- Job type (Full-time, Part-time, Contract)
- Via source (LinkedIn, Indeed, Company site)
- Direct apply link

## 📊 API Usage Limits

**SerpAPI Free Tier:**
- 100 searches per month
- 1 search = 1 resume processed
- Monitor usage: https://serpapi.com/account

**APILayer Free Tier:**
- 100 requests per month

## 🛡️ Security

- ✅ API keys stored in `.env` (never committed to Git)
- ✅ Uploaded resumes deleted after processing
- ✅ No data stored on server
- ✅ Subprocess runs in sandboxed mode

## 🐛 Troubleshooting

**Issue:** "SERPAPI_KEY not found in environment variables"
- **Solution:** Make sure `.env` file exists and contains `SERPAPI_KEY=your_key`

**Issue:** "Failed to extract category from resume"
- **Solution:** Ensure resume is in PDF format and contains clear work experience/skills sections

**Issue:** "No jobs found"
- **Solution:** Check SerpAPI quota or try a more common job category

**Issue:** Upload fails
- **Solution:** Ensure file is PDF and under 16MB

## 📝 To-Do / Future Enhancements

- [ ] Email job alerts
- [ ] Save favorite jobs
- [ ] Support DOCX resumes
- [ ] Skill-based job ranking
- [ ] Multi-location search
- [ ] Job application tracking
- [ ] Mobile app version

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- [SerpAPI](https://serpapi.com/) for Google Jobs API
- [APILayer](https://apilayer.com/) for Resume Parser API
- [Flask](https://flask.palletsprojects.com/) for the web framework
- [scikit-learn](https://scikit-learn.org/) for ML capabilities

## 📞 Support

If you have any questions or issues, please open an issue on GitHub or contact me directly.

---

⭐ **Star this repo if you find it helpful!** ⭐
