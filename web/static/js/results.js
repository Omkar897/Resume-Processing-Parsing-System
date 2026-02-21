// Get results from sessionStorage
const resultsData = JSON.parse(sessionStorage.getItem('jobResults'));

if (!resultsData) {
    window.location.href = '/';
}

function escapeHtml(text) {
    if (text == null) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Display resume info
const resumeInfo = document.getElementById('resumeInfo');
const ragBadge = resultsData.rag_enabled ? 'RAG: ON' : 'RAG: OFF';
const claudeBadge = resultsData.claude_enabled ? 'Analyzer: ON' : 'Analyzer: OFF';
resumeInfo.innerHTML = `
    <div class="info-badge">Category: ${escapeHtml(resultsData.category || 'Unknown')}</div>
    <div class="info-badge">Experience: ${escapeHtml(resultsData.experience)} years</div>
    <div class="info-badge">Jobs: ${escapeHtml(resultsData.total_jobs)}</div>
    <div class="info-badge">${ragBadge}</div>
    <div class="info-badge">${claudeBadge}</div>
`;

function renderResumeInsights(analysis) {
    const section = document.getElementById('resumeInsightsSection');
    const content = document.getElementById('resumeInsightsContent');
    let html = '';
    if (analysis.ats_score != null) {
        html += `<div class="insight-block"><strong>ATS score</strong>: ${analysis.ats_score}/100</div>`;
        if (analysis.ats_explanation) {
            html += `<p class="insight-text">${escapeHtml(analysis.ats_explanation)}</p>`;
        }
    }
    if (analysis.strengths?.length) {
        html += `<div class="insight-block"><strong>Strengths</strong><ul>${analysis.strengths.map(s => `<li>${escapeHtml(s)}</li>`).join('')}</ul></div>`;
    }
    if (analysis.missing_keywords?.length) {
        html += `<div class="insight-block"><strong>Missing keywords</strong><ul>${analysis.missing_keywords.map(k => `<li>${escapeHtml(k)}</li>`).join('')}</ul></div>`;
    }
    if (analysis.suggestions?.length) {
        html += `<div class="insight-block"><strong>Improvement tips</strong><ul>${analysis.suggestions.map(s => `<li>${escapeHtml(s)}</li>`).join('')}</ul></div>`;
    }
    content.innerHTML = html;
    section.style.display = 'block';
}

// Display resume insights if present
const resumeAnalysis = resultsData.resume_analysis || {};
const hasResumeAnalysis = resumeAnalysis.strengths?.length || resumeAnalysis.suggestions?.length ||
    resumeAnalysis.missing_keywords?.length || resumeAnalysis.ats_score != null;
if (hasResumeAnalysis) {
    renderResumeInsights(resumeAnalysis);
}

// On-demand analysis button
const analyzeBtn = document.getElementById('analyzeResumeBtn');
const analyzeStatus = document.getElementById('analyzeStatus');
const resumeExtracted = resultsData.resume_extracted_data || {};
const hasExtracted = (resumeExtracted.skills && resumeExtracted.skills.length) ||
    (resumeExtracted.experience && resumeExtracted.experience.length) ||
    (resumeExtracted.education && resumeExtracted.education.length);

if (analyzeBtn && resultsData.claude_enabled && !hasResumeAnalysis) {
    analyzeBtn.style.display = 'inline-block';
}

if (analyzeBtn) {
    analyzeBtn.addEventListener('click', async () => {
        analyzeStatus.textContent = '';
        if (!hasExtracted) {
            analyzeStatus.textContent = 'Resume text was not extracted for this run. Re-upload your resume to get analysis.';
            return;
        }
        analyzeBtn.disabled = true;
        analyzeBtn.textContent = 'Analyzing...';
        analyzeStatus.textContent = 'Running resume analysis...';

        try {
            const response = await fetch('/analyze-resume', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    category: resultsData.category,
                    extracted_data: resumeExtracted
                })
            });
            const data = await response.json();
            if (!data.success) {
                analyzeStatus.textContent = `Analysis failed: ${data.error || 'Unknown error'}`;
                return;
            }
            resultsData.resume_analysis = data.resume_analysis;
            sessionStorage.setItem('jobResults', JSON.stringify(resultsData));
            renderResumeInsights(data.resume_analysis || {});
            analyzeStatus.textContent = 'Done. Scroll up to see resume insights.';
            document.getElementById('resumeInsightsSection')?.scrollIntoView({ behavior: 'smooth' });
            analyzeBtn.style.display = 'none';
        } catch (e) {
            analyzeStatus.textContent = `Analysis error: ${e.message || e}`;
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.textContent = 'Analyze Resume (AI)';
        }
    });
}

// Display jobs
const jobsContainer = document.getElementById('jobsContainer');

resultsData.jobs.forEach((job, index) => {
    const jobCard = document.createElement('div');
    jobCard.className = 'job-card interactive-surface';
    jobCard.style.animationDelay = `${index * 0.08}s`;
    const matchScore = job.match_score != null ? job.match_score : null;
    const matchExplanation = job.match_explanation || '';

    jobCard.innerHTML = `
        <div class="job-header">
            <div>
                <h3 class="job-title">${escapeHtml(job.title || '')}</h3>
                <div class="job-company">${escapeHtml(job.company || '')}</div>
                ${matchScore != null ? `<div class="job-match"><span class="match-badge">${matchScore}% match</span>${matchExplanation ? `<p class="match-explanation">${escapeHtml(matchExplanation)}</p>` : ''}</div>` : ''}
            </div>
        </div>

        <div class="job-meta">
            ${job.location ? `<span class="meta-item">Location: ${escapeHtml(job.location)}</span>` : ''}
            ${job.posted_at ? `<span class="meta-item">Posted: ${escapeHtml(job.posted_at)}</span>` : ''}
            ${job.schedule_type ? `<span class="meta-item">Type: ${escapeHtml(job.schedule_type)}</span>` : ''}
            ${job.salary ? `<span class="meta-item">Salary: ${escapeHtml(job.salary)}</span>` : ''}
            ${job.via ? `<span class="meta-item">Source: ${escapeHtml(job.via)}</span>` : ''}
        </div>

        ${job.description ? `<p class="job-description">${escapeHtml(job.description)}</p>` : ''}

        <a href="${job.apply_link || '#'}" target="_blank" class="btn-apply">
            Open Application
        </a>
    `;

    jobsContainer.appendChild(jobCard);
});

// Email functionality
const sendEmailBtn = document.getElementById('sendEmailBtn');
const emailInput = document.getElementById('emailInput');
const emailStatus = document.getElementById('emailStatus');

sendEmailBtn.addEventListener('click', async () => {
    const email = emailInput.value.trim();

    if (!email) {
        showEmailStatus('Please enter your email address', 'error');
        return;
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
        showEmailStatus('Please enter a valid email address', 'error');
        return;
    }

    sendEmailBtn.disabled = true;
    sendEmailBtn.textContent = 'Sending...';
    showEmailStatus('Sending email...', 'sending');

    try {
        const response = await fetch('/send-email', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                email: email,
                jobs: resultsData.jobs,
                category: resultsData.category,
                experience: resultsData.experience
            })
        });

        const data = await response.json();

        if (data.success) {
            showEmailStatus('Email sent successfully. Check your inbox.', 'success');
            emailInput.value = '';
        } else {
            showEmailStatus('Failed to send email: ' + data.error, 'error');
        }

    } catch (error) {
        showEmailStatus('Error sending email. Please try again.', 'error');
        console.error('Email error:', error);
    } finally {
        sendEmailBtn.disabled = false;
        sendEmailBtn.textContent = 'Send';
    }
});

function showEmailStatus(message, type) {
    emailStatus.textContent = message;
    emailStatus.className = `email-status ${type}`;
}

// Allow Enter key to send email
emailInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendEmailBtn.click();
    }
});
