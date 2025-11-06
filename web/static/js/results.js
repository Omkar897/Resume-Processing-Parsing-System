// Get results from sessionStorage
const resultsData = JSON.parse(sessionStorage.getItem('jobResults'));

if (!resultsData) {
    window.location.href = '/';
}

// Display resume info
const resumeInfo = document.getElementById('resumeInfo');
resumeInfo.innerHTML = `
    <div class="info-badge">📊 ${resultsData.category}</div>
    <div class="info-badge">💼 ${resultsData.experience} years experience</div>
    <div class="info-badge">🎯 ${resultsData.total_jobs} jobs found</div>
`;

// Display jobs
const jobsContainer = document.getElementById('jobsContainer');

resultsData.jobs.forEach((job, index) => {
    const jobCard = document.createElement('div');
    jobCard.className = 'job-card';
    jobCard.style.animationDelay = `${index * 0.1}s`;
    
    jobCard.innerHTML = `
        <div class="job-header">
            <div>
                <h3 class="job-title">${job.title}</h3>
                <div class="job-company">🏢 ${job.company}</div>
            </div>
        </div>
        
        <div class="job-meta">
            ${job.location ? `<span class="meta-item">📍 ${job.location}</span>` : ''}
            ${job.posted_at ? `<span class="meta-item">⏰ ${job.posted_at}</span>` : ''}
            ${job.schedule_type ? `<span class="meta-item">💼 ${job.schedule_type}</span>` : ''}
            ${job.salary ? `<span class="meta-item">💰 ${job.salary}</span>` : ''}
            ${job.via ? `<span class="meta-item">📢 via ${job.via}</span>` : ''}
        </div>
        
        ${job.description ? `<p class="job-description">${job.description}</p>` : ''}
        
        <a href="${job.apply_link}" target="_blank" class="btn-apply">
            Apply Now →
        </a>
    `;
    
    jobsContainer.appendChild(jobCard);
});
