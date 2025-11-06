const uploadBox = document.getElementById('uploadBox');
const resumeInput = document.getElementById('resumeInput');
const fileName = document.getElementById('fileName');
const uploadBtn = document.getElementById('uploadBtn');
const loadingSection = document.getElementById('loadingSection');
const loadingText = document.getElementById('loadingText');
const progressFill = document.getElementById('progressFill');

let selectedFile = null;

// Drag and drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#764ba2';
    uploadBox.style.background = '#f8f9ff';
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = 'white';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#667eea';
    uploadBox.style.background = 'white';
    
    const files = e.dataTransfer.files;
    if (files.length > 0 && files[0].type === 'application/pdf') {
        selectedFile = files[0];
        showFileName(files[0].name);
    }
});

// File input change
resumeInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        selectedFile = e.target.files[0];
        showFileName(e.target.files[0].name);
    }
});

function showFileName(name) {
    fileName.textContent = `📄 ${name}`;
    uploadBtn.style.display = 'block';
}

// Upload button click
uploadBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    // Hide upload section
    document.querySelector('.upload-section').style.display = 'none';
    document.querySelector('.features').style.display = 'none';
    loadingSection.style.display = 'block';
    
    // Simulate progress
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += 10;
        progressFill.style.width = progress + '%';
        
        if (progress === 30) loadingText.textContent = 'Classifying resume category...';
        if (progress === 60) loadingText.textContent = 'Calculating experience...';
        if (progress === 90) loadingText.textContent = 'Searching for matching jobs...';
        
        if (progress >= 100) clearInterval(progressInterval);
    }, 500);
    
    // Upload file
    const formData = new FormData();
    formData.append('resume', selectedFile);
    
    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            // Store results in sessionStorage
            sessionStorage.setItem('jobResults', JSON.stringify(data));
            
            // Redirect to results page
            setTimeout(() => {
                window.location.href = '/results';
            }, 1000);
        } else {
            alert('Error: ' + data.error);
            location.reload();
        }
    } catch (error) {
        alert('Upload failed: ' + error.message);
        location.reload();
    }
});
