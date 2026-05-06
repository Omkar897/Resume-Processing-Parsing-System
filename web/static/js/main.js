const uploadBox = document.getElementById('uploadBox');
const resumeInput = document.getElementById('resumeInput');
const fileName = document.getElementById('fileName');
const uploadBtn = document.getElementById('uploadBtn');
const loadingSection = document.getElementById('loadingSection');
const loadingText = document.getElementById('loadingText');
const progressFill = document.getElementById('progressFill');

let selectedFile = null;

let progressValue = 0;
let targetProgress = 0;
let progressTicker = null;
let progressRaf = null;

// Drag and drop
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.classList.add('dragging');
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.classList.remove('dragging');
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.classList.remove('dragging');

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
    fileName.textContent = `Selected file: ${name}`;
    uploadBtn.style.display = 'block';
}

function updateLoadingText(progress) {
    if (progress < 25) {
        loadingText.textContent = 'Uploading resume...';
    } else if (progress < 50) {
        loadingText.textContent = 'Extracting and classifying profile...';
    } else if (progress < 75) {
        loadingText.textContent = 'Calculating semantic matches...';
    } else if (progress < 95) {
        loadingText.textContent = 'Finalizing recommendations...';
    } else {
        loadingText.textContent = 'Preparing results...';
    }
}

function runProgressFrame() {
    progressValue += (targetProgress - progressValue) * 0.1;

    if (Math.abs(targetProgress - progressValue) < 0.12) {
        progressValue = targetProgress;
    }

    progressFill.style.width = `${Math.max(0, Math.min(100, progressValue)).toFixed(2)}%`;
    updateLoadingText(progressValue);

    if (progressValue < 99.95 || targetProgress < 100) {
        progressRaf = requestAnimationFrame(runProgressFrame);
    } else {
        progressRaf = null;
    }
}

function startLoadingProgress() {
    progressValue = 0;
    targetProgress = 10;
    progressFill.style.width = '0%';
    updateLoadingText(0);

    if (progressRaf) {
        cancelAnimationFrame(progressRaf);
    }
    progressRaf = requestAnimationFrame(runProgressFrame);

    if (progressTicker) {
        clearInterval(progressTicker);
    }

    // Slowly move toward 88% while backend is still processing.
    progressTicker = setInterval(() => {
        if (targetProgress < 45) {
            targetProgress += 4 + Math.random() * 2;
        } else if (targetProgress < 72) {
            targetProgress += 2 + Math.random() * 1.4;
        } else if (targetProgress < 88) {
            targetProgress += 0.7 + Math.random() * 0.8;
        }
        targetProgress = Math.min(targetProgress, 88);
    }, 700);
}

function stopLoadingProgressTicker() {
    if (progressTicker) {
        clearInterval(progressTicker);
        progressTicker = null;
    }
}

function waitForProgressCompletion(timeoutMs = 2200) {
    return new Promise((resolve) => {
        const start = Date.now();

        function check() {
            if (progressValue >= 99.5 || Date.now() - start > timeoutMs) {
                resolve();
                return;
            }
            requestAnimationFrame(check);
        }

        check();
    });
}

// Upload button click
uploadBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Hide upload section
    document.querySelector('.upload-section').style.display = 'none';
    document.querySelector('.features').style.display = 'none';
    loadingSection.style.display = 'block';

    // Start controlled progress (never reaches 100 until response arrives)
    startLoadingProgress();

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
            stopLoadingProgressTicker();
            targetProgress = 100;
            await waitForProgressCompletion();

            // Store results in sessionStorage
            sessionStorage.setItem('jobResults', JSON.stringify(data));

            // Redirect to results page
            window.location.href = '/results';
        } else {
            alert('Error: ' + data.error);
            location.reload();
        }
    } catch (error) {
        alert('Upload failed: ' + error.message);
        location.reload();
    }
});
