/**
 * DeepFake Detector — Frontend JavaScript
 * Handles: file drag-drop, upload API call, loading steps, result rendering
 */

/* ── Upload Page ──────────────────────────────────────────────────────────── */

(function initUploadPage() {
  const dropZone   = document.getElementById('dropZone');
  const fileInput  = document.getElementById('fileInput');
  const fileInfo   = document.getElementById('fileInfo');
  const fileName   = document.getElementById('fileName');
  const fileSize   = document.getElementById('fileSize');
  const removeBtn  = document.getElementById('removeFile');
  const uploadBtn  = document.getElementById('uploadBtn');
  const uploadText = document.getElementById('uploadBtnText');
  const errorEl    = document.getElementById('uploadError');
  const overlay    = document.getElementById('loadingOverlay');
  const stepEl     = document.getElementById('loadingStep');
  const barEl      = document.getElementById('loadingBar');

  if (!dropZone) return; // Not on upload page

  let selectedFile = null;

  // ── File selection helpers ──────────────────────────────────────────────

  function formatBytes(bytes) {
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  }

  function showFile(file) {
    selectedFile = file;
    fileName.textContent = file.name;
    fileSize.textContent = formatBytes(file.size);
    fileInfo.style.display = 'block';
    uploadBtn.disabled = false;
    uploadText.textContent = 'Analyze Video';
    errorEl.textContent = '';
  }

  function clearFile() {
    selectedFile = null;
    fileInput.value = '';
    fileInfo.style.display = 'none';
    uploadBtn.disabled = true;
    uploadText.textContent = 'Select a video first';
  }

  // ── Drag & Drop ─────────────────────────────────────────────────────────

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelection(file);
  });

  dropZone.addEventListener('click', (e) => {
    if (e.target.tagName !== 'LABEL') fileInput.click();
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFileSelection(fileInput.files[0]);
  });

  removeBtn && removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFile();
  });

  function handleFileSelection(file) {
    const allowed = ['video/mp4', 'video/avi', 'video/quicktime',
                     'video/x-msvideo', 'video/x-matroska'];
    const ext = file.name.split('.').pop().toLowerCase();
    const allowedExt = ['mp4', 'avi', 'mov'];

    if (!allowedExt.includes(ext)) {
      errorEl.textContent = '⚠ Invalid file type. Please select .mp4, .avi, or .mov';
      return;
    }
    showFile(file);
  }

  // ── Upload & Analyze ────────────────────────────────────────────────────

  const STEPS = [
    { text: 'Uploading video...', pct: 20 },
    { text: 'Extracting frames...', pct: 45 },
    { text: 'Running CNN analysis...', pct: 70 },
    { text: 'Aggregating predictions...', pct: 88 },
    { text: 'Finalizing result...', pct: 97 },
  ];

  uploadBtn.addEventListener('click', async () => {
    if (!selectedFile) return;

    // Show loading overlay
    overlay.style.display = 'flex';
    uploadBtn.disabled = true;
    errorEl.textContent = '';

    // Animate loading steps
    let stepIdx = 0;
    const stepInterval = setInterval(() => {
      if (stepIdx < STEPS.length) {
        stepEl.textContent = STEPS[stepIdx].text;
        barEl.style.width = STEPS[stepIdx].pct + '%';
        stepIdx++;
      }
    }, 900);

    try {
      const formData = new FormData();
      formData.append('video', selectedFile);

      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      clearInterval(stepInterval);
      barEl.style.width = '100%';
      stepEl.textContent = 'Done!';

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Server error. Please try again.');
      }

      // Store result in sessionStorage and navigate to result page
      sessionStorage.setItem('deepfake_result', JSON.stringify(data));
      setTimeout(() => { window.location.href = '/result'; }, 400);

    } catch (err) {
      clearInterval(stepInterval);
      overlay.style.display = 'none';
      uploadBtn.disabled = false;
      errorEl.textContent = '⚠ ' + (err.message || 'Upload failed. Please try again.');
    }
  });
})();


/* ── Result Page ──────────────────────────────────────────────────────────── */

(function initResultPage() {
  const verdictBadge  = document.getElementById('verdictBadge');
  const verdictIcon   = document.getElementById('verdictIcon');
  const verdictLabel  = document.getElementById('verdictLabel');
  const resultTitle   = document.getElementById('resultTitle');
  const confidencePct = document.getElementById('confidencePct');
  const confidenceBar = document.getElementById('confidenceBar');
  const videoEl       = document.getElementById('videoPreview');
  const explanationEl = document.getElementById('resultExplanation');

  if (!verdictBadge) return; // Not on result page

  // Retrieve data from sessionStorage
  const raw = sessionStorage.getItem('deepfake_result');
  if (!raw) {
    resultTitle.textContent = 'No result found.';
    explanationEl.innerHTML = '<p>Please <a href="/upload">upload a video</a> first.</p>';
    return;
  }

  const data = JSON.parse(raw);
  const { label, confidence, video_url } = data;
  const isFake = label === 'FAKE';

  // ── Populate UI ─────────────────────────────────────────────────────────

  verdictBadge.classList.add(isFake ? 'fake' : 'real');
  verdictIcon.textContent  = isFake ? '⚠' : '✔';
  verdictLabel.textContent = isFake ? 'Deepfake Detected' : 'Authentic Video';

  resultTitle.textContent = isFake
    ? 'This video appears to be FAKE'
    : 'This video appears to be REAL';
  resultTitle.style.color = isFake ? 'var(--danger)' : 'var(--success)';

  // Animate confidence bar after short delay
  setTimeout(() => {
    confidencePct.textContent = confidence.toFixed(1) + '%';
    confidenceBar.style.width = confidence + '%';
    confidenceBar.classList.add(isFake ? 'fake' : 'real');
  }, 300);

  // Video preview
  if (videoEl && video_url) {
    videoEl.querySelector('source').src = video_url;
    videoEl.load();
  }

  // Explanation text
  if (explanationEl) {
    explanationEl.innerHTML = isFake
      ? `<strong>What this means:</strong> Our CNN model analyzed the video frames and found visual
         artifacts consistent with synthetic face generation or manipulation. The model detected
         anomalies such as unnatural blending boundaries, lighting inconsistencies, or temporal
         flickering that are typical signatures of deepfake generation pipelines.
         <br><br><strong>Confidence:</strong> ${confidence.toFixed(1)}% that this video is artificially generated.`
      : `<strong>What this means:</strong> Our CNN model analyzed the video frames and found no
         significant artifacts or anomalies associated with deepfake generation. The facial features,
         lighting, and temporal consistency all appear natural and unmanipulated.
         <br><br><strong>Confidence:</strong> ${confidence.toFixed(1)}% that this video is genuine.`;
  }
})();
