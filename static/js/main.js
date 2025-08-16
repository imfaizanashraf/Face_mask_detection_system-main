// DOM Elements
const imageUpload = document.getElementById('imageUpload');
const imagePreview = document.getElementById('imagePreview');
const analyzeImageButton = document.getElementById('analyzeImage');
const painLevel = document.getElementById('pain-level');
const confidence = document.getElementById('confidence');
const liveCameraButton = document.getElementById('liveCameraButton');
const previewContainer = document.getElementById('preview-container');

let videoStream = null;
let isCameraActive = false;

// Image Upload Handling
imageUpload.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            // Stop camera if active
            if (isCameraActive) {
                stopCamera();
            }
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            analyzeImageButton.disabled = false;
        };
        reader.readAsDataURL(file);
    }
});

// Live Camera Button Handling
liveCameraButton.addEventListener('click', async function() {
    if (!isCameraActive) {
        try {
            videoStream = await navigator.mediaDevices.getUserMedia({ video: true });
            isCameraActive = true;
            
            // Clear any existing preview
            imagePreview.style.display = 'none';
            
            // Create and show video element
            const videoElement = document.createElement('video');
            videoElement.id = 'cameraPreview';
            videoElement.srcObject = videoStream;
            videoElement.autoplay = true;
            videoElement.style.width = '100%';
            videoElement.style.height = '100%';
            videoElement.style.objectFit = 'cover';
            
            previewContainer.innerHTML = '';
            previewContainer.appendChild(videoElement);
            
            liveCameraButton.innerHTML = '<i class="fas fa-camera"></i> Capture Image';
            analyzeImageButton.disabled = true;
        } catch (error) {
            console.error('Error accessing camera:', error);
            showError('Could not access camera');
        }
    } else {
        captureImage();
    }
});

function captureImage() {
    const videoElement = document.getElementById('cameraPreview');
    if (!videoElement) return;

    // Create canvas and capture image
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    canvas.getContext('2d').drawImage(videoElement, 0, 0);
    
    // Convert canvas to image
    const capturedImage = canvas.toDataURL('image/png');
    
    // Clear the preview container
    previewContainer.innerHTML = '';
    
    // Create and display the captured image
    imagePreview.src = capturedImage;
    imagePreview.style.display = 'block';
    previewContainer.appendChild(imagePreview);
    
    // Clean up camera
    stopCamera();
    
    // Enable analyze button
    analyzeImageButton.disabled = false;
}

function stopCamera() {
    if (videoStream) {
        videoStream.getTracks().forEach(track => track.stop());
        videoStream = null;
    }
    isCameraActive = false;
    const videoElement = document.getElementById('cameraPreview');
    if (videoElement) {
        videoElement.remove();
    }
    liveCameraButton.innerHTML = '<i class="fas fa-camera"></i> Live Camera';
}

// Analyze Image
analyzeImageButton.addEventListener('click', async function() {
    if (!imagePreview.src) return;

    try {
        analyzeImageButton.innerHTML = '<span class="loading"></span> Analyzing...';
        analyzeImageButton.disabled = true;

        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imagePreview.src })
        });

        if (response.ok) {
            const data = await response.json();
            updateResults(data);
            // Voice output is now handled by the backend using pyttsx3
        } else {
            const errorData = await response.json();
            showError(errorData.error || 'Error analyzing image');
        }
    } catch (error) {
        console.error('Error in analysis:', error);
        showError('Network error occurred');
    } finally {
        analyzeImageButton.innerHTML = 'Analyze Image';
        analyzeImageButton.disabled = false;
    }
});

// Update results display
function updateResults(data) {
    painLevel.textContent = `${data.pain_level}%`;
    confidence.textContent = `${data.confidence}%`;
    
    // Add visual feedback based on pain level
    const painLevelElement = document.getElementById('pain-level');
    if (data.pain_level > 70) {
        painLevelElement.style.color = '#e74c3c';
    } else if (data.pain_level > 30) {
        painLevelElement.style.color = '#f39c12';
    } else {
        painLevelElement.style.color = '#3498db';
    }

    // Update pain class if available
    if (data.pain_class) {
        const painClassElement = document.getElementById('pain-class');
        if (painClassElement) {
            painClassElement.textContent = data.pain_class;
        }
    }
}

// Show error message
function showError(message) {
    const errorDiv = document.createElement('div');
    errorDiv.className = 'error-message';
    errorDiv.textContent = message;
    document.body.appendChild(errorDiv);
    
    setTimeout(() => {
        errorDiv.remove();
    }, 3000);
}

// Smooth scrolling for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add event listeners for Services and Contact links
document.querySelector('a[href="#services"]').addEventListener('click', function(e) {
    e.preventDefault();
    const servicesSection = document.querySelector('#services');
    if (servicesSection) {
        servicesSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
});

document.querySelector('a[href="#contact"]').addEventListener('click', function(e) {
    e.preventDefault();
    const contactSection = document.querySelector('#contact');
    if (contactSection) {
        contactSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
});

// Clean up camera when leaving page
window.addEventListener('beforeunload', stopCamera); 