console.log("Ultra-Strict Gamified Squat Logic Loaded");

const video = document.getElementById("webcam");
const canvas = document.getElementById("output");
const ctx = canvas.getContext("2d");

// Dashboard Elements
const repDisplay = document.getElementById("rep-count");
const scoreDisplay = document.getElementById("score");
const timerDisplay = document.getElementById("timer");
const angleDisplay = document.getElementById("live-angle");
const statusDisplay = document.getElementById("squat-status");
const feedbackDisplay = document.getElementById("feedback-msg");
const ball = document.getElementById("ball");
const endSessionBtn = document.getElementById("end-session-btn");

// Handle End Session
endSessionBtn.addEventListener('click', () => {
    // 1. Save Session Data to LocalStorage
    if (repCount > 0) {
        const session = {
            date: new Date().toLocaleString(),
            reps: repCount,
            score: totalScore,
            id: Date.now()
        };

        const history = JSON.parse(localStorage.getItem('fitquest_sessions') || '[]');
        history.unshift(session); // Add to beginning
        localStorage.setItem('fitquest_sessions', JSON.stringify(history));
    }

    // 2. Stop the camera cleanly
    if (camera) {
        camera.stop();
    }
    // 3. Redirect back to dashboard
    window.location.href = "dashboard.html";
});

video.muted = true;
video.playsInline = true;

// --- GAME & WORKOUT STATE ---
let repCount = 0;
let totalScore = 0;
let squatState = "UP";
let isRepValid = false;
let minAngleThisRep = 180;
let startTime = Date.now();

// Stability tools
let angleBuffer = [];
const BUFFER_SIZE = 5;

// MediaPipe Pose
const pose = new Pose({
    locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/pose@0.5/${file}`
});

pose.setOptions({
    modelComplexity: 1,
    smoothLandmarks: true,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
});

// Update Timer every second
setInterval(() => {
    const elapsed = Math.floor((Date.now() - startTime) / 1000);
    const mins = Math.floor(elapsed / 60).toString().padStart(2, '0');
    const secs = (elapsed % 60).toString().padStart(2, '0');
    timerDisplay.textContent = `${mins}:${secs}`;
}, 1000);

pose.onResults(results => {
    if (!results.poseLandmarks) {
        feedbackDisplay.textContent = "STEP INTO FRAME";
        feedbackDisplay.className = "warning";
        return;
    }

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const landmarks = results.poseLandmarks;
    const hip = landmarks[23];
    const knee = landmarks[25];
    const ankle = landmarks[27];

    // Visibility Check
    const visibilityThreshold = 0.6;
    if (hip.visibility < visibilityThreshold || knee.visibility < visibilityThreshold || ankle.visibility < visibilityThreshold) {
        feedbackDisplay.textContent = "STAND CLEARLY SIDEWAYS";
        feedbackDisplay.className = "warning";
        return;
    }

    // Smooth Angle Calculation (Dot Product)
    const rawAngle = calculateAngle(hip, knee, ankle);
    angleBuffer.push(rawAngle);
    if (angleBuffer.length > BUFFER_SIZE) angleBuffer.shift();
    const angle = angleBuffer.reduce((a, b) => a + b) / angleBuffer.length;

    angleDisplay.textContent = `${Math.round(angle)}°`;

    // Move the ball based on angle (Map 70-170 degrees to 0-95% bottom)
    const ballPos = Math.max(0, Math.min(95, ((angle - 70) / (170 - 70)) * 95));
    ball.style.bottom = `${ballPos}%`;

    // 3. DRAW SKELETON Visuals
    drawConnectors(ctx, results.poseLandmarks, POSE_CONNECTIONS, { color: "rgba(0, 255, 0, 0.2)", lineWidth: 4 });
    [hip, knee, ankle].forEach((joint, i) => {
        const px = joint.x * canvas.width;
        const py = joint.y * canvas.height;
        ctx.beginPath();
        ctx.arc(px, py, 12, 0, 2 * Math.PI);
        ctx.fillStyle = i === 1 ? (angle < 100 ? "#00FF00" : "yellow") : "#00FFFF";
        ctx.fill();
        ctx.strokeStyle = "white";
        ctx.lineWidth = 2;
        ctx.stroke();
    });

    // --- STATE MACHINE & SCORING ---

    // START/FINISH (UP Position)
    if (angle > 165 && squatState === "DOWN") {
        if (isRepValid) {
            repCount++;
            let points = 0;

            if (minAngleThisRep < 85) {
                points = 100;
                feedbackDisplay.textContent = "PERFECT! +100";
            } else {
                points = 50;
                feedbackDisplay.textContent = "GOOD! +50";
            }

            totalScore += points;
            repDisplay.textContent = repCount;
            scoreDisplay.textContent = totalScore;
            feedbackDisplay.className = "good-form";

            // Visual Pop
            scoreDisplay.style.transform = "scale(1.5)";
            setTimeout(() => scoreDisplay.style.transform = "scale(1)", 200);
        } else {
            feedbackDisplay.textContent = "NO REP: GO LOWER";
            feedbackDisplay.className = "bad-form";
        }

        squatState = "UP";
        statusDisplay.textContent = "UP";
        statusDisplay.className = "value state-up";
        feedbackDisplay.classList.add("pop-effect");
        setTimeout(() => feedbackDisplay.classList.remove("pop-effect"), 400);
    }

    // DESCENDING
    if (angle < 145 && squatState === "UP") {
        squatState = "DOWN";
        isRepValid = false;
        minAngleThisRep = 180;
        statusDisplay.textContent = "DOWN";
        statusDisplay.className = "value state-down";
    }

    // TRACKING DEPTH
    if (squatState === "DOWN") {
        if (angle < minAngleThisRep) minAngleThisRep = angle;
        if (minAngleThisRep <= 100) {
            isRepValid = true;
            feedbackDisplay.textContent = "DEPTH OK!";
            feedbackDisplay.className = "good-form";
        } else {
            feedbackDisplay.textContent = "GO LOWER!";
            feedbackDisplay.className = "warning";
        }
    }
});

function calculateAngle(A, B, C) {
    const ba = { x: A.x - B.x, y: A.y - B.y };
    const bc = { x: C.x - B.x, y: C.y - B.y };
    const dotProduct = (ba.x * bc.x) + (ba.y * bc.y);
    const magBA = Math.sqrt(ba.x * ba.x + ba.y * ba.y);
    const magBC = Math.sqrt(bc.x * bc.x + bc.y * bc.y);
    const cosine = Math.max(-1, Math.min(1, dotProduct / (magBA * magBC)));
    return (Math.acos(cosine) * 180.0) / Math.PI;
}

function onResize() {
    if (!video.videoWidth) return;
    const scale = Math.min(window.innerWidth / video.videoWidth, window.innerHeight / video.videoHeight);
    const w = video.videoWidth * scale;
    const h = video.videoHeight * scale;
    video.style.width = canvas.style.width = `${w}px`;
    video.style.height = canvas.style.height = `${h}px`;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
}

window.addEventListener('resize', onResize);
video.addEventListener('loadeddata', onResize);

// Set initial loading status
feedbackDisplay.textContent = "LOADING AI MODEL...";
feedbackDisplay.className = "warning";

const camera = new Camera(video, {
    onFrame: async () => {
        await pose.send({ image: video });
    },
    width: 640,
    height: 480
});

// Start Camera by Default
camera.start();

// Handle Video Upload
const uploadInput = document.getElementById("upload-video-trainer");
if (uploadInput) {
    uploadInput.addEventListener("change", function (event) {
        const file = event.target.files[0];
        if (!file) return;

        // Stop live webcam if it's running
        if (camera) {
            camera.stop();
        }

        // Reset state
        repCount = 0;
        totalScore = 0;
        squatState = "UP";
        minAngleThisRep = 180;
        angleBuffer = [];
        repDisplay.textContent = 0;
        scoreDisplay.textContent = 0;
        angleDisplay.textContent = "0°";
        startTime = Date.now();

        // Load video file
        video.srcObject = null;
        video.src = URL.createObjectURL(file);
        video.autoplay = true;
        video.muted = true;
        video.loop = false; // or true if you want it to loop

        video.addEventListener('loadeddata', () => {
            onResize();
            video.play();
        });

        // Loop feed into MediaPipe Pose via requestAnimationFrame
        video.onplay = () => {
            const loop = async () => {
                if (video.paused || video.ended) return;
                await pose.send({ image: video });
                requestAnimationFrame(loop);
            };
            loop();
        };
    });
}

