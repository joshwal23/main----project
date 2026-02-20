const canvas = document.getElementById('raceCanvas');
const ctx = canvas.getContext('2d');

let animationId;
let isRacing = false;

// Physics & State
const CAR_WIDTH = 14;
const CAR_LENGTH = 30;
const MAX_SPEED = 18;
const NITRO_SPEED = 24;

let car = {
    x: 400,
    y: 300,
    angle: -Math.PI / 2, // Facing up
    speed: 0,
    acceleration: 0.2,
    braking: 0.4,
    friction: 0.96,
    turnSpeed: 0.05,
    nitro: 100,
    laps: 0,
    timeMs: 0
};

// Controls
const keys = {
    ArrowUp: false,
    ArrowDown: false,
    ArrowLeft: false,
    ArrowRight: false,
    Space: false
};

// Track / Environment
const trackLine = 200; // Radius of simple circular track

window.addEventListener('keydown', (e) => {
    if (keys.hasOwnProperty(e.code) || keys.hasOwnProperty(e.key)) {
        if (e.code === 'Space') keys.Space = true;
        else keys[e.key] = true;
        e.preventDefault();
    }
});

window.addEventListener('keyup', (e) => {
    if (keys.hasOwnProperty(e.code) || keys.hasOwnProperty(e.key)) {
        if (e.code === 'Space') keys.Space = false;
        else keys[e.key] = false;
    }
});

function initRacingGame() {
    if (isRacing) return;

    // Resize canvas
    const rect = canvas.parentElement.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;

    // Reset State
    car.x = canvas.width / 2;
    car.y = canvas.height - 100;
    car.angle = -Math.PI / 2;
    car.speed = 0;
    car.timeMs = 0;

    isRacing = true;
    lastTime = performance.now();
    gameLoop(lastTime);
}

function stopRacingGame() {
    isRacing = false;
    cancelAnimationFrame(animationId);
}

let lastTime = 0;

function updatePhysics(dt) {
    // Determine max speed based on Nitro
    let currentMax = keys.Space && car.nitro > 0 ? NITRO_SPEED : MAX_SPEED;

    // Handling (turn harder at high speeds, but with drift limit)
    let turnFactor = (car.speed / currentMax);
    if (turnFactor > 1) turnFactor = 1;

    // Turning
    if (Math.abs(car.speed) > 0.5) {
        if (keys.ArrowLeft) car.angle -= car.turnSpeed * turnFactor;
        if (keys.ArrowRight) car.angle += car.turnSpeed * turnFactor;
    }

    // Acceleration / Braking
    if (keys.ArrowUp) {
        car.speed += car.acceleration;
        if (keys.Space && car.nitro > 0) {
            car.nitro -= 0.5; // Consume nitro
        }
    } else if (keys.ArrowDown) {
        car.speed -= car.braking;
    }

    // Throttle / Friction
    if (car.speed > currentMax) car.speed = currentMax;
    if (car.speed < -MAX_SPEED / 2) car.speed = -MAX_SPEED / 2;

    if (!keys.ArrowUp && !keys.ArrowDown) {
        car.speed *= car.friction;
    }

    // Apply Velocity
    car.x += Math.cos(car.angle) * car.speed;
    car.y += Math.sin(car.angle) * car.speed;

    // Screen wrap / Bounds
    if (car.x < 0) car.x = canvas.width;
    if (car.x > canvas.width) car.x = 0;
    if (car.y < 0) car.y = canvas.height;
    if (car.y > canvas.height) car.y = 0;

    // Update UI
    car.timeMs += dt;

    const uiSpeed = document.getElementById('game-speed');
    const uiTime = document.getElementById('game-time');

    if (uiSpeed) uiSpeed.innerHTML = `${Math.round(Math.abs(car.speed) * 8)} <span class="text-sm text-[#ff2a2a] pl-1 font-bold">MPH</span>`;

    if (uiTime) {
        let sec = Math.floor(car.timeMs / 1000);
        let ms = Math.floor((car.timeMs % 1000) / 10);
        uiTime.textContent = `${sec.toString().padStart(2, '0')}:${ms.toString().padStart(2, '0')}`;
    }
}

function render() {
    // Sky/Ground gradient
    ctx.fillStyle = "#111";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    // Draw Grid Floor (Synthwave Aesthetic)
    ctx.strokeStyle = "rgba(69, 162, 158, 0.2)";
    ctx.lineWidth = 1;
    const gridCols = 20;
    const gridRows = 15;
    for (let i = 0; i < gridCols; i++) {
        ctx.beginPath();
        ctx.moveTo((canvas.width / gridCols) * i, 0);
        ctx.lineTo((canvas.width / gridCols) * i, canvas.height);
        ctx.stroke();
    }
    for (let j = 0; j < gridRows; j++) {
        ctx.beginPath();
        ctx.moveTo(0, (canvas.height / gridRows) * j);
        ctx.lineTo(canvas.width, (canvas.height / gridRows) * j);
        ctx.stroke();
    }

    // Draw Checkpoint / Track curve (Simplified visual track)
    ctx.beginPath();
    ctx.arc(canvas.width / 2, canvas.height / 2, Math.min(canvas.width, canvas.height) * 0.4, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.1)";
    ctx.lineWidth = 40;
    ctx.stroke();

    // Inner Neon Line
    ctx.beginPath();
    ctx.arc(canvas.width / 2, canvas.height / 2, Math.min(canvas.width, canvas.height) * 0.4 - 20, 0, Math.PI * 2);
    ctx.strokeStyle = "#ff2a2a";
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw Particles / Trail
    if (Math.abs(car.speed) > 10) {
        ctx.fillStyle = keys.Space ? "#45A29E" : "#ff2a2a";
        ctx.globalAlpha = Math.random() * 0.5 + 0.2;
        ctx.fillRect(
            car.x - Math.cos(car.angle) * CAR_LENGTH * 0.8 + (Math.random() * 4 - 2),
            car.y - Math.sin(car.angle) * CAR_LENGTH * 0.8 + (Math.random() * 4 - 2),
            4, 4
        );
        ctx.globalAlpha = 1.0;
    }

    // Draw Car
    ctx.save();
    ctx.translate(car.x, car.y);
    ctx.rotate(car.angle);

    // Shadow
    ctx.shadowColor = "rgba(255, 42, 42, 0.6)";
    ctx.shadowBlur = keys.Space ? 20 : 10;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;

    // Body
    ctx.fillStyle = "#222";
    ctx.fillRect(-CAR_LENGTH / 2, -CAR_WIDTH / 2, CAR_LENGTH, CAR_WIDTH);

    // Neon Accents
    ctx.strokeStyle = keys.Space ? "#45A29E" : "#ff2a2a";
    ctx.lineWidth = 2;
    ctx.strokeRect(-CAR_LENGTH / 2, -CAR_WIDTH / 2, CAR_LENGTH, CAR_WIDTH);

    // Windshield
    ctx.fillStyle = "#000";
    ctx.fillRect(CAR_LENGTH / 6, -CAR_WIDTH / 2 + 2, CAR_LENGTH / 4, CAR_WIDTH - 4);

    // Headlights
    ctx.fillStyle = "white";
    ctx.shadowColor = "white";
    ctx.shadowBlur = 10;
    ctx.fillRect(CAR_LENGTH / 2 - 2, -CAR_WIDTH / 2 + 1, 2, 3);
    ctx.fillRect(CAR_LENGTH / 2 - 2, CAR_WIDTH / 2 - 4, 2, 3);

    ctx.restore();
}

function gameLoop(timestamp) {
    if (!isRacing) return;

    let dt = timestamp - lastTime;
    lastTime = timestamp;

    updatePhysics(dt);
    render();

    animationId = requestAnimationFrame(gameLoop);
}
