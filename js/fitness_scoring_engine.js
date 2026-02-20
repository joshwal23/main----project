class FitnessScoringEngine {
    constructor() {
        this.reset();
    }

    reset() {
        this.state = "STANDING";

        // Metrics
        this.totalAttempts = 0;
        this.validReps = 0;
        this.almostReps = 0;
        this.miniReps = 0;
        this.invalidReps = 0;
        this.totalScore = 0;

        // Rep Details
        this.repDetails = [];

        // State variables
        this.currentRepMinKneeAngle = 180;
        this.angleBuffer = [];
        this.lastAngle = null;
        this.smoothnessPenalty = 0;

        // Time tracking (assuming processFrame is called at regular intervals, but we can track frame count or performance.now)
        this.startTime = performance.now();
    }

    // Helper: Calculate angle between 3 points (A, B, C with B as vertex)
    _calculateAngle(A, B, C) {
        if (!A || !B || !C) return 180;
        const ba = { x: A.x - B.x, y: A.y - B.y };
        const bc = { x: C.x - B.x, y: C.y - B.y };

        const dotProduct = (ba.x * bc.x) + (ba.y * bc.y);
        const magBA = Math.sqrt(ba.x * ba.x + ba.y * ba.y);
        const magBC = Math.sqrt(bc.x * bc.x + bc.y * bc.y);

        const cosine = Math.max(-1, Math.min(1, dotProduct / (magBA * magBC)));
        return (Math.acos(cosine) * 180.0) / Math.PI;
    }

    // Process a single frame of pose landmarks
    processFrame(landmarks) {
        // Track key joints: Shoulder(11), Hip(23), Knee(25), Ankle(27)
        const shoulder = landmarks[11];
        const hip = landmarks[23];
        const knee = landmarks[25];
        const ankle = landmarks[27];

        if (!shoulder || !hip || !knee || !ankle) return this.getResults();

        // Calculate continuous angles
        const rawKneeAngle = this._calculateAngle(hip, knee, ankle);

        // Anti-Jitter Smoothing (Moving average over 5 frames)
        this.angleBuffer.push(rawKneeAngle);
        if (this.angleBuffer.length > 5) this.angleBuffer.shift();

        const smoothedKneeAngle = this.angleBuffer.reduce((sum, val) => sum + val, 0) / this.angleBuffer.length;

        // Detect and ignore small jitter movements
        if (this.lastAngle !== null) {
            const delta = Math.abs(smoothedKneeAngle - this.lastAngle);
            if (delta > 15) { // Sudden spike threshold = jitter
                this.smoothnessPenalty += delta;
            } else if (delta < 2) {
                // Ignore small jitter entirely (do not process state change)
                return this.getResults();
            }
        }
        this.lastAngle = smoothedKneeAngle;

        // Scoring Engine State Machine
        this._updateState(smoothedKneeAngle);

        return this.getResults();
    }

    _updateState(kneeAngle) {
        switch (this.state) {
            case "STANDING":
                // standing position, knee angle > 160°
                if (kneeAngle < 160) {
                    this.state = "DESCENDING";
                    this.currentRepMinKneeAngle = kneeAngle;
                    this.smoothnessPenalty = 0;
                }
                break;

            case "DESCENDING":
                // angle decreasing
                this.currentRepMinKneeAngle = Math.min(this.currentRepMinKneeAngle, kneeAngle);

                // Assume bottom is reached when the angle starts increasing noticeably (by 5 degrees) 
                // or if it goes below 130 and starts coming up
                if (kneeAngle > this.currentRepMinKneeAngle + 5) {
                    this.state = "BOTTOM";
                }
                break;

            case "BOTTOM":
                // checking angle threshold and transitioning to ascending
                if (kneeAngle > this.currentRepMinKneeAngle + 15) {
                    this.state = "ASCENDING";
                } else {
                    // Still adjusting at the bottom
                    this.currentRepMinKneeAngle = Math.min(this.currentRepMinKneeAngle, kneeAngle);
                }
                break;

            case "ASCENDING":
                // angle increasing
                // Check if we reached the top (standing)
                if (kneeAngle >= 160) {
                    this.totalAttempts++;
                    this._evaluateRep();
                    this.state = "STANDING";
                }
                // Mark as aborted if they go back down significantly
                else if (kneeAngle < this.currentRepMinKneeAngle + 10) {
                    this.state = "BOTTOM";
                }
                break;
        }
    }

    _evaluateRep() {
        let category = "invalid";
        let scoreAwarded = 0;
        let minAngle = this.currentRepMinKneeAngle;

        // 1. FULL REP (100 points)
        if (minAngle <= 90) {
            if (this.smoothnessPenalty < 150) {
                category = "valid";
                scoreAwarded = 100;
                this.validReps++;
            } else {
                category = "invalid";
                this.invalidReps++;
            }
        }
        // 2. ALMOST REP (50-70 points)
        else if (minAngle >= 91 && minAngle <= 110) {
            if (this.smoothnessPenalty < 150) {
                category = "almost";
                // Scale score between 50 and 70 based on angle (closer to 90 is better, 91->70pts, 110->50pts)
                let pct = (110 - minAngle) / 19; // 19 is range between 110 and 91
                scoreAwarded = Math.round(50 + (20 * pct));
                this.almostReps++;
            } else {
                category = "invalid";
                this.invalidReps++;
            }
        }
        // 3. MINI REP (20-40 points)
        else if (minAngle >= 111 && minAngle <= 130) {
            category = "mini";
            // Scale score between 20 and 40 based on angle (111->40pts, 130->20pts)
            let pct = (130 - minAngle) / 19;
            scoreAwarded = Math.round(20 + (20 * pct));
            this.miniReps++;
        }
        // 4. INVALID REP (0 points)
        else {
            category = "invalid";
            scoreAwarded = 0;
            this.invalidReps++;
        }

        this.totalScore += scoreAwarded;

        // Log rep detail
        this.repDetails.push({
            rep_number: this.totalAttempts,
            category: category,
            min_knee_angle: parseFloat(minAngle.toFixed(2)),
            score_awarded: scoreAwarded,
            timestamp: parseFloat(((performance.now() - this.startTime) / 1000).toFixed(2))
        });
    }

    getResultsAsJSON() {
        return JSON.stringify(this.getResults(), null, 2);
    }

    getResults() {
        return {
            total_attempts: this.totalAttempts,
            valid_reps: this.validReps,
            almost_reps: this.almostReps,
            mini_reps: this.miniReps,
            invalid_reps: this.invalidReps,
            total_score: this.totalScore,
            rep_details: this.repDetails
        };
    }
}

// Ensure it can be imported or used globally
if (typeof module !== 'undefined' && typeof module.exports !== 'undefined') {
    module.exports = FitnessScoringEngine;
} else {
    window.FitnessScoringEngine = FitnessScoringEngine;
}
