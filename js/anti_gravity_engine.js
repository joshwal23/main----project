class AntiGravityEngine {
    constructor() {
        this.reset();
    }

    reset() {
        this.state = "GRAVITY_LOCK";

        // Metrics
        this.totalAttempts = 0;
        this.successfulLifts = 0;
        this.failedLifts = 0;

        // Quality variables
        this.currentRepMinKneeAngle = 180;
        this.angleBuffer = [];
        this.smoothnessPenalty = 0;
        this.lastAngle = null;

        // Global Scores
        this.totalEnergy = 0;
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
        const rawHipAngle = this._calculateAngle(shoulder, hip, knee);

        // Anti-Jitter Smoothing (Moving average over 5 frames)
        this.angleBuffer.push({ knee: rawKneeAngle, hip: rawHipAngle });
        if (this.angleBuffer.length > 5) this.angleBuffer.shift();

        const smoothedKneeAngle = this.angleBuffer.reduce((sum, val) => sum + val.knee, 0) / this.angleBuffer.length;
        const smoothedHipAngle = this.angleBuffer.reduce((sum, val) => sum + val.hip, 0) / this.angleBuffer.length;

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

        // Anti-Gravity State Machine
        this._updateState(smoothedKneeAngle, smoothedHipAngle);

        return this.getResults();
    }

    _updateState(kneeAngle, hipAngle) {
        switch (this.state) {
            case "GRAVITY_LOCK":
                // standing position, knee angle > 160°
                if (kneeAngle < 160) {
                    this.state = "DESCENT_PHASE";
                    this.currentRepMinKneeAngle = kneeAngle;
                    this.smoothnessPenalty = 0;
                    this.totalAttempts++;
                }
                break;

            case "DESCENT_PHASE":
                // controlled descent, angle decreasing
                this.currentRepMinKneeAngle = Math.min(this.currentRepMinKneeAngle, kneeAngle);

                if (kneeAngle < 90) {
                    this.state = "ZERO_GRAVITY_ZONE";
                }
                // Mark as FAILED LIFT if Depth not reached (<90° not achieved) and ascending back to 160
                else if (kneeAngle >= 160) {
                    this.failedLifts++;
                    this.state = "GRAVITY_LOCK";
                }
                break;

            case "ZERO_GRAVITY_ZONE":
                // bottom position, knee angle < 90°
                this.currentRepMinKneeAngle = Math.min(this.currentRepMinKneeAngle, kneeAngle);
                if (kneeAngle >= 90) {
                    this.state = "LIFT_OFF";
                }
                break;

            case "LIFT_OFF":
                // ascending phase, angle increasing back > 160°
                if (kneeAngle >= 160) {
                    // Check smoothness parameter (avoids counting if movement was highly erratic/jittery spikes)
                    if (this.smoothnessPenalty < 100) {
                        this.successfulLifts++; // 1 Anti-Gravity Rep
                        this._calculateEnergyScore(this.currentRepMinKneeAngle, this.smoothnessPenalty);
                    } else {
                        // Failed due to bad movement quality
                        this.failedLifts++;
                    }
                    this.state = "GRAVITY_LOCK";
                }
                // Mark as FAILED LIFT if no full extension on return (fell back down)
                else if (kneeAngle < 90) {
                    this.failedLifts++;
                    this.state = "ZERO_GRAVITY_ZONE";
                }
                break;
        }
    }

    _calculateEnergyScore(depth, jitter) {
        // Base score for hitting depth
        let score = 100;

        // Bonus for going deeper than 90
        if (depth < 80) score += 20;
        if (depth < 70) score += 30;

        // Deduction for lack of control/jitter
        score -= (jitter * 0.5);

        this.totalEnergy += Math.max(10, Math.round(score));
    }

    getResultsAsJSON() {
        return JSON.stringify(this.getResults(), null, 2);
    }

    getResults() {
        // Calculate Gravity Resistance Rating (0-100)
        let rating = 0;
        if (this.totalAttempts > 0) {
            const successRate = this.successfulLifts / this.totalAttempts;
            // 80% weight on success rate, 20% on energy score execution
            rating = Math.max(0, Math.min(100, Math.round((successRate * 80) + (this.totalEnergy / (this.totalAttempts * 100) * 20))));
        }

        return {
            totalLiftAttempts: this.totalAttempts,
            successfulAntiGravityLifts: this.successfulLifts,
            failedLifts: this.failedLifts,
            energyScore: this.totalEnergy,
            gravityResistanceRating: rating,
            currentState: this.state
        };
    }
}

// Ensure it can be imported or used globally
if (typeof module !== 'undefined' && typeof module.exports !== 'undefined') {
    module.exports = AntiGravityEngine;
} else {
    window.AntiGravityEngine = AntiGravityEngine;
}
