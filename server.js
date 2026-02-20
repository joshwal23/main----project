// ===============================
// FitQuest Backend Server
// ===============================

const express = require("express");
const mysql = require("mysql2");
const cors = require("cors");

const app = express();
app.use(express.json());
app.use(cors());

// ===============================
// Database Connection
// ===============================

const db = mysql.createConnection({
    host: "localhost",
    user: "root",
    password: "yourpassword",
    database: "fitquest"
});

db.connect(err => {
    if (err) throw err;
    console.log("Database Connected");
});

// ===============================
// Save Workout Session
// ===============================

app.post("/save-session", (req, res) => {
    const { user_id, total_reps, valid_reps, invalid_reps, best_depth } = req.body;

    const sql = `
        INSERT INTO workout_sessions 
        (user_id, total_reps, valid_reps, invalid_reps, best_depth)
        VALUES (?, ?, ?, ?, ?)
    `;

    db.query(sql, [user_id, total_reps, valid_reps, invalid_reps, best_depth], 
    (err, result) => {
        if (err) return res.status(500).send(err);
        res.send("Session Saved Successfully");
    });
});

app.listen(5000, () => console.log("Server running on port 5000"));
