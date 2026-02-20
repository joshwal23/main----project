-- ===============================
-- FitQuest Database Schema
-- ===============================

-- Create Database
CREATE DATABASE fitquest;

USE fitquest;

-- ===============================
-- USERS TABLE
-- ===============================
CREATE TABLE users (
    user_id INT PRIMARY KEY AUTO_INCREMENT,
    username VARCHAR(50) NOT NULL UNIQUE,
    email VARCHAR(100) UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ===============================
-- WORKOUT SESSIONS TABLE
-- ===============================
CREATE TABLE workout_sessions (
    session_id INT PRIMARY KEY AUTO_INCREMENT,
    user_id INT,
    total_reps INT DEFAULT 0,
    valid_reps INT DEFAULT 0,
    invalid_reps INT DEFAULT 0,
    best_depth FLOAT,
    session_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- ===============================
-- REP DETAILS TABLE
-- ===============================
CREATE TABLE rep_details (
    rep_id INT PRIMARY KEY AUTO_INCREMENT,
    session_id INT,
    min_knee_angle FLOAT,
    is_valid BOOLEAN,
    rep_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (session_id) REFERENCES workout_sessions(session_id)
);
