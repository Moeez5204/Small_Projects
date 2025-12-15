-- FITNESS TRACKER DATABASE SCHEMA
-- This file creates all the tables for the fitness tracker

-- USERS TABLE - Stores user account information
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    username TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    weight REAL,
    height REAL,
    goal TEXT
);

-- WORKOUTS TABLE - Stores workout session data
CREATE TABLE IF NOT EXISTS workouts (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    type TEXT,
    duration INTEGER,
    calories INTEGER,
    entry_number INTEGER,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- NUTRITION TABLE - Stores meal and food intake data
CREATE TABLE IF NOT EXISTS nutrition (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    meal TEXT,
    food TEXT,
    calories INTEGER,
    entry_number INTEGER,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- WEIGHT_LOG TABLE - Stores weight tracking data
CREATE TABLE IF NOT EXISTS weight_log (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    weight REAL,
    entry_number INTEGER,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_workouts_user ON workouts(user_id);
CREATE INDEX IF NOT EXISTS idx_nutrition_user ON nutrition(user_id);
CREATE INDEX IF NOT EXISTS idx_weight_log_user ON weight_log(user_id);