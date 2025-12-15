"""
DEMO SCRIPT FOR FITNESS TRACKER
Run this to generate sample data and create showcase graphs for GitHub
"""

import sqlite3
import hashlib
import random
from datetime import datetime, timedelta


def create_sample_database():
    """Create sample database with demo data"""
    print("=" * 60)
    print("CREATING SAMPLE DATABASE FOR FITNESS TRACKER")
    print("=" * 60)

    # Connect to database
    conn = sqlite3.connect('fitness.db')
    cursor = conn.cursor()

    # Create tables (simplified version)
    cursor.executescript("""
                         DROP TABLE IF EXISTS users;
                         DROP TABLE IF EXISTS workouts;
                         DROP TABLE IF EXISTS nutrition;
                         DROP TABLE IF EXISTS weight_log;

                         CREATE TABLE users
                         (
                             id       INTEGER PRIMARY KEY,
                             username TEXT UNIQUE,
                             password TEXT,
                             weight   REAL,
                             height   REAL,
                             goal     TEXT
                         );

                         CREATE TABLE workouts
                         (
                             id           INTEGER PRIMARY KEY,
                             user_id      INTEGER,
                             type         TEXT,
                             duration     INTEGER,
                             calories     INTEGER,
                             entry_number INTEGER,
                             FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                         );

                         CREATE TABLE nutrition
                         (
                             id           INTEGER PRIMARY KEY,
                             user_id      INTEGER,
                             meal         TEXT,
                             food         TEXT,
                             calories     INTEGER,
                             entry_number INTEGER,
                             FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                         );

                         CREATE TABLE weight_log
                         (
                             id           INTEGER PRIMARY KEY,
                             user_id      INTEGER,
                             weight       REAL,
                             entry_number INTEGER,
                             FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
                         );
                         """)

    # Create demo user
    demo_password = hashlib.sha256("demo123".encode()).hexdigest()
    cursor.execute(
        "INSERT INTO users (username, password, weight, height, goal) VALUES (?, ?, ?, ?, ?)",
        ("demo_user", demo_password, 75.0, 180, "Lose 5kg and build muscle")
    )
    user_id = cursor.lastrowid

    print("✅ Created demo user: demo_user (password: demo123)")
    print("   Goal: Lose 5kg and build muscle")
    print("   Starting weight: 75.0 kg, Height: 180 cm")

    # Sample workout types
    workout_types = ["Gym", "Running", "Cycling", "Swimming", "Yoga", "HIIT"]

    # Generate 20 workout entries (showing progress)
    print("\n💪 Generating workout data...")
    for i in range(1, 21):
        workout_type = random.choice(workout_types)

        # Create progression - duration increases over time
        base_duration = 30
        progression = i * 2  # Workouts get longer
        duration = base_duration + min(progression, 40)  # Cap at 70 min

        # Calories burned correlates with duration
        calories = int(duration * random.uniform(7, 10))

        cursor.execute(
            "INSERT INTO workouts (user_id, type, duration, calories, entry_number) VALUES (?, ?, ?, ?, ?)",
            (user_id, workout_type, duration, calories, i)
        )

    print(f"   Added 20 workout entries")
    print(f"   Types: {', '.join(set(workout_types))}")

    # Generate 20 nutrition entries
    print("\n🍎 Generating nutrition data...")
    meals = ["Breakfast", "Lunch", "Dinner", "Snack"]
    foods = {
        "Breakfast": ["Oatmeal", "Eggs & Toast", "Smoothie", "Yogurt"],
        "Lunch": ["Chicken Salad", "Tuna Sandwich", "Quinoa Bowl", "Soup"],
        "Dinner": ["Grilled Salmon", "Lean Beef", "Vegetable Stir-fry", "Chicken Breast"],
        "Snack": ["Protein Bar", "Fruit", "Nuts", "Greek Yogurt"]
    }

    for i in range(1, 21):
        meal = random.choice(meals)
        food = random.choice(foods[meal])

        # Calories vary by meal type
        calorie_ranges = {
            "Breakfast": (300, 450),
            "Lunch": (400, 600),
            "Dinner": (450, 700),
            "Snack": (100, 250)
        }
        min_cal, max_cal = calorie_ranges[meal]
        calories = random.randint(min_cal, max_cal)

        cursor.execute(
            "INSERT INTO nutrition (user_id, meal, food, calories, entry_number) VALUES (?, ?, ?, ?, ?)",
            (user_id, meal, food, calories, i)
        )

    print(f"   Added 20 nutrition entries")
    print(f"   Meal types: {', '.join(meals)}")

    # Generate 15 weight entries showing progress
    print("\n⚖️ Generating weight progress data...")
    start_weight = 75.0
    target_weight = 70.0

    for i in range(1, 16):
        # Create realistic weight loss trend with some fluctuations
        progress = (i - 1) / 14  # 0 to 1 over 15 entries
        target = start_weight - (progress * 5)  # Aim for 5kg loss
        fluctuation = random.uniform(-0.3, 0.3)  # Small daily fluctuations
        weight = round(target + fluctuation, 1)

        cursor.execute(
            "INSERT INTO weight_log (user_id, weight, entry_number) VALUES (?, ?, ?)",
            (user_id, weight, i)
        )

    print(f"   Added 15 weight entries")
    print(f"   Shows progress from 75.0 kg to ~70.0 kg")

    # Update user's current weight
    cursor.execute("UPDATE users SET weight = ? WHERE id = ?", (70.2, user_id))

    conn.commit()
    conn.close()

    print("\n" + "=" * 60)
    print("✅ SAMPLE DATABASE CREATED SUCCESSFULLY!")
    print("=" * 60)
    print("\n📊 Database now contains:")
    print("   - 1 demo user (demo_user / demo123)")
    print("   - 20 workout entries")
    print("   - 20 nutrition entries")
    print("   - 15 weight entries")
    print("\n🚀 Now run the main application to see graphs!")
    print("   Command: python fitness_tracker.py")


def show_sample_queries():
    """Show sample SQL queries that can be used for graphs"""
    print("\n" + "=" * 60)
    print("📋 SAMPLE GRAPH QUERIES FOR DEMONSTRATION")
    print("=" * 60)

    print("\n1. WEIGHT PROGRESS (Line Graph):")
    print("""   SELECT entry_number, weight
                FROM weight_log
                WHERE user_id = 1
                ORDER BY entry_number""")

    print("\n2. WORKOUT DURATION TREND (Line Graph):")
    print("""   SELECT entry_number, duration
                FROM workouts
                WHERE user_id = 1
                ORDER BY entry_number""")

    print("\n3. DAILY CALORIE INTAKE (Line Graph):")
    print("""   SELECT entry_number, calories
                FROM nutrition
                WHERE user_id = 1
                ORDER BY entry_number""")

    print("\n4. WORKOUT TYPE DISTRIBUTION (Bar Graph):")
    print("""   SELECT type, COUNT(*)
                FROM workouts
                WHERE user_id = 1
                GROUP BY type
                ORDER BY COUNT(*) DESC""")

    print("\n5. AVERAGE CALORIES PER MEAL TYPE (Bar Graph):")
    print("""   SELECT meal, AVG(calories)
                FROM nutrition
                WHERE user_id = 1
                GROUP BY meal
                ORDER BY AVG(calories) DESC""")


def generate_graph_screenshots():
    """Instructions for generating graph screenshots"""
    print("\n" + "=" * 60)
    print("📸 HOW TO GENERATE GRAPH SCREENSHOTS FOR GITHUB")
    print("=" * 60)

    print("\n1. Run the fitness tracker:")
    print("   python fitness_tracker.py")

    print("\n2. Login with:")
    print("   Username: demo_user")
    print("   Password: demo123")

    print("\n3. Generate these graphs (take screenshots):")
    print("   📈 Graph 1: Select 'View Graphs' → 'Weight Progress'")
    print("   📈 Graph 2: Select 'View Graphs' → 'Workout Duration'")
    print("   📈 Graph 3: Select 'View Graphs' → 'Calories Intake'")
    print("   📈 Graph 4: Select 'View Graphs' → 'Workout Types'")

    print("\n4. Custom Graph Example:")
    print("   Select 'View Graphs' → 'Custom Graph'")
    print("   Query: SELECT meal, AVG(calories) FROM nutrition WHERE user_id = 1 GROUP BY meal")
    print("   Title: 'Average Calories per Meal Type'")
    print("   X-label: 'Meal Type'")
    print("   Y-label: 'Average Calories'")
    print("   Graph type: 'bar'")


if __name__ == "__main__":
    create_sample_database()
    show_sample_queries()
    generate_graph_screenshots()