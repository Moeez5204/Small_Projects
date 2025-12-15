import sqlite3
import hashlib
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple


class SQLGraphGenerator:
    """Universal graph generator for SQL data"""

    @staticmethod
    def create_line_graph(db_path: str, query: str, params: tuple = (),
                          title: str = "Graph", x_label: str = "Entry",
                          y_label: str = "Value", output_file: str = None,
                          show_graph: bool = True):
        """
        Create a line graph from SQL query results
        """
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(query, params)
            data = cursor.fetchall()

            if not data:
                print("⚠️ No data found for graph")
                return

            x_vals = []
            y_vals = []

            for i, row in enumerate(data):
                if len(row) >= 2:
                    x_vals.append(f"Entry {i + 1}")
                    y_vals.append(float(row[1]))

            if not x_vals or not y_vals:
                print("⚠️ No valid data points for graph")
                return

            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, y_vals, 'bo-', linewidth=2, markersize=8)
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel(x_label, fontsize=12)
            plt.ylabel(y_label, fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45 if len(x_vals) > 5 else 0)

            for i, (x, y) in enumerate(zip(x_vals, y_vals)):
                plt.text(x, y, f'{y:.1f}', ha='center', va='bottom' if y >= 0 else 'top')

            plt.tight_layout()

            if output_file:
                plt.savefig(output_file, dpi=300)
                print(f"✅ Graph saved to {output_file}")

            if show_graph:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            print(f"❌ Error creating graph: {e}")
        finally:
            conn.close()

    @staticmethod
    def create_bar_graph(db_path: str, query: str, params: tuple = (),
                         title: str = "Graph", x_label: str = "Category",
                         y_label: str = "Value", output_file: str = None,
                         show_graph: bool = True):
        """Create a bar graph from SQL query results"""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        try:
            cursor.execute(query, params)
            data = cursor.fetchall()

            if not data:
                print("⚠️ No data found for graph")
                return

            x_vals = []
            y_vals = []

            for i, row in enumerate(data):
                if len(row) >= 2:
                    x_vals.append(f"Entry {i + 1}")
                    y_vals.append(float(row[1]))

            if not x_vals or not y_vals:
                print("⚠️ No valid data points for graph")
                return

            plt.figure(figsize=(10, 6))
            bars = plt.bar(x_vals, y_vals, color='skyblue', edgecolor='black')
            plt.title(title, fontsize=14, fontweight='bold')
            plt.xlabel(x_label, fontsize=12)
            plt.ylabel(y_label, fontsize=12)
            plt.grid(True, alpha=0.3, axis='y')

            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5,
                         f'{height:.1f}', ha='center', va='bottom')

            plt.tight_layout()

            if output_file:
                plt.savefig(output_file, dpi=300)
                print(f"✅ Graph saved to {output_file}")

            if show_graph:
                plt.show()
            else:
                plt.close()

        except Exception as e:
            print(f"❌ Error creating graph: {e}")
        finally:
            conn.close()


class FitnessTracker:
    def __init__(self):
        self.conn = sqlite3.connect('fitness.db')
        self.cursor = self.conn.cursor()
        self.current_user = None
        self.graph_gen = SQLGraphGenerator()
        self.setup_database()

    def setup_database(self):
        """Load database schema from SQL file"""
        try:
            with open('fitness_schema.sql', 'r') as f:
                schema = f.read()
            self.cursor.executescript(schema)
            self.conn.commit()
        except FileNotFoundError:
            print("⚠️ SQL schema file not found. Creating default tables...")
            self.create_default_tables()

    def create_default_tables(self):
        """Create default tables if SQL file is missing"""
        self.cursor.executescript("""
                                  CREATE TABLE IF NOT EXISTS users
                                  (
                                      id
                                      INTEGER
                                      PRIMARY
                                      KEY,
                                      username
                                      TEXT
                                      UNIQUE,
                                      password
                                      TEXT,
                                      weight
                                      REAL,
                                      height
                                      REAL,
                                      goal
                                      TEXT
                                  );

                                  CREATE TABLE IF NOT EXISTS workouts
                                  (
                                      id
                                      INTEGER
                                      PRIMARY
                                      KEY,
                                      user_id
                                      INTEGER,
                                      type
                                      TEXT,
                                      duration
                                      INTEGER,
                                      calories
                                      INTEGER,
                                      entry_number
                                      INTEGER,
                                      FOREIGN
                                      KEY
                                  (
                                      user_id
                                  ) REFERENCES users
                                  (
                                      id
                                  ) ON DELETE CASCADE
                                      );

                                  CREATE TABLE IF NOT EXISTS nutrition
                                  (
                                      id
                                      INTEGER
                                      PRIMARY
                                      KEY,
                                      user_id
                                      INTEGER,
                                      meal
                                      TEXT,
                                      food
                                      TEXT,
                                      calories
                                      INTEGER,
                                      entry_number
                                      INTEGER,
                                      FOREIGN
                                      KEY
                                  (
                                      user_id
                                  ) REFERENCES users
                                  (
                                      id
                                  ) ON DELETE CASCADE
                                      );

                                  CREATE TABLE IF NOT EXISTS weight_log
                                  (
                                      id
                                      INTEGER
                                      PRIMARY
                                      KEY,
                                      user_id
                                      INTEGER,
                                      weight
                                      REAL,
                                      entry_number
                                      INTEGER,
                                      FOREIGN
                                      KEY
                                  (
                                      user_id
                                  ) REFERENCES users
                                  (
                                      id
                                  ) ON DELETE CASCADE
                                      );
                                  """)
        self.conn.commit()

    def hash_password(self, password: str) -> str:
        return hashlib.sha256(password.encode()).hexdigest()

    def register(self, username: str, password: str) -> bool:
        """Register new user"""
        try:
            self.cursor.execute(
                "INSERT INTO users (username, password) VALUES (?, ?)",
                (username, self.hash_password(password))
            )
            self.conn.commit()
            print(f"✅ Registered: {username}")
            return True
        except sqlite3.IntegrityError:
            print("❌ Username taken")
            return False

    def login(self, username: str, password: str) -> bool:
        """User login"""
        self.cursor.execute(
            "SELECT id, username FROM users WHERE username=? AND password=?",
            (username, self.hash_password(password))
        )
        user = self.cursor.fetchone()
        if user:
            self.current_user = {'id': user[0], 'username': user[1]}
            print(f"✅ Welcome {username}!")
            return True
        print("❌ Invalid login")
        return False

    def logout(self):
        self.current_user = None
        print("✅ Logged out")

    def update_profile(self, weight: float = None, height: float = None, goal: str = None):
        """Update user profile"""
        if not self.current_user:
            return

        updates = []
        params = []

        if weight is not None:
            updates.append("weight = ?")
            params.append(weight)
        if height is not None:
            updates.append("height = ?")
            params.append(height)
        if goal is not None:
            updates.append("goal = ?")
            params.append(goal)

        if updates:
            params.append(self.current_user['id'])
            sql = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
            self.cursor.execute(sql, params)
            self.conn.commit()
            print("✅ Profile updated")

    def get_next_entry_number(self, table: str) -> int:
        """Get next entry number for a table"""
        self.cursor.execute(f"""
            SELECT MAX(entry_number) FROM {table} WHERE user_id = ?
        """, (self.current_user['id'],))
        result = self.cursor.fetchone()[0]
        return (result or 0) + 1

    def log_workout(self, workout_type: str, duration: int, calories: int = None):
        """Log a workout"""
        if not self.current_user:
            return

        entry_num = self.get_next_entry_number('workouts')

        self.cursor.execute(
            """INSERT INTO workouts (user_id, type, duration, calories, entry_number)
               VALUES (?, ?, ?, ?, ?)""",
            (self.current_user['id'], workout_type, duration, calories, entry_num)
        )
        self.conn.commit()
        print(f"✅ Logged {workout_type} workout (Entry #{entry_num})")

    def log_meal(self, meal: str, food: str, calories: int):
        """Log a meal"""
        if not self.current_user:
            return

        entry_num = self.get_next_entry_number('nutrition')

        self.cursor.execute(
            """INSERT INTO nutrition (user_id, meal, food, calories, entry_number)
               VALUES (?, ?, ?, ?, ?)""",
            (self.current_user['id'], meal, food, calories, entry_num)
        )
        self.conn.commit()
        print(f"✅ Logged {meal}: {food} (Entry #{entry_num})")

    def log_weight(self, weight: float):
        """Log weight measurement"""
        if not self.current_user:
            return

        entry_num = self.get_next_entry_number('weight_log')

        self.cursor.execute(
            "INSERT INTO weight_log (user_id, weight, entry_number) VALUES (?, ?, ?)",
            (self.current_user['id'], weight, entry_num)
        )

        self.cursor.execute(
            "UPDATE users SET weight = ? WHERE id = ?",
            (weight, self.current_user['id'])
        )

        self.conn.commit()
        print(f"✅ Weight logged: {weight} kg (Entry #{entry_num})")

    def get_recent_entries(self, table: str, limit: int = 5) -> List[Tuple]:
        """Get recent entries from a table"""
        if not self.current_user:
            return []

        self.cursor.execute(f"""
            SELECT * FROM {table} 
            WHERE user_id = ? 
            ORDER BY entry_number DESC 
            LIMIT ?
        """, (self.current_user['id'], limit))

        return self.cursor.fetchall()

    def show_recent_entries(self):
        """Show recent entries from all tables"""
        if not self.current_user:
            return

        print(f"\n📝 Recent Entries for {self.current_user['username']}")
        print("=" * 50)

        workouts = self.get_recent_entries('workouts', 3)
        if workouts:
            print("\n💪 Recent Workouts:")
            for w in workouts:
                print(f"  Entry #{w[5]}: {w[2]} - {w[3]} min ({w[4] or 'N/A'} cal)")

        meals = self.get_recent_entries('nutrition', 3)
        if meals:
            print("\n🍎 Recent Meals:")
            for m in meals:
                print(f"  Entry #{m[5]}: {m[2]} - {m[3]} ({m[4]} cal)")

        weights = self.get_recent_entries('weight_log', 3)
        if weights:
            print("\n⚖️ Recent Weights:")
            for w in weights:
                print(f"  Entry #{w[3]}: {w[2]} kg")

    def show_graph_menu(self):
        """Show graph menu options"""
        print("\n📈 Graph Types:")
        print("1. Weight Progress (Line)")
        print("2. Workout Duration (Line)")
        print("3. Calories Intake (Line)")
        print("4. Workout Types (Bar)")
        print("5. Custom Graph")

        choice = input("\nSelect graph type (1-5): ").strip()

        if not self.current_user:
            print("❌ Please login first")
            return

        if choice == '1':
            self.graph_gen.create_line_graph(
                db_path='fitness.db',
                query="SELECT entry_number, weight FROM weight_log WHERE user_id = ? ORDER BY entry_number",
                params=(self.current_user['id'],),
                title=f"{self.current_user['username']}'s Weight Progress",
                x_label="Entry Number",
                y_label="Weight (kg)"
            )

        elif choice == '2':
            self.graph_gen.create_line_graph(
                db_path='fitness.db',
                query="SELECT entry_number, duration FROM workouts WHERE user_id = ? ORDER BY entry_number",
                params=(self.current_user['id'],),
                title=f"{self.current_user['username']}'s Workout Duration",
                x_label="Entry Number",
                y_label="Duration (minutes)"
            )

        elif choice == '3':
            self.graph_gen.create_line_graph(
                db_path='fitness.db',
                query="SELECT entry_number, calories FROM nutrition WHERE user_id = ? ORDER BY entry_number",
                params=(self.current_user['id'],),
                title=f"{self.current_user['username']}'s Calorie Intake",
                x_label="Entry Number",
                y_label="Calories"
            )

        elif choice == '4':
            self.graph_gen.create_bar_graph(
                db_path='fitness.db',
                query="SELECT type, COUNT(*) FROM workouts WHERE user_id = ? GROUP BY type",
                params=(self.current_user['id'],),
                title=f"{self.current_user['username']}'s Workout Types",
                x_label="Workout Type",
                y_label="Count"
            )

        elif choice == '5':
            self.create_custom_graph()

    def create_custom_graph(self):
        """Create custom graph with user SQL"""
        print("\n🛠️ Custom Graph Generator")
        print("-" * 40)
        print("Example queries:")
        print("  SELECT entry_number, weight FROM weight_log WHERE user_id = ?")
        print("  SELECT entry_number, duration FROM workouts WHERE user_id = ?")
        print("  SELECT entry_number, calories FROM nutrition WHERE user_id = ?")
        print("  SELECT type, COUNT(*) FROM workouts WHERE user_id = ? GROUP BY type")
        print("-" * 40)

        query = input("\nEnter SQL query: ").strip()
        title = input("Graph title: ").strip() or "Custom Graph"
        x_label = input("X-axis label (default: Entry): ").strip() or "Entry"
        y_label = input("Y-axis label (default: Value): ").strip() or "Value"
        graph_type = input("Graph type (line/bar): ").strip().lower() or "line"

        try:
            if graph_type == 'bar':
                self.graph_gen.create_bar_graph(
                    db_path='fitness.db',
                    query=query,
                    params=(self.current_user['id'],),
                    title=title,
                    x_label=x_label,
                    y_label=y_label
                )
            else:
                self.graph_gen.create_line_graph(
                    db_path='fitness.db',
                    query=query,
                    params=(self.current_user['id'],),
                    title=title,
                    x_label=x_label,
                    y_label=y_label
                )
        except Exception as e:
            print(f"❌ Error: {e}")

    def close(self):
        """Close database connection"""
        self.conn.close()


def main():
    tracker = FitnessTracker()

    print("=" * 50)
    print("FITNESS TRACKER - ENTRY NUMBER BASED")
    print("=" * 50)

    while True:
        if tracker.current_user:
            print(f"\n👤 User: {tracker.current_user['username']}")
            print("1. Log Workout")
            print("2. Log Meal")
            print("3. Log Weight")
            print("4. Update Profile")
            print("5. Show Recent Entries")
            print("6. View Graphs")
            print("7. Logout")
            print("8. Exit")
        else:
            print("\n1. Login")
            print("2. Register")
            print("3. Exit")

        choice = input("\nChoose: ").strip()

        if not tracker.current_user:
            if choice == '1':
                username = input("Username: ")
                password = input("Password: ")
                tracker.login(username, password)
            elif choice == '2':
                username = input("Username: ")
                password = input("Password: ")
                tracker.register(username, password)
            elif choice == '3':
                print("Goodbye!")
                break

        else:
            if choice == '1':
                workout_type = input("Workout type: ")
                duration = int(input("Duration (minutes): "))
                calories = input("Calories burned (optional): ")
                calories = int(calories) if calories.isdigit() else None
                tracker.log_workout(workout_type, duration, calories)

            elif choice == '2':
                meal = input("Meal (breakfast/lunch/dinner/snack): ")
                food = input("Food: ")
                calories = int(input("Calories: "))
                tracker.log_meal(meal, food, calories)

            elif choice == '3':
                weight = float(input("Weight (kg): "))
                tracker.log_weight(weight)

            elif choice == '4':
                weight = input("New weight (kg, optional): ")
                weight = float(weight) if weight else None
                height = input("Height (cm, optional): ")
                height = float(height) if height else None
                goal = input("Fitness goal (optional): ") or None
                tracker.update_profile(weight, height, goal)

            elif choice == '5':
                tracker.show_recent_entries()

            elif choice == '6':
                tracker.show_graph_menu()

            elif choice == '7':
                tracker.logout()

            elif choice == '8':
                print("Goodbye!")
                break

    tracker.close()

if __name__ == "__main__":
    main()