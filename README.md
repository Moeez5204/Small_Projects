# Single-File Projects Portfolio 🚀

## Overview

A collection of complete, self-contained applications in a single file each. Each project solves a specific problem or demonstrates a particular technology in the most minimal, portable format possible.

## 📊 Project Catalog

### 🔵 Financial Time Series Analysis (`FinancialTimeSeriesAnalysis.R`)

**📈 Interactive financial dashboard for stock analysis and volatility modeling**

A professional-grade Shiny application for analyzing financial markets:

* Multi-asset analysis with automatic USD conversion
* GARCH volatility modeling (sGARCH, eGARCH, gjrGARCH, apARCH)
* Risk metrics calculation (VaR, CVaR, Sharpe ratios)
* Interactive visualizations using Plotly
* Export capabilities for further analysis

**Tech Stack:** R, Shiny, quantmod, rugarch, Plotly, DT

### 🟡 Sudoku Solver (`Sudokosolver.py`)

**🧠 Advanced Sudoku puzzle solver using backtracking algorithm**

A Python implementation that can solve any valid Sudoku puzzle:

* Backtracking algorithm with optimization
* Input validation and error checking
* Step-by-step solving visualization (optional)
* Multiple puzzle difficulty support
* Puzzle generation capabilities

**Tech Stack:** Python, backtracking algorithm, recursion

### 🔥 PyTorch Neural Network Trainer (`PyTorchNeuralNetworkTrainer.py`)

**🤖 Complete neural network training pipeline with PyTorch**

A comprehensive deep learning framework for training and evaluating neural networks:

* Flexible neural network architecture configuration
* Multiple optimization algorithms (SGD, Adam, RMSprop)
* Learning rate scheduling with plateau detection
* Real-time training visualization with loss/accuracy plots
* Model checkpointing and early stopping
* GPU acceleration support (CUDA)

**Tech Stack:** Python, PyTorch, NumPy, Matplotlib, scikit-learn

### 🏋️ Fitness Tracker (`FitnessTracker/` folder)

A comprehensive fitness application with database backend and graphical analytics:

* **User Management**: Registration, login, profile updates with secure password hashing
* **Workout Tracking**: Log workouts with type, duration, calories burned
* **Nutrition Tracking**: Log meals with calories and food details
* **Weight Progress**: Track weight changes with sequential entry system (Entry 1, Entry 2...)
* **Data Visualization**: Line and bar graphs for progress tracking using matplotlib
* **Universal Graph Generator**: Modular SQLGraphGenerator class for any database visualization
* **SQL Database**: SQLite backend with 4 normalized tables (users, workouts, nutrition, weight_log)

**Tech Stack:** Python, SQLite, matplotlib, hashlib for security, universal graph generator

**Special Features:**
* Entry-based tracking system (instead of dates) - Entry 1, Entry 2, Entry 3...
* Built-in demo data generator for instant showcase
* Custom SQL query support for creating any graph
* Modular architecture that can be reused in other projects
* **Use demo credentials:** Username: `demo_user`, Password: `demo123`

## 📝 License

These projects are provided for educational and portfolio purposes. Feel free to use, modify, and distribute with proper attribution.

---

