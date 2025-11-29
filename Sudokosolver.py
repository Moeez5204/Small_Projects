class Solution(object):
    def solveSudoku(self, board):
        self.solve_optimized(board)

    def solve_optimized(self, board):
        made_progress = True
        while made_progress:
            made_progress = False

            # Strategy 1: Check for single missing numbers in boxes
            if self.fill_single_missing_in_boxes(board):
                made_progress = True
                continue

            # Strategy 2: Check for single missing numbers in rows
            if self.fill_single_missing_in_rows(board):
                made_progress = True
                continue

            # Strategy 3: Check for single missing numbers in columns
            if self.fill_single_missing_in_columns(board):
                made_progress = True
                continue

        # If strategies didn't solve it completely, use optimized backtracking
        self.backtrack_optimized(board)

    def fill_single_missing_in_boxes(self, board):
        filled_any = False
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                if self.fill_single_missing_in_box(board, box_row, box_col):
                    filled_any = True
        return filled_any

    def fill_single_missing_in_rows(self, board):
        filled_any = False
        for row in range(9):
            present_numbers = set()
            empty_cols = []

            for col in range(9):
                if board[row][col] != '.':
                    present_numbers.add(board[row][col])
                else:
                    empty_cols.append(col)

            if len(present_numbers) == 8 and len(empty_cols) == 1:
                missing_num = (set('123456789') - present_numbers).pop()
                board[row][empty_cols[0]] = missing_num
                filled_any = True

        return filled_any

    def fill_single_missing_in_columns(self, board):
        filled_any = False
        for col in range(9):
            present_numbers = set()
            empty_rows = []

            for row in range(9):
                if board[row][col] != '.':
                    present_numbers.add(board[row][col])
                else:
                    empty_rows.append(row)

            if len(present_numbers) == 8 and len(empty_rows) == 1:
                missing_num = (set('123456789') - present_numbers).pop()
                board[empty_rows[0]][col] = missing_num
                filled_any = True

        return filled_any

    def fill_single_missing_in_box(self, board, start_row, start_col):
        present_numbers = set()
        empty_cells = []

        for i in range(3):
            for j in range(3):
                row = start_row + i
                col = start_col + j
                if board[row][col] != '.':
                    present_numbers.add(board[row][col])
                else:
                    empty_cells.append((row, col))

        if len(present_numbers) == 8 and len(empty_cells) == 1:
            missing_num = (set('123456789') - present_numbers).pop()
            row, col = empty_cells[0]
            board[row][col] = missing_num
            return True
        return False

    def backtrack_optimized(self, board):
        empty_cell = self.find_best_empty_cell(board)
        if not empty_cell:
            return True

        row, col = empty_cell
        possible_numbers = self.get_possible_numbers(board, row, col)

        for num in possible_numbers:
            board[row][col] = num
            if self.backtrack_optimized(board):
                return True
            board[row][col] = '.'

        return False

    def find_best_empty_cell(self, board):
        best_cell = None
        min_possibilities = 10

        for row in range(9):
            for col in range(9):
                if board[row][col] == '.':
                    possibilities = self.get_possible_numbers(board, row, col)
                    if len(possibilities) < min_possibilities:
                        min_possibilities = len(possibilities)
                        best_cell = (row, col)

        return best_cell

    def get_possible_numbers(self, board, row, col):
        possible = set('123456789')

        # Check row and column
        for i in range(9):
            possible.discard(board[row][i])
            possible.discard(board[i][col])

        # Check 3x3 box
        start_row, start_col = (row // 3) * 3, (col // 3) * 3
        for i in range(3):
            for j in range(3):
                possible.discard(board[start_row + i][start_col + j])

        return possible

    def is_valid_placement(self, board, row, col, num):
        # Check row
        for i in range(9):
            if board[row][i] == num:
                return False

        # Check column
        for i in range(9):
            if board[i][col] == num:
                return False

        # Check box
        start_row, start_col = (row // 3) * 3, (col // 3) * 3
        for i in range(3):
            for j in range(3):
                if board[start_row + i][start_col + j] == num:
                    return False

        return True

    def print_board(self, board):
        """Helper function to print the board in a readable format"""
        for i in range(9):
            if i % 3 == 0 and i != 0:
                print("-" * 21)
            for j in range(9):
                if j % 3 == 0 and j != 0:
                    print("|", end=" ")
                print(board[i][j], end=" ")
            print()


# Single test case that runs when the file is executed directly
if __name__ == "__main__":
    solution = Solution()

    print("SUDOKU SOLVER TEST")
    print("=" * 40)

    # Test Case: Classic Sudoku
    board = [
        ["5", "3", ".", ".", "7", ".", ".", ".", "."],
        ["6", ".", ".", "1", "9", "5", ".", ".", "."],
        [".", "9", "8", ".", ".", ".", ".", "6", "."],
        ["8", ".", ".", ".", "6", ".", ".", ".", "3"],
        ["4", ".", ".", "8", ".", "3", ".", ".", "1"],
        ["7", ".", ".", ".", "2", ".", ".", ".", "6"],
        [".", "6", ".", ".", ".", ".", "2", "8", "."],
        [".", ".", ".", "4", "1", "9", ".", ".", "5"],
        [".", ".", ".", ".", "8", ".", ".", "7", "9"]
    ]

    print("Original board:")
    solution.print_board(board)

    solution.solveSudoku(board)

    print("\nSolved board:")
    solution.print_board(board)

    print("\nTest completed successfully!")