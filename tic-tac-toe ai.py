import copy

# Function to print the Tic-Tac-Toe board
def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 5)

# Function to check if a player has won
def check_winner(board, player):
    # Check rows and columns
    for i in range(3):
        if all(board[i][j] == player for j in range(3)) or all(board[j][i] == player for j in range(3)):
            return True

    # Check diagonals
    if all(board[i][i] == player for i in range(3)) or all(board[i][2 - i] == player for i in range(3)):
        return True

    return False

# Function to check if the board is full
def is_board_full(board):
    return all(board[i][j] != ' ' for i in range(3) for j in range(3))

# Function to evaluate the current state of the board for the AI player
def evaluate(board):
    if check_winner(board, 'X'):
        return -1  # Human wins
    elif check_winner(board, 'O'):
        return 1   # AI wins
    elif is_board_full(board):
        return 0   # It's a draw
    else:
        return None  # Game is still ongoing

# Minimax algorithm with Alpha-Beta Pruning
def minimax(board, depth, maximizing_player, alpha, beta):
    score = evaluate(board)

    if score is not None:
        return score

    if maximizing_player:
        max_eval = float('-inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'O'
                    eval = minimax(board, depth + 1, False, alpha, beta)
                    board[i][j] = ' '

                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)

                    if beta <= alpha:
                        break

        return max_eval
    else:
        min_eval = float('inf')
        for i in range(3):
            for j in range(3):
                if board[i][j] == ' ':
                    board[i][j] = 'X'
                    eval = minimax(board, depth + 1, True, alpha, beta)
                    board[i][j] = ' '

                    min_eval = min(min_eval, eval)
                    beta = min(beta, eval)

                    if beta <= alpha:
                        break

        return min_eval

# Function to find the best move for the AI player
def find_best_move(board):
    best_val = float('-inf')
    best_move = None

    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                board[i][j] = 'O'
                move_val = minimax(board, 0, False, float('-inf'), float('inf'))
                board[i][j] = ' '

                if move_val > best_val:
                    best_move = (i, j)
                    best_val = move_val

    return best_move

# Function to play the Tic-Tac-Toe game
def play_game():
    board = [[' ' for _ in range(3)] for _ in range(3)]

    print("Welcome to Tic-Tac-Toe!")
    print_board(board)

    while True:
        # Human player's move
        row = int(input("Enter the row (0, 1, or 2): "))
        col = int(input("Enter the column (0, 1, or 2): "))

        if board[row][col] == ' ':
            board[row][col] = 'X'
        else:
            print("Invalid move. The cell is already occupied. Try again.")
            continue

        print_board(board)

        # Check if human player wins or the game is a draw
        if check_winner(board, 'X'):
            print("Congratulations! You win!")
            break
        elif is_board_full(board):
            print("It's a draw!")
            break

        # AI player's move
        print("AI is making a move...")
        ai_row, ai_col = find_best_move(board)
        board[ai_row][ai_col] = 'O'

        print_board(board)

        # Check if AI player wins or the game is a draw
        if check_winner(board, 'O'):
            print("AI wins! Better luck next time.")
            break
        elif is_board_full(board):
            print("It's a draw!")
            break

if __name__ == "__main__":
    play_game()
