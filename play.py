# - - -
# - - -
# - - -
# 0

# O - -
# - - -
# - - -
# 1

# - O -
# - - -
# - - -
# 2

# ...
# - - -
# - - -
# - - O
# 9

# O X O
# - - -
# - - -
# 10

# O X -
# O - -
# - - -
# 11

# O X -
# - O -
# - - -
# 12

# ...

# O X -
# - - -
# - - O
# 16

def get_other_player(player):
    if player == 'X':
        return 'O'
    else:
        return 'X'

def is_potential_loser_on_next_move(board, player):
    other_player = get_other_player(player)
    # can other player win if it were to play on one of the available moves
    available_cells = get_available_cells(board)
    for cell in available_cells:
        potential_board = add_move(other_player, cell, board)
        if is_winner(potential_board, other_player):
            return True

    return False


def display_board(board):
    for row in board:
        print(row)

    print('------------')

def get_columns(board):
    columns = []
    for i in range(0,3):
        col = [board[0][i], board[1][i], board[2][i]]
        columns.append(col)

    return columns

def get_diagonals(board):
    diag1 = [board[0][0],board[1][1],board[2][2]]
    diag2 = [board[0][2],board[1][1],board[2][0]]
    return [diag1,diag2]

def is_cells_winner(cells, player):
    n_match = 0
    for cell in cells:
        if cell == player:
            n_match += 1

    return n_match == len(cells)

def is_winner(board, player):
    # find cells with this player's value in them
    cells = []

    # check rows
    for row in board:
        if (is_cells_winner(row, player)):
            return True

    columns = get_columns(board)
    for col in columns:
        if (is_cells_winner(col, player)):
            return True

    diagonals = get_diagonals(board)
    for diagonal in diagonals:
        if (is_cells_winner(diagonal, player)):
            return True

    return False

def is_cat_game(board):
    # all cells occupied, but no winner
    num_occupied = 0
    for row in board:
        for value in row:
            if not value == '-':
                num_occupied += 1

    return num_occupied == 9

def get_available_cells(board):
    available_cells = []
    for row in range(0,len(board)):
        for col in range(0,len(board[0])):
            if not is_occupied(board,(row,col)):
                available_cells.append((row,col))

    return available_cells

def is_occupied(board, cell):
    row = cell[0]
    col = cell[1]
    return not (board[row][col] == '-')

def is_outside_board(cell):
    row = cell[0]
    col = cell[1]
    if (row < 0 or row > 2):
        return True

    if (col < 0 or col > 2):
        return True

    return False

def copy(board):
    result = []
    for row in range(3):
        values = []
        for col in range(3):
            values.append(board[row][col])
        result.append(values)

    return result

def add_move(player, cell, board):
    updated_board = copy(board)
    if (is_occupied(updated_board,cell)):
        return updated_board
    elif (is_outside_board(cell)):
        return updated_board
    else:
        row = cell[0]
        col = cell[1]
        updated_board[row][col] = player
        #print("Updated board: ", updated_board)

        return updated_board

def get_rows(board):
    return range(0,len(board))

def get_cols(board):
    return range(0,len(board[0]))

def empty_board():
    board = []
    for row in range(3):
        board.append(['-','-','-'])

    return board

def board_to_key(board):
    return str(board)

def is_same(board1, board2):
    return board_to_key(board1) == board_to_key(board2)

def add_or_update_value(board, value):
    key = board_to_key(board)
    board_to_value[key] = value

def get_value(board):
    key = board_to_key(board)
    if key in board_to_value.keys():
        return board_to_value[key]
    else:
        return 0
