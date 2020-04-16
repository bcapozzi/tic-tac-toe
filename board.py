import numpy as np

def to_state(cell):
    nrows = 3
    ncols = 3
    return cell[0]*ncols + cell[1]

def to_cell(state):
    nrows = 3
    ncols = 3
    in_row = np.floor(state / (ncols))
    in_col = (state/ncols - in_row)*nrows
    return (int(in_row),int(in_col))

def copy(board):
    #print("COPY --> INPUT BOARD: ", board)
    result = []
    for row in range(3):
        values = []
        for col in range(3):
            values.append(board[row][col])
        result.append(values)

    return result

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
