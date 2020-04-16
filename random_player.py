import play as p
import numpy as np
import player
import board

class RandomPlayer(player.Player):
    """
    create a random player
    """
    def __init__(self, player):
        self.player = player

    def pick_next_move(self, current_board):
        # identify all available cells
        available_cells = []
        for row in p.get_rows(current_board):
            for col in p.get_cols(current_board):
                cell = (row,col)
                if (not p.is_occupied(current_board,cell)):
                    available_cells.append(cell)


        n = len(available_cells)
        inx_select = np.random.randint(n)
        return (self.player,available_cells[inx_select])


class GreedyRandomPlayer(player.Player):
    """
    create a greedy random player
    if it sees a chance to win it takes it
    """
    def __init__(self, player):
        self.player = player

    def pick_next_move(self, current_board):
        # identify all available cells
        available_cells = []
        for row in p.get_rows(current_board):
            for col in p.get_cols(current_board):
                cell = (row,col)
                if (not p.is_occupied(current_board,cell)):
                    available_cells.append(cell)

        for cell in available_cells:
            tmp_board = board.add_move(self.player, cell, current_board)
            if p.is_winner(tmp_board, self.player):
                return self.player, cell

        # otherwise, pick move at random
        n = len(available_cells)
        inx_select = np.random.randint(n)
        return (self.player,available_cells[inx_select])
