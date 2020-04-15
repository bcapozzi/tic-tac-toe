import play as p
import numpy as np
import player

class RandomPlayer(player.Player):
    """
    create a random player
    """
    def __init__(self, player):
        self.player = player

    def pick_next_move(self, board):
        # identify all available cells
        available_cells = []
        for row in p.get_rows(board):
            for col in p.get_cols(board):
                cell = (row,col)
                if (not p.is_occupied(board,cell)):
                    available_cells.append(cell)


        n = len(available_cells)
        inx_select = np.random.randint(n)
        return (self.player,available_cells[inx_select])
