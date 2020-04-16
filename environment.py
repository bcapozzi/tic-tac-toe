import board
import play

class Environment:

    def __init__(self, opponent):
        self.opponent = opponent
        self.is_complete = False
        self.outcome = "UNKNOWN"

    def update(self, state, action):

        print("ENVIRONMENT UPDATE: ", state, action)

        # given the current state, determine reward and next state
        next_state = board.add_move(action['player'], action['cell'], state)

        reward = 0.0
        if play.is_winner(next_state, action['player']):
            reward = 1.0
            self.is_complete = True
            self.outcome = action['player'] + "_WINNER"
        elif play.is_cat_game(next_state):
            self.is_complete = True
            self.outcome = "CAT_GAME"

        if not self.is_complete:
            # OTHERWISE, let the OTHER player play
            opponent_id, opponent_cell = self.opponent.pick_next_move(next_state)
            next_state = board.add_move(opponent_id, opponent_cell, next_state)

            if play.is_winner(next_state, opponent_id):
                reward = -1.0
                self.is_complete = True
                self.outcome = opponent_id + "_WINNER"
            elif play.is_cat_game(next_state):
                self.is_complete = True
                self.outcome = "CAT_GAME"

        return next_state, reward

    def is_completed(self):
        return self.is_complete

    def get_outcome(self):
        return self.outcome
