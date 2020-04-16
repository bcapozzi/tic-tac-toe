import game

class Episode:

    def __init__(self, agent, environment):
        self.agent = agent
        self.environment = environment

    def execute(self, initial_board):

        current_board = initial_board
        while not self.environment.is_completed():

            player_id, player_move = self.agent.pick_next_move(current_board)
            action = {}
            action['player'] = player_id
            action['cell'] = player_move

            updated_board, reward = self.environment.update(current_board, action)
            current_board = updated_board

        print("EPISODE TERMINATED --> OUTCOME: ", self.environment.get_outcome())
        print(game.to_display_string(current_board))
        return self.agent, current_board
