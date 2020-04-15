import player
import numpy as np
import play as p
import csv

class RLPlayer(player.Player):
    """
    create a RL player
    """
    def __init__(self, player):
        self.player = player
        self.board_to_value = {}
        self.learning_rate = 0.1
        self.epsilon = 0.05
        self.gamma = 0.99
        self.Q = {}
        self.log = open("learning/learning.txt","w")

    def finish(self):
        self.log.close()

    def find_actions_for_state(self, board):

        p.display_board(board)

        # given board, find available / unoccupied cells
        # this defines the next possible moves
        available_cells = p.get_available_cells(board)

        print("Found ", len(available_cells), " available cells --> ", available_cells)

        # create the next board
        actions = []
        for cell in available_cells:
            bnext = p.add_move(self.player,cell,board)
            action = {}
            action['cell'] = cell
            action['board'] = bnext
            actions.append(action)

        return actions

    def board_to_key(self, board):
        return str(board)

    def board_to_state(self, board):
        return str(board)

    def find_action(self, possible_actions, cell):
        for action in possible_actions:
            if action['cell'] == cell:
                return action

        return None

    def update_q_value_for_state_and_action(self, board, cell, value):
        state = self.board_to_state(board)
        # get actions for this state
        if state in self.Q.keys():
            possible_actions = self.Q[state]
        else:
            possible_actions = []
            self.Q[state] = possible_actions

        # now find the action corresponding to this cell
        action = self.find_action(possible_actions, cell)

        if action is None:
            action = {}
            action['cell'] = cell
            action['value'] = value
            possible_actions.append(action)
            self.log.write("SETTING Q VALUE FOR STATE ----- GIVEN ACTION " + str(cell) + "\n")
            self.log.write(self.to_string(board))
            self.log.write("Q-VALUE IS NOW: " + str(value) + "\n")
        else:
            print("Updating value for cell: ", cell, " to value: ", value)

            self.log.write("UPDATING Q VALUE FOR STATE ----- GIVEN ACTION " + str(cell) + "\n")
            self.log.write(self.to_string(board))
            self.log.write("Q-VALUE WAS: " + str(action['value']) + " --> IS NOW: " + str(value) + "\n")

            action['value'] = value

        self.log.write("---------- FINISHED UPDATING Q-VALUE --------\n")

    def get_value_for_state_and_action(self, board, cell):
        state = self.board_to_state(board)

        if not state in self.Q.keys():
            print("state: ", state, " not yet in Q ... returning 0")
            return 0

        possible_actions = self.Q[state]
        # now find the action corresponding to this cell
        action = self.find_action(possible_actions, cell)

        if action is None:
            print("action: ", cell, " is not yet defined for state ... returning 0")
            return 0

        return action['value']

    def add_or_update_value(self, board, value):
        key = self.board_to_key(board)
        self.board_to_value[key] = value

    def get_value(self, board):
        key = self.board_to_key(board)
        if key in self.board_to_value.keys():
            return self.board_to_value[key]
        else:
            return 0

    def find_q_values(self, actions):
        q_values = []
        for action in actions:
            value = self.get_value_for_state_and_action(action['board'],action['cell'])
            #value = self.get_value(action['board'])
            q_values.append(value)

        return q_values


    def contains_no_knowledge(self, q_values):
        num_zeros = 0
        for v in q_values:
            if (v == 0):
                num_zeros += 1

        return num_zeros == len(q_values)

    def update_q_value(self, board, action, reward):

        self.log.write("UPDATING Q VALUE FOR BOARD ------\n")
        self.log.write(self.to_string(board))
        self.log.write("GIVEN ACTION: " + str(action['cell']) + "\n")

#        previous_value = self.get_value(board)
        previous_value = self.get_value_for_state_and_action(board,action['cell'])

        self.log.write("PREVIOUS VALUE FOR STATE/ACTION: " + str(previous_value) + "\n")
        self.log.flush()
        # approximate FUTURE reward from the NEXT states ...
        # max_a Q(s',a)
        # first we advance from s --> s' given action A and reward R
        # then we take the max over all actions FROM s'
        # do we need to approximate the OTHER player's actions here??
        # otherwise, the board is OUT OF SYNC with GAME PLAY
        next_board = p.copy(action['board'])

        # examine other player moves --> for now, just do a random move??
        opponent_possible_cells = p.get_available_cells(next_board)

        if len(opponent_possible_cells) > 0:
            self.log.write("POSSIBLE OPPONENT NEXT MOVES: " + str(opponent_possible_cells) + "\n")
            # pick one at random
            inx_select = np.random.randint(0,len(opponent_possible_cells))
            opponent_move = opponent_possible_cells[inx_select]

            self.log.write("ASSUME OPPONENT NEXT MOVE: " + str(opponent_move) + "\n")

            next_board_seen = p.add_move(p.get_other_player(self.player),opponent_move,next_board)

            self.log.write("ASSUMED NEXT BOARD SEEN ----\n")
            self.log.write(self.to_string(next_board_seen))
            self.log.flush()

            next_actions = self.find_actions_for_state(next_board_seen)

            if (len(next_actions) > 0):
                next_q_values = self.find_q_values(next_actions)


                for i in range(0,len(next_actions)):
                    action = next_actions[i]
                    self.log.write("Q VALUE FOR ACTION " + str(i) + " --> " + str(action['cell']) + " : " + str(next_q_values[i]) + "\n")

                    print("Got q values for actions: ", next_q_values)

                    max_next_q = max(next_q_values)
            else:
                max_next_q = 0
        else:
            print("EMPTY POSSIBLE OPPONENT NEXT CELLS")
            max_next_q = 0

        next_value = previous_value + self.learning_rate*(reward + self.gamma * max_next_q - previous_value)

        print("Updated q value ==> ", next_value)

        self.add_or_update_value(board, next_value)

        self.update_q_value_for_state_and_action(board, action['cell'], next_value)


    def to_string(self,board):
        result = ""
        for row in board:
            result += str(row)
            result += "\n"

        return result

    def get_reward(self, board, action):

        next_board = action['board']
        print("In REWARD given next board -----")
        p.display_board(next_board)

        if p.is_winner(next_board, self.player):
            return 1.0
        elif p.is_potential_loser_on_next_move(next_board, self.player):
            print("POTENTIAL LOSER BOARD: ")
            self.log.write("POTENTIAL LOSING BOARD -----\n")
            self.log.write(self.to_string(next_board))
            p.display_board(next_board)
            return -0.5
        else:
            return 0.0

    def dump_q_values(self):
        with open('q_values.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["state", "action_values"])

            for key in self.Q.keys():
                writer.writerow([key, self.Q[key]])


    def pick_next_move(self, board):

#        s = derive_state_given(board)

        self.log.write("Given BOARD ...\n")
        self.log.write(self.to_string(board))

        actions = self.find_actions_for_state(board)
        print("Have ", len(actions), " possible actions...")

        q_values = self.find_q_values(actions)
        print("found ", len(q_values), " q values : ", q_values)

        # with probability epsilon, choose random action
        sample = np.random.uniform()
        action = None
        if sample < self.epsilon or self.contains_no_knowledge(q_values):
            inx_select = np.random.randint(0,len(q_values))
            action = actions[inx_select]
        else:
            inx_sorted = np.argsort(q_values)
            action = actions[inx_sorted[-1]]

        print("Selected action ==> ", action)

        self.log.write("SELECTED ACTION ==> " + str(action['cell']) + "\n")
        self.log.write("UPDATED BOARD ------\n")
        self.log.write(self.to_string(action['board']))

        # get reward
        reward = self.get_reward(board, action)

        self.log.write("RECEIVED REWARD: " + str(reward) + "\n")

        print("Got reward: ", reward)

        # update the value
        self.update_q_value(board,action,reward)

        # convert from action to cell value
        return (self.player, action['cell'])
