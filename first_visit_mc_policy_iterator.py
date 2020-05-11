import board
import play as p
import game
import numpy as np
from random_player import GreedyRandomPlayer

# utility functions
def to_board_state(board):
    return str(board)

def get_actions(Q, state):
    valuesByAction = Q[state]
    actions = valuesByAction.keys()
    return set(actions)

def update_action_probability(astar, possible_actions, epsilon):
    result = {}
    for action in possible_actions:
        if action == astar:
            result[action] = 1.0 - epsilon + epsilon / len(possible_actions)
        else:
            result[action] = epsilon / len(possible_actions)

    return result

class FirstVisitMCPolicyIterator:

    def sample_tic_tac_toe_policy(self, current_board, policy):

        current_state = to_board_state(current_board)
        # make a move
        if current_state not in policy.keys():

            # enumerate the set of available actions
            # initialize a random policy
            available_cells = board.get_available_cells(current_board)

            policy[current_state] = {}

            for cell in available_cells:
                action = board.to_state(cell)
                policy[current_state][action] = 1.0/len(available_cells)

        probabilityByAction = policy[current_state]
        actions = list(probabilityByAction.keys()) # these are integer values 0-8

        # compute CDF for each action
        pSelect = []
        pSoFar = 0
        for action in actions:
            p = probabilityByAction[action]
            pSoFar += p
            pSelect.append(pSoFar)

        self.file.write("policy by action: " + str(probabilityByAction) + "\n")
#        print("Actions: ", actions)
        self.file.write("Action: " + str(actions) + "\n")
#        print("PROBABILITIES: ", pSelect)
        self.file.write("PROBABILITIES: " + str(pSelect) + "\n")
        self.file.flush()

        # draw a value
        sample = np.random.uniform()
#        print("Sample value: ", sample)
        selectedAction = None
        for i in range(0,len(pSelect)):
            if sample < pSelect[i]:
                selectedAction = actions[i]
                break

#        print("Selected action: ", selectedAction)
        self.file.write("Selected action: " + str(selectedAction) + "\n")
        self.file.flush()
        return board.to_cell(selectedAction)

    def model_environment(self, opponent, state, action):

        game_complete = False
        initial_board = state

        self.file.write("AGENT MAKING MOVE: " + str(action) + str(board.to_state(action)) + "\n")

        current_board = p.add_move('X',action,initial_board)

#        print("AFTER AGENT MOVE:")
#        print(game.to_display_string(current_board))

        self.file.write("AFTER AGENT MOVE:\n")
        self.file.write(game.to_display_string(current_board))

        reward = 0.0

        if p.is_winner(current_board,'X'):
            game_complete = True
            reward = 1.0
        elif p.is_cat_game(current_board):
            game_complete = True
            reward = 0.0

        if not game_complete:
            # let the opponent make a move ...
            (opponent_id, opponent_move) = opponent.pick_next_move(current_board)

            current_board = p.add_move(opponent_id, opponent_move, current_board)

#            print("AFTER OPPONENT MOVE")
#            print(game.to_display_string(current_board))

            self.file.write("AFTER OPPONENT MOVE\n")
            self.file.write(game.to_display_string(current_board))

            if p.is_winner(current_board,opponent_id):
                game_complete = True
                reward = -1.0
            elif p.is_cat_game(current_board):
                game_complete = True
                reward = 0

        return current_board, reward, game_complete

    def generate_tic_tac_toe_episode(self, policy):

        current_board = p.empty_board()

        opponent = GreedyRandomPlayer('O')

        game_complete = False
        transitions = []
        while not game_complete:

            previous_state = current_board

#            print("PRIOR TO MOVE:")
#            print(game.to_display_string(current_board))
            self.file.write("PRIOR TO MOVE:\n")
            self.file.write(game.to_display_string(current_board))

            selectedAction = self.sample_tic_tac_toe_policy(current_board, policy)

            # model the environment --> returns a next state and a reward
            next_state, reward, game_complete = self.model_environment(opponent, current_board, selectedAction)

            # append the episode
            transition = {}
            transition['from_state'] = current_board
            transition['to_state'] = next_state
            transition['action'] = selectedAction
            transition['reward'] = reward
            transitions.append(transition)

            self.file.write("ADDING TRANSITION: " + str(transition) + "\n")

            current_board = next_state

        # now figure out the reward
#        print("BOARD AT END OF EPISODE")
#        print(game.to_display_string(current_board))
        self.file.write("BOARD AT END OF EPISODE\n")
        self.file.write(game.to_display_string(current_board))

        return transitions

    # note --> this will mutate the values passed in
    def update_state_value_estimates(self, transitions, Q, returns):

        gamma = 0.95
        for i in range(0,len(transitions)):
            cumulative_reward = 0
            self.file.write("Computing future returns for transition [" + str(i) + "] : " + str(transitions[i]) + "\n")

            visited = transitions[i]
            for j in range(i,len(transitions)):
                transition = transitions[j]
                discount = pow(gamma,j-i)
                self.file.write("Reward for transition[" + str(j) + "] : " + str(transition['reward']) + " , discount: " + str(discount) + "\n")
                cumulative_reward += pow(gamma,j-i)*transition['reward']

            state = to_board_state(visited['from_state'])
            cell = visited['action'] # this is a cell
            action = board.to_state(cell)

            # update returns(s,a)
            if not state in returns.keys():
                returns[state] = {}

            returnsByAction = returns[state]
            if not action in returnsByAction.keys():
                returnsByAction[action] = []

            previous_returns = returnsByAction[action]
            previous_returns.append(cumulative_reward)

            self.file.write("Reward history for state " + str(state) + " / action " + str(action) +  " --> " + str(len(previous_returns)) +  " : " + str(previous_returns) + "\n")

            # update Q(s,a)
            if not state in Q:
                self.file.write("state " + str(state) + " not yet in Q(s,a) --> initializing empty map\n")
                Q[state] = {}
                # TODO:  enumerate possible actions, and set values to 0
                available_cells = board.get_available_cells(visited['from_state'])
                for cell in available_cells:
                    action = board.to_state(cell)
                    Q[state][action] = 0.0

            valueByAction = Q[state]
            valueByAction[action] = np.mean(previous_returns)
            self.file.write("Updating value for state " + str(state) + " / action: " + str(action) + " to: " + str(np.mean(previous_returns)) + "\n")
            self.file.flush()

        return Q, returns

    def find_max_action(self, Q, s):

        valuesByAction = Q[s]
        self.file.write("ValuesByAction for state: " + str(s) +  " --> " + str(valuesByAction) + "\n")

        actions = list(valuesByAction.keys())
        values = []
        for a in actions:
            values.append(valuesByAction[a])

        self.file.write("values --> " + str(values) + "\n")
        # now sort
        inx_sorted = np.argsort(values)

        self.file.write("sorted indices: " + str(inx_sorted) + "\n")

        # return highest value
        return actions[inx_sorted[-1]]

    def update_policy(self, transitions, Q, policy, epsilon):

#        print("PREVIOUS POLICY: ", policy)

        self.file.write("PREVIOUS POLICY: \n")
        for state in policy.keys():
            self.file.write(state + "\n")
            pByAction = policy[state]
            self.file.write("p(a): " + str(pByAction) + "\n")

        for transition in transitions:
            state = to_board_state(transition['from_state'])

#            print("UPDATE POLICY:  FROM STATE")
#            print(game.to_display_string(transition['from_state']))
            self.file.write(game.to_display_string(transition['from_state']))

            astar = self.find_max_action(Q, state)
#            print("UPDATE POLICY:  ASTAR ==> ", astar)
            self.file.write("UPDATE POLICY:  ASTAR ==> " + str(astar) + "\n")

            possible_actions = get_actions(policy, state)
#            print("POSSIBLE ACTIONS: ", possible_actions)
            self.file.write("POSSIBLE ACTIONS: " + str(possible_actions) + "\n")

            for action in possible_actions:
                if action == astar:
                    policy[state][action] = 1.0 - epsilon + epsilon / len(possible_actions)
                else:
                    policy[state][action] = epsilon / len(possible_actions)

#            print("UPDATED POLICY ==> ", policy)
            self.file.write("UPDATED POLICY: " + str(policy) + "\n")

        return policy

    def learn_tic_tac_toe(self, n=1, initial_epsilon=0.1):

        self.file = open("ipe_tic_tac_toe.txt","w")

        np.random.seed(0)

        Q = {}
        policy = {}  # map [state] --> [action] --> probability
        returns = {} # map [state] --> [action] --> list of returns

        iter_count = 0
        epsilon = initial_epsilon
        while True:

            iter_count += 1

            self.file.write("GENERATING EPISODE " + str(iter_count) + "----------\n")
            transitions = self.generate_tic_tac_toe_episode(policy)

            # TODO:  update Q(s,a) estimates
            Q, returns = self.update_state_value_estimates(transitions, Q, returns)

            # TODO:  update policy pi(a|s) --> act epsilon-greedy w.r.t. Q(s,a)
            policy = self.update_policy(transitions, Q, policy, epsilon)

            # TODO:  termination criteria?
            if iter_count >=n:
                break

        self.file.close()
        return Q, policy
