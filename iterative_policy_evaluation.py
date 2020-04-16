import numpy as np
import board
import play as p
import game
from random_player import GreedyRandomPlayer

def to_string(grid):
    result = ""
    for row in grid:
        result += "["
        for value in row:
            result += str(value).rjust(3)

        result += "]"
        result += "\n"
    return result

def to_cell(grid, state):
    nrows = len(grid)
    ncols = len(grid[0])
    in_row = np.floor(state / (ncols))
    in_col = (state/ncols - in_row)*nrows
    return (int(in_row),int(in_col))

def get_transition_probability(grid, state1, state2, action):

    # convert to equivalent cell
    cell1 = to_cell(grid, state1)
    cell2 = to_cell(grid, state2)

    drow = cell2[0] - cell1[0]
    dcol = cell2[1] - cell1[1]
    nrows = len(grid)
    ncols = len(grid[0])

    # if in terminal state, then transition to anything other than the same state is zero
    if cell1[0] == cell1[1] == 0:
        if drow == 0 and dcol == 0:
            return 1.0
        else:
            return 0.0
    elif cell1[0] == cell1[1] == 3:
        if drow == 0 and dcol == 0:
            return 1.0
        else:
            return 0.0

    if action == 'R':
        if dcol == 1 and drow == 0:
            return 1.0
        elif dcol == 0 and drow == 0 and cell1[1] == ncols - 1:
            return 1.0
        else:
            return 0.0
    elif action == 'L':
        if dcol == -1 and drow == 0:
            return 1.0
        elif dcol == 0 and drow == 0 and cell1[1] == 0:
            return 1.0
        else:
            return 0.0
    elif action == 'U':
        if dcol == 0 and drow == -1:
            return 1.0
        elif dcol == 0 and drow == 0 and cell1[0] == 0:
            return 1.0
        else:
            return 0.0
    elif action == 'D':
        if dcol == 0 and drow == 1:
            return 1.0
        elif dcol == 0 and drow == 0 and cell1[0] == nrows-1:
            return 1.0
        else:
            return 0.0
    else:
        return 0.0


def create_grid():
    grid = []
    for row in range(0,4):
        states = []
        for col in range(0,4):
            s = row*4 + col
            states.append(s)
        grid.append(states)

    return grid

def init_value_function(grid):
    V = {}
    for row in grid:
        for value in row:
            V[value] = 0.0
    return V

def get_states(grid):
    states = []
    for row in grid:
        for value in row:
            states.append(value)
    return states

def get_reward(grid, state1, state2):
    cell1 = to_cell(grid, state1)
    if cell1[0] == cell1[1] == 0:
        return 0.0
    elif cell1[0] == cell1[1] == 3:
        return 0.0
    else:
        return -1.0

def get_reward2(grid, cell1, cell2):
    if cell1[0] == cell1[1] == 0:
        return 0.0
    elif cell1[0] == cell1[1] == 3:
        return 0.0
    else:
        return -1.0

def test_iteration_on_grid(n):

    # given states
    grid = create_grid()
    states = get_states(grid)

    print("GRID")
    print(to_string(grid))

    # initialize value function for all states
    V = init_value_function(grid)
    print("INITIAL VALUE FUNCTION: ", V)

    actions = ['R','L','U','D']
    gamma = 1.0 # undiscounted
    theta = 0.0001

    iter_count = 0
    while True:
        iter_count += 1
        delta = 0
        Vnew = {}
        for s in states:
            v = V[s]
            sum = 0
            for a in actions:
                # sample --> uniform distribution over all actions
                pi = 1.0/len(actions)
                inner_sum = 0
                for snext in states:
                    r = get_reward(grid, s, snext)
                    p = get_transition_probability(grid, s, snext, a)
                    inner_sum += p*(r + gamma*V[snext])
                    #print("s: ", s, " via action: ", a, " --> snext: ", snext, " r = ", r, " p = ", p, " inner_sum: ", inner_sum)


                #print("State ", s, " , Finished action ", a, " --> a_sum: ", inner_sum)
                sum += pi*inner_sum
            Vnew[s] = sum
            delta = max(delta, abs(v - Vnew[s]))

        # dump the updated value function
        print("Iteration ", iter_count, " --> ", Vnew)
        if (delta < theta):
            break

        # update the value function
        V = Vnew

        if (iter_count >= n):
            break

def do_transition(grid, from_cell, action):

    if action == 'R':
        dcol = 1
        drow = 0
    elif action == 'L':
        dcol = -1
        drow = 0
    elif action == 'U':
        dcol = 0
        drow = -1
    elif action == 'D':
        dcol = 0
        drow = 1

    if (from_cell[0] == 0 and from_cell[1] == 0):
        drow = 0
        dcol = 0
    elif (from_cell[0] == 3 and from_cell[1] == 3):
        drow = 0
        dcol = 0

    to_row = from_cell[0] + drow
    to_col = from_cell[1] + dcol

    to_row = max(0, to_row)
    to_row = min(len(grid)-1, to_row)

    to_col = max(0, to_col)
    to_col = min(len(grid[0])-1, to_col)

    reward = get_reward2(grid,from_cell,(to_row,to_col))

    return (to_row, to_col), reward

def sample_policy(policy, state):
    probabilityByAction = policy[state]
    actions = list(probabilityByAction.keys())

    # compute CDF for each action
    pSelect = []
    pSoFar = 0
    for action in actions:
        p = probabilityByAction[action]
        pSoFar += p
        pSelect.append(pSoFar)

    #print("Actions: ", actions)
    #print("PROBABILITIES: ", pSelect)

    # draw a value
    sample = np.random.uniform()
    #print("Sample value: ", sample)
    selectedAction = None
    for i in range(0,len(pSelect)):
        if sample < pSelect[i]:
            selectedAction = actions[i]
            break

    #print("Selected action: ", selectedAction)
    return selectedAction

def generate_state_action_episode(grid, from_cell, policy):
    states = get_states(grid)
    current_cell = from_cell

    print("Generating episode starting from state: ", to_state(grid, current_cell), " --> ", current_cell)

    transitions = []
    steps = 0
    while True:

        if (current_cell == (0,0) or current_cell == (3,3)):
            break

        # now do a transition
        #inx_select = np.random.randint(0,len(actions))
        #action = actions[inx_select]
        state = to_state(grid,current_cell)

        selectedAction = sample_policy(policy, state)
        # do a transition
        next_cell, reward = do_transition(grid, current_cell, selectedAction)

        steps += 1

        transition = {}
        transition['from_cell'] = current_cell
        transition['to_cell'] = next_cell
        transition['action'] = selectedAction
        transition['reward'] = reward

        transitions.append(transition)

        current_cell = next_cell

        # if we end up seeing the same cell too many times in a row, we should bail out

        if (steps > 100):
            print("NOT CONVERGING ...")
            print("POLICY: ", policy[state], " State: ", state, " Cell: ", current_cell)
            for j in range(0,10):
                print(transitions[j])

            return transitions


    return transitions

def generate_episode(grid, from_cell, actions):

    states = get_states(grid)
    current_cell = from_cell

    transitions = []
    while True:

        if (current_cell == (0,0) or current_cell == (3,3)):
            break

        # now do a transition
        inx_select = np.random.randint(0,len(actions))
        action = actions[inx_select]

        # do a transition
        next_cell, reward = do_transition(grid, current_cell, action)

        transition = {}
        transition['from_cell'] = current_cell
        transition['to_cell'] = next_cell
        transition['action'] = action
        transition['reward'] = reward

        transitions.append(transition)

        current_cell = next_cell

    return transitions

def extract_state_actions_visited(transitions):

    value_dict = {}
    for t in transitions:
        # t['from_cell']
#        state_action = {}
        tmp = str(t['from_cell']) + ":" + t['action']
        value_dict[tmp] = (t['from_cell'], t['action'])
#        state_action['from_cell'] = t['from_cell']
#        state_action['action'] = t['action']
#        state_actions.append(state_action)

    state_actions = []
    for key in value_dict.keys():
        sa = value_dict[key]
        state_actions.append(sa)

    return state_actions


def extract_states_visited(transitions):
    states = []
    for t in transitions:
        states.append(t['from_cell'])
        states.append(t['to_cell'])

    return set(states)

def find_first_occurence_of(transitions, state):
    for i in range(0,len(transitions)):
        transition = transitions[i]
        if transition['from_cell'] == state:
            return i

    return -1

def pick_initial_cell(grid):
    nrows = len(grid)
    ncols = len(grid[0])

    inx_r = np.random.randint(0,nrows)
    inx_c = np.random.randint(0,ncols)

    return (inx_r, inx_c)

def to_state(grid, cell):
    nrows = len(grid)
    ncols = len(grid[0])
    return cell[0]*ncols + cell[1]

def init_state_action_value_function(states, actions):
    Q = {}
    for s in states:
        Q[s] = {}
        for a in actions:
            Q[s][a] = 0
    return Q

def init_state_action_returns(states, actions):
    returns = {}
    for s in states:
        returns[s] = {}
        for a in actions:
            returns[s][a] = []
    return returns

def to_states(grid, cells):
    states = []
    for cell in cells:
        state = to_state(grid, cell)
        states.append(state)

    return states

def init_random_policy(states, actions):
    policy = {}
    for state in states:
        policy[state] = {}
        for action in actions:
            policy[state][action] = 1.0/len(actions)

#    for action in actions:
#        policy[action] = {}
#        for state in states:
#            policy[action][state] = 1.0/len(actions)

    return policy

def find_max_action(Q, s):

    valuesByAction = Q[s]
    #print("ValuesByAction for state: ", s, " --> ", valuesByAction)

    actions = list(valuesByAction.keys())
    values = []
    for a in actions:
        values.append(valuesByAction[a])

    #print("values --> ", values)
    # now sort
    inx_sorted = np.argsort(values)

    #print("sorted indices: ", inx_sorted)

    # return highest value
    return actions[inx_sorted[-1]]

def get_actions(Q, state):
    valuesByAction = Q[state]
    actions = valuesByAction.keys()
    return set(actions)

def find_first_transition_index(transitions, state_action):

    #print("Looking for match for state/action: ", state_action)

    for i in range(0,len(transitions)):
        transition = transitions[i]
        if transition['from_cell'] == state_action[0] and transition['action'] == state_action[1]:
            return i

    return -1

def to_board_state(board):
    return str(board)

def test_sample_policy():

    board = p.empty_board()
    policy = {}
    move = sample_tic_tac_toe_policy(board,policy)
    print("SELECTED MOVE: ", move)

def sample_tic_tac_toe_policy(current_board, policy):

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

    print("Actions: ", actions)
    print("PROBABILITIES: ", pSelect)

    # draw a value
    sample = np.random.uniform()
    print("Sample value: ", sample)
    selectedAction = None
    for i in range(0,len(pSelect)):
        if sample < pSelect[i]:
            selectedAction = actions[i]
            break

    print("Selected action: ", selectedAction)
    return board.to_cell(selectedAction)

def test_episode():
    policy = {}
    transitions = generate_tic_tac_toe_episode(policy)
    for t in transitions:
        print(t)

def model_environment(opponent, state, action):

    game_complete = False
    initial_board = state
    current_board = p.add_move('X',action,initial_board)

    print("AFTER AGENT MOVE:")
    print(game.to_display_string(current_board))

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

        print("AFTER OPPONENT MOVE")
        print(game.to_display_string(current_board))


        if p.is_winner(current_board,opponent_id):
            game_complete = True
            reward = -1.0
        elif p.is_cat_game(current_board):
            game_complete = True
            reward = 0

    return current_board, reward, game_complete

def generate_tic_tac_toe_episode(policy):

    current_board = p.empty_board()

    opponent = GreedyRandomPlayer('O')

    game_complete = False
    transitions = []
    while not game_complete:

        previous_state = current_board

        print("PRIOR TO MOVE:")
        print(game.to_display_string(current_board))

        selectedAction = sample_tic_tac_toe_policy(current_board, policy)

        # model the environment --> returns a next state and a reward
        next_state, reward, game_complete = model_environment(opponent, current_board, selectedAction)

        # append the episode
        transition = {}
        transition['from_state'] = current_board
        transition['to_state'] = next_state
        transition['action'] = selectedAction
        transition['reward'] = reward
        transitions.append(transition)

        current_board = next_state

    # now figure out the reward
    print("BOARD AT END OF EPISODE")
    print(game.to_display_string(current_board))

    return transitions

# note --> this will mutate the values passed in
def update_state_value_estimates(transitions, Q, returns):

    gamma = 0.95
    for i in range(0,len(transitions)):
        cumulative_reward = 0
        for j in range(i,len(transitions)):
            transition = transitions[j]
            cumulative_reward = gamma*cumulative_reward + transition['reward']

            state = to_board_state(transition['from_state'])
            cell = transition['action'] # this is a cell
            action = board.to_state(cell)

            # update returns(s,a)
            if not state in returns.keys():
                returns[state] = {}

            returnsByAction = returns[state]
            if not action in returnsByAction.keys():
                returnsByAction[action] = []

            previous_returns = returnsByAction[action]
            previous_returns.append(cumulative_reward)

            #print("Reward history for state ", state, " / action ", action, " --> ", len(returns[state][action]), " : ", returns[state][action])

            # update Q(s,a)
            if not state in Q:
                Q[state] = {}

            valueByAction = Q[state]
            valueByAction[action] = np.mean(previous_returns)

    return Q, returns


def test_tic_tac_toe(n=1):

    Q = {}
    policy = {}  # map [state] --> [action] --> probability
    returns = {} # map [state] --> [action] --> list of returns

    iter_count = 0
    while True:

        iter_count += 1

        transitions = generate_tic_tac_toe_episode(policy)

        # TODO:  update Q(s,a) estimates
        Q, returns = update_state_value_estimates(transitions, Q, returns)

        # TODO:  update policy pi(a|s) --> act epsilon-greedy w.r.t. Q(s,a)

        # TODO:  termination criteria?
        if iter_count >=n:
            break


    return Q, policy



def test_on_policy_first_time_mc(n = 1):

    grid = create_grid()
    states = get_states(grid)
    actions = ['L','R','U','D']

    policy = init_random_policy(states, actions)

    # initialize state-action value function
    Q = init_state_action_value_function(states, actions)
    returns = init_state_action_returns(states, actions)

    print("INITIAL RETURNS: ", returns)

    initial_cell = (2,1)

    iter_count = 0
    while True:

        iter_count += 1

#        initial_cell = pick_initial_cell(grid)

        transitions = generate_state_action_episode(grid, initial_cell, policy)

        print("Number of transitions found in episode[", iter_count,"]: ", len(transitions))

        unique_state_actions = extract_state_actions_visited(transitions)
        print("FOUND ", len(unique_state_actions), " unique state/action pairs...")

        unique_cells = []
        for sa in unique_state_actions:
            unique_cells.append(sa[0])
    #    extract_states_visited(transitions)

        for state_action in unique_state_actions:
            cell = state_action[0]
            state = to_state(grid,cell)
            action = state_action[1]

            #print("State/action: ", state_action, " --> Action: ", action)

            inx = find_first_transition_index(transitions, state_action)
            #print("First transition index: ", inx)
            if (inx < 0):
                continue

            gamma = 0.95
            cumulative_reward = 0
            # go in the opposite order ?
            #tmp = list(range(inx,len(transitions)))
            #tmp.reverse()
            for i in range(inx,len(transitions)):
                transition = transitions[i]
                cumulative_reward = gamma*cumulative_reward + transition['reward']

            #print("Adding cumulative reward for state ", state, " / action ", action, " --> ", cumulative_reward)

            returns[state][action].append(cumulative_reward)

            #print("Reward history for state ", state, " / action ", action, " --> ", len(returns[state][action]), " : ", returns[state][action])

            Q[state][action] = np.mean(returns[state][action])

        # now update policy
        epsilon = 0.001
        for cell in unique_cells:
            state = to_state(grid, cell)
            astar = find_max_action(Q, state)

            possible_actions = get_actions(Q, state)

            for action in possible_actions:
                if action == astar:
                    policy[state][action] = 1.0 - epsilon + epsilon / len(possible_actions)
                else:
                    policy[state][action] = epsilon / len(possible_actions)

        if iter_count >= n:
            break

    return Q, policy


def test_first_time_mc(n = 1):
    grid = create_grid()
    states = get_states(grid)

    # initialization
    V = init_value_function(grid)
    returns = {}
    for s in states:
        returns[s] = []

    iter_count = 0
    while True:
        iter_count += 1
        initial_cell = pick_initial_cell(grid)
        transitions = generate_episode(grid, initial_cell, ['L','R','U','D'])
        cells_appearing = extract_states_visited(transitions)

        for cell in cells_appearing:
            inx = find_first_occurence_of(transitions, cell)
            if inx < 0:
                continue

            # accumulate rewards from that state forward to end of episode
            cumulative_reward = 0
            for i in range(inx,len(transitions)):
                transition = transitions[i]
                cumulative_reward += transition['reward']

            state = to_state(grid, cell)

            returns[state].append(cumulative_reward)
            V[state] = np.mean(returns[state])

        if (iter_count > n):
            break

    return V
