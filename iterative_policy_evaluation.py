import numpy as np

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
    return selectedAction

def generate_state_action_episode(grid, from_cell, policy):
    states = get_states(grid)
    current_cell = from_cell

    transitions = []
    while True:

        if (current_cell == (0,0) or current_cell == (3,3)):
            break

        # now do a transition
        #inx_select = np.random.randint(0,len(actions))
        #action = actions[inx_select]
        state = to_state(current_cell)

        selectedAction = sample_policy(policy, state)
        # do a transition
        next_cell, reward = do_transition(grid, current_cell, selectedAction)

        transition = {}
        transition['from_cell'] = current_cell
        transition['to_cell'] = next_cell
        transition['action'] = action
        transition['reward'] = reward

        transitions.append(transition)

        current_cell = next_cell

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
    state_actions = []
    for t in transitions:
        # t['from_cell']
        state_action = {}
        state_action['from_cell'] = t['from_cell']
        state_action['action'] = t['action']
        state_actions.append(state_action)

    return set(state_actions)

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
    actions = valuesByAction.keys()
    values = []
    for a in actions:
        values.append(valuesByAction[a])

    # now sort
    inx_sorted = np.argsort(values)

    # return highest value
    return actions[inx_sorted[-1]]

def get_actions(Q, state):
    valuesByAction = Q[s]
    actions = valuesByAction.keys()
    return set(actions)

def test_on_policy_first_time_mc(n = 1):

    grid = create_grid()
    states = get_states(grid)
    actions = ['L','R','U','D']

    policy = init_random_policy(states, actions)

    # initialize state-action value function
    Q = init_state_action_value_function(states, actions)
    returns = init_state_action_returns(states, actions)

    initial_cell = pick_initial_cell(grid)
    transitions = generate_state_action_episode(grid, initial_cell, policy)

    unique_state_actions = extract_state_actions_visited(transitions)
    unique_cells = extract_states_visited(transitions)




    for state_action in unique_state_actions:
        cell = state_action['from_cell']
        state = to_state(cell)
        action = state_action['action']

        inx = find_first_transition_index(transitions, state_action)
        cumulative_reward = 0
        for i in range(inx,len(transitions)):
            transition = transitions[i]
            cumulative_reward += transition['reward']

        returns[state][action].append(cumulative_reward)
        Q[state][action] = np.mean(returns[state][action])


    # now update policy
    epsilon = 0.001
    for cell in unique_cells:
        state = to_state(grid, cell)
        astar = find_max_action(Q, state)

        possible_actions = get_actions(Q, state)

        for action in possible_actions:
            if action == astar:
                policy[action][state] = 1.0 - epsilon + epsilon / len(possible_actions)
            else:
                policy[action][state] = epsilon / len(possible_actions)

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
