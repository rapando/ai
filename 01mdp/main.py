# Markov Decision Process (MDP)
# The Belman equations adapted to Reinforced Learning
# R is the Reward Matrix for each state
# 0 means no path
# 1 means there is a path
# 100 means destination (highest reward - C)

# The Bellman Equation:
# Q[current_state, action] = R[current_state, action] + gamma * MaxValue


import numpy as ql


def get_possible_actions(state):
    '''the possible "a" actions when agent is in a given state'''
    current_state_row = R[state, ]
    possible_act = ql.where(current_state_row > 0)[1]
    return possible_act


def get_action_choice(possible_actions):
    '''out of the possible actions, choose one at random'''
    next_action = int(ql.random.choice(possible_actions, 1))
    return next_action


def reward(current_state, action, gamma):
    ''' reward system '''
    max_state = ql.where(Q[action, ] == ql.max(Q[action, ]))[1]

    if max_state.shape[0] > 1:
        max_state = int(ql.random.choice(max_state, size=1))
    else:
        max_state = int(max_state)

    max_value = Q[action, max_state]
    Q[current_state, action] = R[current_state, action] + gamma * max_value


if __name__ == "__main__":
    # R is the reward matrix
    R = ql.matrix([
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 100, 1, 0, 0],
        [0, 1, 1, 0, 1, 0],
        [1, 0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0, 0]
    ])

    # Q is the learning matrix set to 0 but similar structure as R
    Q = ql.matrix(ql.zeros([6, 6]))
    # gamma means the system has 80% chance of being correct every time
    gamma = 0.8

    # note that 0-> A and 5->F
    # agent_s_state = 1
    # possible_actions = get_possible_actions(agent_s_state)
    # next_action = get_action_choice(possible_actions)
    # reward(agent_s_state, next_action, gamma)

    print("\n\n-->Q: \n", Q)

    for i in range(50000):
        current_state = ql.random.randint(0, int(Q.shape[0]))
        possible_actions = get_possible_actions(current_state)
        action = get_action_choice(possible_actions)
        reward(current_state, action, gamma)

    print("\n\n-->Normed Q: \n", Q)
