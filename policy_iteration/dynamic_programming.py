import numpy as np

KEY_TO_ACTION = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}


def policy_evaluation(grid, values, policy, gamma, theta, goals):
    """
    Runs one sweep of policy evaluation.

    Args:
        grid (np.array): Rewards at each position.
        values (np.array): Values at each position.
        policy (np.array): Policy at each position.
        gamma (float): Discount factor.
        theta (float): Convergence criteria.
        goals (list): Goal positions in the grid.

    Returns:
        tuple: Contains next state-values and convergence boolean.
    """
    new_values = np.zeros(values.shape)
    for row in range(new_values.shape[0]):
        for column in range(new_values.shape[1]):
            if (row, column) not in goals:
                actions = policy[row][column]
                p = 1 / len(actions)
                for action in actions:
                    action_row, action_column = KEY_TO_ACTION[action]
                    next_row = row + action_row
                    next_col = column + action_column
                    if next_row < 0 or next_col < 0 or next_row >= new_values.shape[0] or next_col >= new_values.shape[1]:
                        new_values[row][column] += p * (grid[row][column] + gamma * values[row][column])
                    else:
                        new_values[row][column] += p * (grid[next_row][next_col] + gamma * values[next_row][next_col])
    delta = np.max(abs(new_values - values))
    values = new_values
    return (values, delta < theta)


def value_iteration(grid, values, policy, gamma, theta, goals):
    """
    Runs one sweep of policy evaluation for value iteration.

    Args:
        grid (np.array): Rewards at each position.
        values (np.array): Values at each position.
        policy (np.array): Policy at each position.
        gamma (float): Discount factor.
        theta (float): Convergence criteria.
        goals (list): Goal positions in the grid.

    Returns:
        tuple: Contains next state-values and convergence boolean.
    """
    new_values = np.zeros(values.shape)
    for row in range(new_values.shape[0]):
        for column in range(new_values.shape[1]):
            neighbor_values = []
            if (row, column) not in goals:
                for (action_row, action_column) in KEY_TO_ACTION.values():
                    next_row = row + action_row
                    next_col = column + action_column
                    if next_row < 0 or next_col < 0 or next_row >= new_values.shape[0] or next_col >= new_values.shape[1]:
                        neighbor_values.append(grid[row][column] + gamma * values[row][column])
                    else:
                        neighbor_values.append(grid[next_row][next_col] + gamma * values[next_row][next_col])
                new_values[row][column] = max(neighbor_values)
    delta = np.max(abs(new_values - values))
    return (new_values, delta < theta)


def get_max_neighbors(grid, values, gamma, row, column):
    """
    Gets neighbors positions with highest value.

    Args:
        grid (np.array): Rewards at each position.
        values (np.array): Values at each position.
        gamma (float): Discount factor.
        row (int): Current row position.
        column (int): Current column position.

    Returns:
        list: Four values (UDLR) with 1s in the highest neighbors.
    """
    value_neighbors = np.zeros((4, ))
    for idx, action in enumerate("UDLR"):
        action_row, action_column = KEY_TO_ACTION[action]
        next_row = row + action_row
        next_col = column + action_column
        if next_row < 0 or next_col < 0 or next_row >= values.shape[0] or next_col >= values.shape[1]:
            value_neighbors[idx] = -np.inf
        else:
            value_neighbors[idx] = grid[next_row][next_col] + gamma * values[next_row][next_col]
    return (value_neighbors == np.nanmax(value_neighbors)).astype(int).tolist()


def policy_improvement(grid, values, gamma, goals):
    """
    Runs one sweep of policy improvement.

    Args:
        grid (np.array): Rewards at each position.
        values (np.array): Values at each position.
        gamma (float): Discount factor.
        goals (list): Goal positions in the grid.

    Returns:
        np.array: Updated policy.
    """
    new_policy = np.zeros(values.shape, dtype=object)
    for row in range(new_policy.shape[0]):
        for column in range(new_policy.shape[1]):
            if (row, column) not in goals:
                policy_neighbors = ""
                max_neighbors = get_max_neighbors(grid, values, gamma, row, column)
                up, down, left, right = max_neighbors
                if up:
                    policy_neighbors += "U"
                if down:
                    policy_neighbors += "D"
                if left:
                    policy_neighbors += "L"
                if right:
                    policy_neighbors += "R"
                new_policy[row][column] = policy_neighbors
    return new_policy


def zero_values(grid):
    """
    Sets the state-values array to 0.

    Args:
        grid (np.array): Rewards at each position.

    Returns:
        np.arrays: Arrays of zeros.
    """
    return np.zeros(grid.shape)


def random_policy(grid, goals):
    """
    Initializes policy to be random at each position except in goal.

    Args:
        grid (np.array): Rewards at each position.
        goals (list): Goal positions in the grid.

    Returns:
        np.array: Random policy.
    """
    policy = np.zeros(grid.shape, dtype=object)
    policy[policy == 0] = "UDLR"
    for (row, column) in goals:
        policy[row][column] = ''
    return policy
