import argparse
import matplotlib.pyplot as plt
import numpy as np
from utils import draw_arrows


WIDTH = 50
MARGIN = 2

KEY_TO_ACTION = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}

MIN_VALUE = -100
MAX_VALUE = 100
CMAP_VALUE = plt.cm.jet
CMAP_VALUE.set_bad("black")
CMAP_POLICY = plt.cm.gray_r
CMAP_POLICY.set_bad("black")


def load_grid(path):
    """
    Loads file with the rewards.

    Args:
        path (str): Path to the grid.

    Returns:
        np.array: Array of ints containing the rewards at each position.
    """
    with open(path, 'r') as file:
        grid = [[int(x) for x in line.strip().split(',')] for line in file]
    return np.array(grid)


def run_policy_evaluation(grid, values, policy, gamma, theta, goals):
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


def run_policy_improvement(grid, values, gamma, goals):
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


def initialize_initial_values(grid):
    """
    Initializes the state-values array.

    Args:
        grid (np.array): Rewards at each position.

    Returns:
        np.arrays: Arrays of zeros.
    """
    return np.zeros(grid.shape)


def initialize_random_policy(grid, goals):
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


def plot_gridworld(ax, grid):
    """
    Plots the rewards at each position.

    Args:
        ax (matplotlib.AxesSubplots): Position to plot the gridworld.
        grid (np.array): Rewards at each position.
    """
    ax.imshow(grid, cmap=CMAP_VALUE, vmin=MIN_VALUE, vmax=MAX_VALUE)
    for row in range(grid.shape[0]):
        for column in range(grid.shape[1]):
            ax.text(column, row, f"{grid[row][column]:.1f}", va="center", ha="center", color="black", fontsize=12)
    ax.set_title("$R$")
    ax.grid(color="black", linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.draw()


def plot_state_values(ax, values, iteration):
    """
    Plots the state-value of each position.

    Args:
        ax (matplotlib.AxesSubplots): Position to plot the state values.
        values (np.array): Values at each position.
        iteration (int): Current step of policy improvement.
    """
    ax.imshow(values, cmap=CMAP_VALUE, vmin=MIN_VALUE, vmax=MAX_VALUE)
    for row in range(values.shape[1]):
        for column in range(values.shape[1]):
            color = "black" if MIN_VALUE < values[row][column] < MAX_VALUE else "white"
            fontsize = 12 if MIN_VALUE < values[row][column] < MAX_VALUE else 8
            ax.text(column, row, f"{values[row][column]:.1f}", va="center", ha="center", color=color, fontsize=fontsize)
    ax.set_title(f"$V_{{{iteration}}}$")
    ax.grid(color="black", linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.draw()


def plot_policy(ax, policy, iteration, goals):
    """
    Plots the policy at each position.
    Policy is represented up to four letters: "U" (Up), "D" (Down), "L" (Left) and "R" (Right).
    If a position contains "UR", it means that the policy at the position is to go up and right.

    Args:
        ax (matplotlib.AxesSubplots): Position to plot the state values.
        policy (np.array): Policy at each position.
        iteration (int): Current step of policy improvement.
        goals (list): Goal positions in the grid.
    """
    background = np.zeros(policy.shape)
    ax.imshow(background, cmap=CMAP_POLICY)
    for row in range(policy.shape[0]):
        for column in range(policy.shape[1]):
            if (row, column) not in goals:
                draw_arrows(ax, policy[row][column], column, row)
    ax.set_title(r"$\Pi_{" + str(iteration) + "}$")
    ax.grid(color="black", linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.draw()


def main(parameters):
    """
    Main function.

    Args:
        parameters (dict): Command line arguments.
    """
    gamma = parameters["gamma"]
    theta = parameters["theta"]
    goals = parameters["goals"]
    grid = load_grid(parameters["filename"])

    has_converged = False  # Checks if policy evaluation has converged
    iteration = 0
    state_values = initialize_initial_values(grid)
    policy = initialize_random_policy(grid, goals)

    # Initial plot
    plt.ion()
    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    plot_gridworld(ax[0], grid)
    plot_state_values(ax[1], state_values, iteration)
    plot_policy(ax[2], policy, iteration, goals)
    while True:
        if not has_converged:
            state_values, has_converged = run_policy_evaluation(grid, state_values, policy, gamma, theta, goals)
            ax[1].cla()
            plot_state_values(ax[1], state_values, iteration)
            plt.pause(0.001)
        else:
            new_policy = run_policy_improvement(grid, state_values, gamma, goals)
            if (new_policy == policy).all():
                break
            policy = new_policy
            state_values = initialize_initial_values(grid)
            has_converged = False
            ax[2].cla()
            iteration += 1
            plot_policy(ax[2], policy, iteration, goals)
            plt.pause(1)
    ax[1].set_title(f"$V_{{{iteration}}} = V_{{*}}$")
    ax[2].set_title(r"$\Pi_{" + str(iteration) + r"} = \Pi_{{*}}$")
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Policy iteration for gridworld.")
    parser.add_argument("filename", help="Path to the gridworld file")
    parser.add_argument("goals", help="Goal positions in gridworld", type=int, nargs="*")
    parser.add_argument("-g", "--gamma", help="Discount factor", type=float, default=0.9)
    parser.add_argument("-t", "--theta", help="Convergence parameter", type=float, default=0.001)
    parser.add_argument("-v", "--verbose", help="Print run information", action="store_true")
    args = parser.parse_args()

    # Group arguments to pass to the main function
    params = {
        "filename": args.filename,
        "gamma": args.gamma,
        "theta": args.theta,
        "goals": [tuple(args.goals[pos:pos + 2]) for pos in range(0, len(args.goals), 2)],
    }

    if args.verbose:
        print(f"Filename: {params['filename']}")
        print(f"Goals: {params['goals']}")
        print(f"Gamma: {params['gamma']}")
        print(f"Theta: {params['theta']}")

    main(params)
