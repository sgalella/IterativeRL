import matplotlib.pyplot as plt
import numpy as np
from utils import draw_arrows


GAMMA = 0.9
REWARD = -1
THETA = 0.001

WIDTH = 50
MARGIN = 2

KEY_TO_ACTION = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
GOALS = [(0, 0), (3, 3)]

MIN_VALUE = -50
MAX_VALUE = 50
CMAP_VALUE = plt.cm.jet
CMAP_VALUE.set_bad("black")
CMAP_POLICY = plt.cm.gray_r
CMAP_POLICY.set_bad("black")


def load_grid(path):
    with open(path, 'r') as file:
        grid = [[int(x) for x in line.strip().split(',')] for line in file]
    return np.array(grid)


def run_policy_evaluation(grid, values, policy):
    new_values = np.zeros(values.shape)
    for row in range(new_values.shape[0]):
        for column in range(new_values.shape[1]):
            if (row, column) not in GOALS:
                actions = policy[row][column]
                p = 1 / len(actions)
                for action in actions:
                    action_row, action_column = KEY_TO_ACTION[action]
                    next_row = row + action_row
                    next_col = column + action_column
                    if next_row < 0 or next_col < 0 or next_row >= new_values.shape[0] or next_col >= new_values.shape[1]:
                        new_values[row][column] += p * (grid[row][column] + GAMMA * values[row][column])
                    else:
                        new_values[row][column] += p * (grid[row][column] + GAMMA * values[next_row][next_col])
    delta = np.max(abs(new_values - values))
    values = new_values
    return (values, delta < THETA)


def get_max_neighbors(values, row, column):
    value_neighbors = np.zeros((4, ))
    for idx, action in enumerate("UDLR"):
        action_row, action_column = KEY_TO_ACTION[action]
        next_row = row + action_row
        next_col = column + action_column
        if next_row < 0 or next_col < 0 or next_row >= values.shape[0] or next_col >= values.shape[1]:
            value_neighbors[idx] = -np.inf
        else:
            value_neighbors[idx] = values[next_row][next_col]
    return (value_neighbors == np.nanmax(value_neighbors)).astype(int).tolist()


def run_policy_improvement(values):
    new_policy = np.zeros(values.shape, dtype=object)
    for row in range(new_policy.shape[0]):
        for column in range(new_policy.shape[1]):
            if (row, column) not in GOALS:
                policy_neighbors = ""
                max_neighbors = get_max_neighbors(values, row, column)
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
    return np.zeros(grid.shape)


def initialize_random_policy(grid):
    policy = np.zeros(grid.shape, dtype=object)
    policy[policy == 0] = "UDLR"
    for (row, column) in GOALS:
        policy[row][column] = ''
    return policy


def plot_gridworld(ax, grid):
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
    ax.imshow(values, cmap=CMAP_VALUE, vmin=MIN_VALUE, vmax=MAX_VALUE)
    for row in range(values.shape[1]):
        for column in range(values.shape[1]):
            ax.text(column, row, f"{values[row][column]:.1f}", va="center", ha="center", color="black", fontsize=12)
    ax.set_title(f"$V_{{{iteration}}}$")
    ax.grid(color="black", linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.draw()


def plot_policy(ax, policy, iteration):
    background = np.zeros(policy.shape)
    ax.imshow(background, cmap=CMAP_POLICY)
    for row in range(policy.shape[0]):
        for column in range(policy.shape[1]):
            if (row, column) not in GOALS:
                draw_arrows(ax, policy[row][column], column, row)
    ax.set_title(r"$\Pi_{" + str(iteration) + "}$")
    ax.grid(color="black", linewidth=1)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.draw()


def main():
    has_converged = False
    iteration = 0
    state_values = initialize_initial_values(grid)
    policy = initialize_random_policy(grid)
    # Initial plot
    plt.ion()
    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    plot_gridworld(ax[0], grid)
    plot_state_values(ax[1], state_values, iteration)
    plot_policy(ax[2], policy, iteration)
    while True:
        if not has_converged:
            state_values, has_converged = run_policy_evaluation(grid, state_values, policy)
            ax[1].cla()
            plot_state_values(ax[1], state_values, iteration)
            plt.pause(0.01)
        else:
            new_policy = run_policy_improvement(state_values)
            if (new_policy == policy).all():
                break
            policy = new_policy
            state_values = initialize_initial_values(grid)
            has_converged = False
            ax[2].cla()
            iteration += 1
            plot_policy(ax[2], policy, iteration)
            plt.pause(1)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    filename = "examples/example1.txt"
    grid = load_grid(filename)
    main()
