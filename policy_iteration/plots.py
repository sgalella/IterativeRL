import matplotlib.pyplot as plt
import numpy as np


MIN_VALUE = -100
MAX_VALUE = 100
CMAP_VALUE = plt.cm.jet
CMAP_VALUE.set_bad("black")
CMAP_POLICY = plt.cm.gray_r
CMAP_POLICY.set_bad("black")

ARROW_LEN = 0.4
ARROW_COLOR = "black"
CENTER_OFFSET = 0.05


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
            if grid[row][column] != "*":
                ax.text(column, row, f"{grid[row][column]:.1f}", va="center", ha="center", color="black", fontsize=12)
    ax.set_title("$R$")
    ax.set_xticks(np.arange(-0.5, len(grid), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(grid[0]), 1), minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which="minor", color="black", linewidth=1)
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
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
            if not np.isnan(values[row][column]):
                color = "black" if MIN_VALUE < values[row][column] < MAX_VALUE else "white"
                fontsize = 12 if MIN_VALUE < values[row][column] < MAX_VALUE else 8
                ax.text(column, row, f"{values[row][column]:.1f}", va="center", ha="center", color=color, fontsize=fontsize)
    ax.set_title(f"$V_{{{iteration}}}$")
    ax.set_xticks(np.arange(-0.5, len(values), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(values[0]), 1), minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which="minor", color="black", linewidth=1)
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
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
    ax.set_xticks(np.arange(-0.5, len(policy), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(policy[0]), 1), minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(which="minor", color="black", linewidth=1)
    ax.xaxis.set_ticks_position("none")
    ax.yaxis.set_ticks_position("none")
    plt.draw()


def draw_arrows(ax, policy, row, column):
    """ Draws arrows in cell """
    if policy == "U":
        draw_arrow_U(ax, row, column)
    elif policy == "D":
        draw_arrow_D(ax, row, column)
    elif policy == "L":
        draw_arrow_L(ax, row, column)
    elif policy == "R":
        draw_arrow_R(ax, row, column)
    elif policy == "UD":
        draw_arrow_U(ax, row, column)
        draw_arrow_D(ax, row, column)
    elif policy == "UL":
        draw_arrow_U(ax, row, column)
        draw_arrow_L(ax, row, column)
    elif policy == "UR":
        draw_arrow_U(ax, row, column)
        draw_arrow_R(ax, row, column)
    elif policy == "DL":
        draw_arrow_D(ax, row, column)
        draw_arrow_L(ax, row, column)
    elif policy == "DR":
        draw_arrow_D(ax, row, column)
        draw_arrow_R(ax, row, column)
    elif policy == "LR":
        draw_arrow_L(ax, row, column)
        draw_arrow_R(ax, row, column)
    elif policy == "UDL":
        draw_arrow_U(ax, row, column)
        draw_arrow_D(ax, row, column)
        draw_arrow_L(ax, row, column)
    elif policy == "UDR":
        draw_arrow_U(ax, row, column)
        draw_arrow_D(ax, row, column)
        draw_arrow_R(ax, row, column)
    elif policy == "ULR":
        draw_arrow_U(ax, row, column)
        draw_arrow_L(ax, row, column)
        draw_arrow_R(ax, row, column)
    elif policy == "DLR":
        draw_arrow_D(ax, row, column)
        draw_arrow_L(ax, row, column)
        draw_arrow_R(ax, row, column)
    elif policy == "UDLR":
        draw_arrow_U(ax, row, column)
        draw_arrow_D(ax, row, column)
        draw_arrow_L(ax, row, column)
        draw_arrow_R(ax, row, column)


def draw_arrow_U(ax, row, column):
    """ Draws arrow pointing to north """
    ax.annotate("", xy=(row, column - ARROW_LEN), xytext=(row, column + CENTER_OFFSET), arrowprops=dict(arrowstyle="->", color=ARROW_COLOR))


def draw_arrow_D(ax, row, column):
    """ Draws arrow pointing to south """
    ax.annotate("", xy=(row, column + ARROW_LEN), xytext=(row, column - CENTER_OFFSET), arrowprops=dict(arrowstyle="->", color=ARROW_COLOR))


def draw_arrow_L(ax, row, column):
    """ Draws arrow pointing to west """
    ax.annotate("", xy=(row - ARROW_LEN, column), xytext=(row + CENTER_OFFSET, column), arrowprops=dict(arrowstyle="->", color=ARROW_COLOR))


def draw_arrow_R(ax, row, column):
    """ Draws arrow pointing to east """
    ax.annotate("", xy=(row + ARROW_LEN, column), xytext=(row - CENTER_OFFSET, column), arrowprops=dict(arrowstyle="->", color=ARROW_COLOR))
