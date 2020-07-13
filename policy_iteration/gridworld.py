import matplotlib.pyplot as plt
import numpy as np
from utils import draw_arrows


WIDTH = 50
MARGIN = 2

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

BLOCK = 'X'
GOAL = 'G'

MIN_VALUE = -50
MAX_VALUE = 50
CMAP_VALUE = plt.cm.jet
CMAP_VALUE.set_bad("black")
CMAP_POLICY = plt.cm.gray_r
CMAP_POLICY.set_bad("black")


class Gridworld:
    """ MDP Environment """
    def __init__(self, path, gamma, reward, theta):
        self.grid = self.__load_grid(path)
        self.values = np.zeros(self.grid.shape)
        self.values[self.grid == BLOCK] = np.nan
        self.policy = np.zeros(self.grid.shape)
        self.policy[self.grid == BLOCK] = np.nan
        self.rows, self.columns = self.grid.shape
        self.size = (WIDTH + MARGIN) * self.columns + MARGIN, (WIDTH + MARGIN) * self.rows + MARGIN
        self.gamma = gamma
        self.reward = reward
        self.theta = theta

    def __repr__(self):
        """ Print values for each cell """
        return self.values

    def __load_grid(self, path):
        """
        Loads the grid stored in path.

        Args:
            path (str): Path to txt file with the gridworld.

        Returns:
            np.array: Loaded grid.
        """
        with open(path, 'r') as file:
            grid = [line.strip().split(',') for line in file]
        return np.array(grid)

    def __get_max_neighbors(self, neighbors_array, row, column):
        """
        Finds the neighbor with the maximum state-value function.

        Args:
            neighbors_array (np.array): Paded rray with the neighbors values.
            row (int): Current row.
            column (int): Current column.

        Returns:
            [list]: 1 if neighbor is max else 0.
        """
        neighbors = np.array([neighbors_array[row][column + 1], neighbors_array[row + 2][column + 1], 
                              neighbors_array[row + 1][column], neighbors_array[row + 1][column + 2]])
        return (neighbors == np.nanmax(neighbors)).astype(int).tolist()

    def plot(self, ax, iteration):
        """
        Plots the state-value function and policy.

        Args:
            ax (numpy.ndarray): Subplots axis
        """
        neighbors_array = np.round(np.pad(self.values, 1, mode="constant", constant_values=-np.inf), 2)
        ax[0].imshow(self.values, cmap=CMAP_VALUE, vmin=MIN_VALUE, vmax=MAX_VALUE)
        ax[1].imshow(self.policy, cmap=CMAP_POLICY)
        for row in range(self.rows):
            for column in range(self.columns):
                if not np.isnan(self.values[row][column]):
                    ax[0].text(column, row, f"{self.values[row][column]:.1f}", va="center", ha="center", color="black", fontsize=12)
                    max_neighbors = self.__get_max_neighbors(neighbors_array, row, column)
                    if self.grid[row][column] != GOAL:
                        draw_arrows(max_neighbors, ax[1], column, row)
        ax[0].set_title("$V_{" + str(iteration) + "}$")
        ax[1].set_title("$\Pi_{" + str(iteration) + "}$")
        ax[0].grid(color="black", linewidth=1)
        ax[1].grid(color="black", linewidth=1)
        ax[0].set_xticks([])
        ax[0].set_yticks([])
        ax[1].set_xticks([])
        ax[1].set_yticks([])
        plt.draw()
        plt.pause(0.01)
        ax[0].cla()
        ax[1].cla()

    def run(self):
        """ Run the policy iteration in the gridworld """
        new_values = np.zeros(self.grid.shape)
        new_values[self.grid == BLOCK] = np.nan
        for row in range(self.rows):
            for column in range(self.columns):
                if self.grid[row][column] != GOAL and self.grid[row][column] != BLOCK:
                    for (action_row, action_col) in ACTIONS:
                        next_row = action_row + row
                        next_col = action_col + column
                        if (next_row < 0 or next_col < 0 or next_row >= self.rows or next_col >= self.columns
                                or self.grid[next_row][next_col] == BLOCK):
                            new_values[row][column] += 0.25 * (self.reward + self.gamma * self.values[row][column])
                        else:
                            new_values[row][column] += 0.25 * (self.reward + self.gamma * self.values[next_row][next_col])
        delta = np.max(abs(new_values - self.values))
        self.values = new_values
        if delta < self.theta:
            return True
        return False
