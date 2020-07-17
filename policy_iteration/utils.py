import numpy as np


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
