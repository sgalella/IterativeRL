import argparse

import matplotlib.pyplot as plt
import numpy as np

from algorithms import value_iteration, policy_improvement, zero_values
from utils import load_grid
from visualization import plot_gridworld, plot_state_values, plot_policy


WIDTH = 50
MARGIN = 2


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
    state_values = zero_values(grid)
    policy = np.zeros(grid.shape, dtype=object)

    # Initial plot
    plt.ion()
    fig, ax = plt.subplots(1, 3, figsize=(8, 4))
    plot_gridworld(ax[0], grid)
    plot_state_values(ax[1], state_values, iteration)
    plot_policy(ax[2], policy, iteration, goals)
    while True:
        if not has_converged:
            state_values, has_converged = value_iteration(grid, state_values, policy, gamma, theta, goals)
            ax[1].cla()
            plot_state_values(ax[1], state_values, iteration)
            plt.pause(0.001)
        else:
            new_policy = policy_improvement(grid, state_values, gamma, goals)
            if (new_policy == policy).all():
                break
            policy = new_policy
            state_values = zero_values(grid)
            has_converged = False
            ax[2].cla()
            iteration += 1
            plot_policy(ax[2], policy, iteration - 1, goals)
            plt.pause(1)
    ax[1].set_title(f"$V_{{{iteration}}} = V_{{*}}$")
    ax[2].set_title(r"$\Pi_{" + str(iteration - 1) + r"} = \Pi_{{*}}$")
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
