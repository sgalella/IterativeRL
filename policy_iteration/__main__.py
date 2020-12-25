import argparse

import matplotlib.pyplot as plt
import numpy as np

import algorithms as al
from utils import load_grid
from visualization import plot_gridworld, plot_state_values, plot_policy


WIDTH = 50
MARGIN = 2

# Get arguments from command line
parser = argparse.ArgumentParser(description="Policy iteration for gridworld.")
parser.add_argument("filename", help="Path to the gridworld file")
parser.add_argument("goals", help="Goal positions in gridworld", type=int, nargs="*")
parser.add_argument("-m", "--method", help="Iteration method", type=str, default='policy')
parser.add_argument("-g", "--gamma", help="Discount factor", type=float, default=0.9)
parser.add_argument("-t", "--theta", help="Convergence parameter", type=float, default=0.001)
parser.add_argument("-v", "--verbose", help="Print run information", action="store_true")
args = parser.parse_args()

filename = args.filename
gamma = args.gamma
goals = [tuple(args.goals[pos:pos + 2]) for pos in range(0, len(args.goals), 2)]
method = 'Value Iteration' if args.method.lower() == 'value' else 'Policy Iteration'
theta = args.theta

if args.verbose:
    print(f"Filename: {filename}")
    print(f"Gamma: {gamma}")
    print(f"Goals: {goals}")
    print(f"Method: {method}")
    print(f"Theta: {theta}")

grid = load_grid(filename)
has_converged = False  # Checks if policy evaluation has converged
iteration = 0
state_values = al.zero_values(grid)

if method == 'Value Iteration':
    function = al.value_iteration
    policy = np.zeros(grid.shape, dtype=object)
    current_iter = -1
else:
    function = al.policy_evaluation
    policy = al.random_policy(grid, goals)
    current_iter = 0

# Initial plot
plt.ion()
fig, ax = plt.subplots(1, 3, figsize=(8, 4))
plot_gridworld(ax[0], grid)
plot_state_values(ax[1], state_values, iteration)
plot_policy(ax[2], policy, iteration, goals)
while True:
    if not has_converged:
        state_values, has_converged = function(grid, state_values, policy, gamma, theta, goals)
        ax[1].cla()
        plot_state_values(ax[1], state_values, iteration)
        plt.pause(0.001)
    else:
        new_policy = al.policy_improvement(grid, state_values, gamma, goals)
        if (new_policy == policy).all():
            break
        policy = new_policy
        state_values = al.zero_values(grid)
        has_converged = False
        ax[2].cla()
        iteration += 1
        plot_policy(ax[2], policy, iteration + current_iter, goals)
        plt.pause(1)

if method == 'Value Iteration':
    ax[1].set_title(f"$V_{{{iteration}}} = V_{{*}}$")
    ax[2].set_title(r"$\Pi_{" + str(iteration + current_iter) + r"} = \Pi_{{*}}$")
else:
    ax[1].set_title(f"$V_{{{iteration}}} = V_{{*}}$")
    ax[2].set_title(r"$\Pi_{" + str(iteration) + r"} = \Pi_{{*}}$")
plt.ioff()
plt.show()
