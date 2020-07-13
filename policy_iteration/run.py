import matplotlib.pyplot as plt
from gridworld import Gridworld


GAMMA = 1
REWARD = -1
THETA = 0.001


def main():
    has_converged = False
    plt.ion()
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    iteration = 0
    while not has_converged:
        env.plot(ax, iteration)
        has_converged = env.run()
        iteration += 1
    plt.ioff()
    env.plot(ax, iteration)
    plt.show()


if __name__ == "__main__":
    filename = "examples/example1.txt"
    env = Gridworld(filename, GAMMA, REWARD, THETA)
    main()
