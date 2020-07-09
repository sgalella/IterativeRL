import pygame
from gridworld import Gridworld


GAMMA = 1
REWARD = -1
THETA = 0.001


def load_grid(path):
    with open(path, 'r') as file:
        grid = [line.strip().split(',') for line in file]
    return grid


def main():
    iterations = 0
    has_converged = False
    pygame.display.set_caption(f"Iterations: {iterations}")
    clock = pygame.time.Clock()
    is_running = True
    while is_running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                is_running = False
        env.draw(screen, font)
        if not has_converged:
            has_converged = env.run()
            iterations += 1
        pygame.display.flip()
        pygame.display.set_caption(f"Iterations: {iterations}")
        clock.tick(10)
    pygame.quit()


if __name__ == "__main__":
    pygame.init()
    filename = "examples/example1.txt"
    env = Gridworld(filename, GAMMA, REWARD, THETA)
    screen = pygame.display.set_mode(env.size)
    font = pygame.font.SysFont("Arial", 18)
    main()
