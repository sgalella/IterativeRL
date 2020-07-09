import pygame
import numpy as np


WIDTH = 50
MARGIN = 2

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GAMMA = 1
REWARD = -1
THETA = 0.001

BLOCK = 'X'
START = 'S'
GOAL = 'G'

COLOR_FONT = (255, 255, 255)
COLOR_BLOCK = (0, 0, 0)
COLOR_GOAL = (200, 200, 0)
COLOR_STATES = [(200, 200, 0), (200, 150, 0), (200, 100, 0), (200, 50, 0), (200, 0, 0)]


class Gridworld:
    def __init__(self, grid, gamma, reward, theta):
        self.grid = grid
        self.rows = len(grid)
        self.columns = len(grid[0])
        self.size = (WIDTH + MARGIN) * self.columns + MARGIN, (WIDTH + MARGIN) * self.rows + MARGIN
        self.cells = []
        self.start = None
        self.goal = None
        self.gamma = gamma
        self.reward = reward
        self.theta = theta
        self.__initialize_cells()

    def __repr__(self):
        string = ""
        for row in self.cells:
            string += " ".join(str(column) for column in row)
            string += "\n"
        return string

    def __get_cell_values(self):
        values = np.zeros((self.rows, self.columns))
        for row in range(self.rows):
            for column in range(self.columns):
                values[row][column] = self.cells[row][column].value
        return values

    def __initialize_cells(self):
        for row in range(self.rows):
            self.cells.append([])
            for column in range(self.columns):
                self.cells[row].append(Cell(row, column, self.grid[row][column]))
                if self.grid[row][column] == START:
                    self.start = row, column
                elif self.grid[row][column] == GOAL:
                    self.goal = row, column

    def __update_cell_values(self, values):
        for row in range(self.rows):
            for column in range(self.columns):
                self.cells[row][column].value = values[row][column]

    def draw(self):
        for row in range(self.rows):
            for column in range(self.columns):
                self.cells[row][column].draw()

    def run(self):
        values = self.__get_cell_values()
        new_values = np.zeros(values.shape)
        for row in range(self.rows):
            for column in range(self.columns):
                cell = self.cells[row][column]
                if cell.state != GOAL:
                    for (action_row, action_col) in ACTIONS:
                        next_row = action_row + cell.row
                        next_col = action_col + cell.column
                        if next_row < 0 or next_col < 0 or next_row >= self.rows or next_col >= self.columns:
                            new_values[row][column] += 0.25 * (self.reward + self.gamma * cell.value)
                        else:
                            new_values[row][column] += 0.25 * (self.reward + self.gamma * self.cells[next_row][next_col].value)
        self.__update_cell_values(new_values)
        if np.max(abs(new_values - values)) < self.theta:
            return True
        return False


class Cell:
    def __init__(self, row, column, state):
        self.row = row
        self.column = column
        self.value = 0.0
        self.state = state

    def __repr__(self):
        return f"{round(self.value, 2):.1f}"

    def draw(self):
        if self.state == BLOCK:
            pygame.draw.rect(screen, COLOR_BLOCK, ((WIDTH + MARGIN) * self.column + MARGIN,
                                                   (WIDTH + MARGIN) * self.row + MARGIN, WIDTH, WIDTH))
        elif self.state == GOAL:
            pygame.draw.rect(screen, COLOR_GOAL, ((WIDTH + MARGIN) * self.column + MARGIN,
                                                  (WIDTH + MARGIN) * self.row + MARGIN, WIDTH, WIDTH))
        else:
            if self.value >= 0:
                pygame.draw.rect(screen, COLOR_GOAL, ((WIDTH + MARGIN) * self.column + MARGIN,
                                                      (WIDTH + MARGIN) * self.row + MARGIN, WIDTH, WIDTH))
            elif 0 > self.value >= -10:
                pygame.draw.rect(screen, COLOR_STATES[0], ((WIDTH + MARGIN) * self.column + MARGIN,
                                                           (WIDTH + MARGIN) * self.row + MARGIN, WIDTH, WIDTH))
            elif -10 > self.value >= -15:
                pygame.draw.rect(screen, COLOR_STATES[1], ((WIDTH + MARGIN) * self.column + MARGIN,
                                                           (WIDTH + MARGIN) * self.row + MARGIN, WIDTH, WIDTH))
            elif -15 > self.value >= -20:
                pygame.draw.rect(screen, COLOR_STATES[2], ((WIDTH + MARGIN) * self.column + MARGIN,
                                                           (WIDTH + MARGIN) * self.row + MARGIN, WIDTH, WIDTH))
            else:
                pygame.draw.rect(screen, COLOR_STATES[3], ((WIDTH + MARGIN) * self.column + MARGIN,
                                                           (WIDTH + MARGIN) * self.row + MARGIN, WIDTH, WIDTH))

        if self.state != BLOCK:
            text = font.render(f"{round(self.value, 2):.1f}", False, COLOR_FONT)
            if self.value >= 0:
                screen.blit(text, ((WIDTH + MARGIN) * self.column + MARGIN + 14,
                                   (WIDTH + MARGIN) * self.row + MARGIN + 13))  # 13 is distance to center
            elif 0 > self.value >= -10:
                screen.blit(text, ((WIDTH + MARGIN) * self.column + MARGIN + 9,
                                   (WIDTH + MARGIN) * self.row + MARGIN + 13))
            else:
                screen.blit(text, ((WIDTH + MARGIN) * self.column + MARGIN + 4,
                                   (WIDTH + MARGIN) * self.row + MARGIN + 13))


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
        env.draw()
        if not has_converged:
            has_converged = env.run()
            iterations += 1
        pygame.display.flip()
        pygame.display.set_caption(f"Iterations: {iterations}")
        clock.tick(5)
    pygame.quit()


if __name__ == "__main__":
    pygame.init()
    filename = "examples/example1.txt"
    grid = load_grid(filename)
    env = Gridworld(grid, GAMMA, REWARD, THETA)
    screen = pygame.display.set_mode(env.size)
    font = pygame.font.SysFont("Arial", 18)
    main()
