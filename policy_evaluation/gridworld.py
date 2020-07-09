import pygame
import numpy as np


WIDTH = 50
MARGIN = 2

ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

BLOCK = 'X'
GOAL = 'G'

COLOR_FONT = (255, 255, 255)
COLOR_BLOCK = (0, 0, 0)
COLOR_GOAL = (200, 200, 0)
COLOR_STATES = [(200, 200, 0), (200, 150, 0), (200, 100, 0), (200, 50, 0), (200, 0, 0)]


class Gridworld:
    def __init__(self, path, gamma, reward, theta):
        self.grid = self.__load_grid(path)
        self.values = np.zeros(self.grid.shape)
        self.rows, self.columns = self.grid.shape
        self.size = (WIDTH + MARGIN) * self.columns + MARGIN, (WIDTH + MARGIN) * self.rows + MARGIN
        self.gamma = gamma
        self.reward = reward
        self.theta = theta

    def __repr__(self):
        return self.values

    def __load_grid(self, path):
        with open(path, 'r') as file:
            grid = [line.strip().split(',') for line in file]
        return np.array(grid)

    def draw(self, screen, font):
        for row in range(self.rows):
            for column in range(self.columns):
                current_value = self.values[row][column]
                current_state = self.grid[row][column]
                if current_state == BLOCK:
                    pygame.draw.rect(screen, COLOR_BLOCK, ((WIDTH + MARGIN) * column + MARGIN,
                                                           (WIDTH + MARGIN) * row + MARGIN, WIDTH, WIDTH))
                elif current_state == GOAL:
                    pygame.draw.rect(screen, COLOR_GOAL, ((WIDTH + MARGIN) * column + MARGIN,
                                                          (WIDTH + MARGIN) * row + MARGIN, WIDTH, WIDTH))
                else:
                    if current_value >= 0:
                        pygame.draw.rect(screen, COLOR_GOAL, ((WIDTH + MARGIN) * column + MARGIN,
                                                              (WIDTH + MARGIN) * row + MARGIN, WIDTH, WIDTH))
                    elif 0 > current_value >= -10:
                        pygame.draw.rect(screen, COLOR_STATES[0], ((WIDTH + MARGIN) * column + MARGIN,
                                                                   (WIDTH + MARGIN) * row + MARGIN, WIDTH, WIDTH))
                    elif -10 > current_value >= -15:
                        pygame.draw.rect(screen, COLOR_STATES[1], ((WIDTH + MARGIN) * column + MARGIN,
                                                                   (WIDTH + MARGIN) * row + MARGIN, WIDTH, WIDTH))
                    elif -15 > current_value >= -20:
                        pygame.draw.rect(screen, COLOR_STATES[2], ((WIDTH + MARGIN) * column + MARGIN,
                                                                   (WIDTH + MARGIN) * row + MARGIN, WIDTH, WIDTH))
                    else:
                        pygame.draw.rect(screen, COLOR_STATES[3], ((WIDTH + MARGIN) * column + MARGIN,
                                                                   (WIDTH + MARGIN) * row + MARGIN, WIDTH, WIDTH))

                if current_state != BLOCK:
                    text = font.render(f"{round(current_value, 2):.1f}", False, COLOR_FONT)
                    if current_value >= 0:
                        screen.blit(text, ((WIDTH + MARGIN) * column + MARGIN + 14,
                                           (WIDTH + MARGIN) * row + MARGIN + 13))  # 13 is distance to center
                    elif 0 > current_value >= -10:
                        screen.blit(text, ((WIDTH + MARGIN) * column + MARGIN + 9,
                                           (WIDTH + MARGIN) * row + MARGIN + 13))
                    else:
                        screen.blit(text, ((WIDTH + MARGIN) * column + MARGIN + 4,
                                           (WIDTH + MARGIN) * row + MARGIN + 13))

    def run(self):
        new_values = np.zeros(self.values.shape)
        for row in range(self.rows):
            for column in range(self.columns):
                if self.grid[row][column] != GOAL:
                    for (action_row, action_col) in ACTIONS:
                        next_row = action_row + row
                        next_col = action_col + column
                        if next_row < 0 or next_col < 0 or next_row >= self.rows or next_col >= self.columns:
                            new_values[row][column] += 0.25 * (self.reward + self.gamma * self.values[row][column])
                        else:
                            new_values[row][column] += 0.25 * (self.reward + self.gamma * self.values[next_row][next_col])
        delta = np.max(abs(new_values - self.values))
        self.values = new_values
        if delta < self.theta:
            return True
        return False
