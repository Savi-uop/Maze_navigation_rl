import numpy as np
import matplotlib.pyplot as plt

class MazeEnv:
    def __init__(self):
        self.maze = np.array([
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 1, 1, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
        ])
        self.start = (0, 0)
        self.goal = (9, 8)
        self.position = self.start

    def reset(self):
        self.position = self.start
        return self.position

    def step(self, action):
        moves = {
            0: (-1, 0),  # up
            1: (1, 0),   # down
            2: (0, -1),  # left
            3: (0, 1)    # right
        }

        new_position = (
            self.position[0] + moves[action][0],
            self.position[1] + moves[action][1]
        )

        if (0 <= new_position[0] < self.maze.shape[0] and
                0 <= new_position[1] < self.maze.shape[1] and
                self.maze[new_position] == 0):
            self.position = new_position
            reward = 1 if self.position == self.goal else -0.1
            done = self.position == self.goal
        else:
            reward = -1
            done = False

        return self.position, reward, done

    def render(self):
        maze_copy = self.maze.copy()
        maze_copy[self.start] = 0.5
        maze_copy[self.goal] = 0.8
        maze_copy[self.position] = 0.6
        plt.imshow(maze_copy, cmap='gray')
        plt.show()
