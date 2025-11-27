import numpy as np
import random

class TerrainEnv:
    def __init__(self, size=20, terrain_map=None):
        self.size = size
        self.external_map = terrain_map
        self.reset()

    def set_size(self,size):
        self.size = size

    def reset(self):
        self.grid = np.zeros((self.size, self.size)) if self.external_map is None else np.copy(self.external_map)
        self.start = (0, 0)
        self.goal = (self.size - 1, self.size - 1)
        self.grid[self.goal] = 2
        self.pos = self.start
        self.trajectory = [self.start]
        return self.pos

    def generate_random(self):
        if self.external_map is None:
            for x in range(self.size):
                for y in range(self.size):
                    if (x, y) not in [self.start, self.goal]:
                        rnd = random.random()
                        if rnd < 0.05:
                            self.grid[x, y] = 3  # Mountain
                        elif rnd < 0.1:
                            self.grid[x, y] = 4  # Swamp
                        elif rnd < 0.15:
                            self.grid[x, y] = 1  # Rough

        self.pos = self.start
        self.trajectory = [self.start]
        return self.pos 

    def step(self, action):
        x, y = self.pos
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        dx, dy = moves[action]
        new_x, new_y = x + dx, y + dy

        if 0 <= new_x < self.size and 0 <= new_y < self.size:
            self.pos = (new_x, new_y)
            self.trajectory.append(self.pos)

        cell_type = self.grid[self.pos]
        reward = -0.1
        if self.pos == self.goal:
            reward = 10
            done = True
        elif cell_type == 1:
            reward = -1
            done = False
        elif cell_type == 3:
            reward = -2
            done = False
        elif cell_type == 4:
            reward = -0.5
            done = False
        else:
            done = False

        return self.pos, reward, done  
    
    def get_grid(self):
        return self.grid