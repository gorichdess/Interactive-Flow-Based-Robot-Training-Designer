import numpy as np
import random

class TerrainEnv:
    def __init__(self, size=20):
        self.size = size
        self.grid = None
        self.start = None
        self.goal = None
        self.position = None
        self.trajectory = [(0,0)]
        self.reached_goal = False
        
    def generate_random(self):
        self.grid = np.zeros((self.size, self.size), dtype=int)
        
        # Set terrain types (adjust probabilities as needed)
        for i in range(self.size):
            for j in range(self.size):
                r = random.random()
                if r < 0.1:
                    self.grid[i, j] = 1  # Rough
                elif r < 0.15:
                    self.grid[i, j] = 3  # Mountain
                elif r < 0.2:
                    self.grid[i, j] = 4  # Swamp
        
        # Set start position (top-left corner)
        self.start = (0, 0)
        self.grid[0, 0] = 0  # Ensure start is empty
        
        # Set goal position (bottom-right corner)
        self.goal = (self.size-1, self.size-1)
        self.grid[self.goal] = 2  # Goal
        
        self.reset()
        
    def reset(self):
        self.position = self.start
        self.trajectory = [(0,0)]
        self.reached_goal = False
        return self.get_state()
    
    def set_size(self,size):
        self.size = size
    
    def get_state(self):
        return self.position
    
    def step(self, action):
        i, j = self.position
        
        # Move based on action
        if action == 0 and i > 0:  # Up
            i -= 1
        elif action == 1 and j < self.size - 1:  # Right
            j += 1
        elif action == 2 and i < self.size - 1:  # Down
            i += 1
        elif action == 3 and j > 0:  # Left
            j -= 1
        # If action would move out of bounds, stay in place
        
        # Update position
        old_pos = self.position
        self.position = (i, j)
        
        # Add to trajectory if position changed
        if self.position != old_pos:
            self.trajectory.append(self.position)
        
        # Check if reached goal
        if self.position == self.goal:
            self.reached_goal = True
            reward = 10.0
            done = True
        else:
            # Penalty based on terrain type
            terrain_type = self.grid[i, j]
            if terrain_type == 0:  # Empty
                reward = -0.1
            elif terrain_type == 1:  # Rough
                reward = -0.5
            elif terrain_type == 3:  # Mountain
                reward = -1.0
            elif terrain_type == 4:  # Swamp
                reward = -0.8
            else:
                reward = -0.1
            done = False
        
        return self.get_state(), reward, done
    
    def get_grid(self):
        return self.grid.copy()
    
    def get_position_from_state(self, state):
        if isinstance(state, tuple):
            return state
        elif hasattr(state, '__len__') and len(state) == 2:
            return tuple(state)
        return self.position  # Fallback