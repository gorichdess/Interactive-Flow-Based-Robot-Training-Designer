# environment.py
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
        self.evaluation_trajectory = []  # Додайте це
        
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
        self.grid[self.goal[0]][self.goal[1]] = 2
        
        self.reset()
        
    def reset(self):
        self.position = self.start
        self.trajectory = [self.start]
        self.evaluation_trajectory = [self.start]  # Додайте це
        self.reached_goal = False
        return self.get_state()
    
    def set_size(self, size):
        self.size = size
    
    def get_simplified_state(self):
        i, j = self.position
        
        # Використовуємо тільки ключові характеристики
        state_features = [
            i,  # X позиція
            j,  # Y позиція
            abs(i - self.goal[0]),  # Відстань до цілі по X
            abs(j - self.goal[1]),  # Відстань до цілі по Y
        ]
        
        # Додаємо інформацію про територію навколо (обмежено)
        for di, dj in [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.size and 0 <= nj < self.size:
                state_features.append(self.grid[ni, nj])
            else:
                state_features.append(-1)  # Межа
        
        return tuple(state_features)

    def get_state(self, simplified=False):
        if simplified:
            return self.get_simplified_state()
        
        # Оригінальний метод для RL
        i, j = self.position
        
        local_view = []
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                ni, nj = i + di, j + dj
                if 0 <= ni < self.size and 0 <= nj < self.size:
                    local_view.append(self.grid[ni, nj])
                else:
                    local_view.append(-1)
        
        distance_to_goal = abs(i - self.goal[0]) + abs(j - self.goal[1])
        local_view.append(distance_to_goal)
        local_view.append(i)
        local_view.append(j)
        
        return tuple(local_view)
    
    def step(self, action):
        i, j = self.position
        
        old_pos = self.position
        
        # Move based on action
        if action == 0 and i > 0:  # Up
            i -= 1
        elif action == 1 and j < self.size - 1:  # Right
            j += 1
        elif action == 2 and i < self.size - 1:  # Down
            i += 1
        elif action == 3 and j > 0:  # Left
            j -= 1
        else:
            return self.get_state(), -1.0, False  # НЕ встановлюємо reached_goal!
        
        self.position = (i, j)
        
        # Always record trajectory when position changes
        if self.position != old_pos:
            self.trajectory.append(self.position)
            self.evaluation_trajectory.append(self.position)
        
        terrain_type = self.grid[i, j]
        
        if self.position == self.goal:
            self.reached_goal = True
            reward = 100.0
            done = True
            return self.get_state(), reward, done
        else:
            self.reached_goal = False 
        
        if terrain_type == 1:  # Rough
            reward = -0.3
        elif terrain_type == 3:  # Mountain
            reward = -0.5
        elif terrain_type == 4:  # Swamp
            reward = -0.4
        else:  # Empty
            reward = -0.1
        
        old_distance = abs(old_pos[0] - self.goal[0]) + abs(old_pos[1] - self.goal[1])
        new_distance = abs(i - self.goal[0]) + abs(j - self.goal[1])
        
        if new_distance < old_distance:
            reward += 0.2
        elif new_distance > old_distance:
            reward -= 0.2
        
        done = False
        
        return self.get_state(), reward, done
    
    def get_grid(self):
        return self.grid.copy()
    
    def get_position_from_state(self, state):
        if isinstance(state, tuple):
            # Останні 2 значення - це позиція
            if len(state) >= 2:
                return (state[-2], state[-1])
            elif len(state) == 2:
                return state
        elif hasattr(state, '__len__') and len(state) == 2:
            return tuple(state)
        return self.position  # Fallback