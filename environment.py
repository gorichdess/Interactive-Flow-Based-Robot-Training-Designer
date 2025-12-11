import numpy as np
import random
from enum import Enum

class TerrainType(Enum):
    EMPTY = 0
    ROUGH = 1
    GOAL = 2
    MOUNTAIN = 3
    SWAMP = 4
    WALL = 5  
    START = 6

class BaseEnvironment:
    def __init__(self, size=20):
        self.size = size
        self.grid = None
        self.start = None
        self.goal = None
        self.position = None
        self.trajectory = []
        self.reached_goal = False
        self.evaluation_trajectory = []
        
    
    def _is_valid_position(self, pos):
        if not pos or len(pos) != 2:
            return False
        row, col = pos
        if not (0 <= row < self.size and 0 <= col < self.size):
            return False
        # Check if it's not a wall
        if hasattr(self, 'wall_density'):  # For labyrinth
            return self.grid[row, col] != TerrainType.WALL.value
        return True
    
    def generate_random(self):
        raise NotImplementedError
        
    def reset(self):
        self.position = self.start
        self.trajectory = [self.start]
        self.evaluation_trajectory = [self.start]
        self.reached_goal = False

        return self.get_state()
    
    def set_size(self, size):
        self.size = size

    def set_pos(self, start, goal):
        if self._is_valid_position(start):
            self.start = start
        else:
            self.start = (0, 0)
        
        if self._is_valid_position(goal):
            self.goal = goal
        else:
            self.goal = (self.size - 1, self.size - 1)
        # Reset position to new start
        self.position = self.start
        
    def get_state(self):
        raise NotImplementedError
        
    def step(self, action):
        raise NotImplementedError
        
    def get_grid(self):
        return self.grid.copy()
    
    def get_position_from_state(self, state):
        if isinstance(state, tuple):
            if len(state) >= 2:
                return (state[-2], state[-1])
            elif len(state) == 2:
                return state
        elif hasattr(state, '__len__') and len(state) == 2:
            return tuple(state)
        return self.position

class TerrainEnv(BaseEnvironment):
    def __init__(self, size=20):
        super().__init__(size)
        
    def generate_random(self):
        self.grid = np.zeros((self.size, self.size), dtype=int)
        
        for i in range(self.size):
            for j in range(self.size):
                r = random.random()
                if r < 0.1:
                    self.grid[i, j] = TerrainType.ROUGH.value
                elif r < 0.15:
                    self.grid[i, j] = TerrainType.MOUNTAIN.value
                elif r < 0.2:
                    self.grid[i, j] = TerrainType.SWAMP.value
        
        # Use existing start/goal if they exist, otherwise defaults
        if self.start is None:
            self.start = (0, 0)
        if self.goal is None:
            self.goal = (self.size-1, self.size-1)
        
        # Update grid markers
        self.grid[self.start[0], self.start[1]] = TerrainType.START.value
        self.grid[self.goal[0], self.goal[1]] = TerrainType.GOAL.value
        
        self.reset()
        
    def get_state(self, simplified=False):
        if simplified:
            return self.get_simplified_state()
        
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
    
    def get_simplified_state(self):
        i, j = self.position
        
        state_features = [
            i,
            j,
            abs(i - self.goal[0]),
            abs(j - self.goal[1]),
        ]
        
        for di, dj in [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.size and 0 <= nj < self.size:
                state_features.append(self.grid[ni, nj])
            else:
                state_features.append(-1)
        
        return tuple(state_features)
    
    def step(self, action):
        i, j = self.position
        
        old_pos = self.position
        
        if action == 0 and i > 0:
            i -= 1
        elif action == 1 and j < self.size - 1:
            j += 1
        elif action == 2 and i < self.size - 1:
            i += 1
        elif action == 3 and j > 0:
            j -= 1
        else:
            return self.get_state(), -1.0, False
        
        self.position = (i, j)
        
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
        
        if terrain_type == TerrainType.ROUGH.value:
            reward = -0.3
        elif terrain_type == TerrainType.MOUNTAIN.value:
            reward = -0.5
        elif terrain_type == TerrainType.SWAMP.value:
            reward = -0.4
        else:
            reward = -0.1
        
        old_distance = abs(old_pos[0] - self.goal[0]) + abs(old_pos[1] - self.goal[1])
        new_distance = abs(i - self.goal[0]) + abs(j - self.goal[1])
        
        if new_distance < old_distance:
            reward += 0.2
        elif new_distance > old_distance:
            reward -= 0.2
        
        done = False
        
        return self.get_state(), reward, done

class LabyrinthEnv(BaseEnvironment):
    def __init__(self, size=20):
        super().__init__(size)
        self.wall_density = 0.2
        
    def generate_random(self):
        self.grid = np.zeros((self.size, self.size), dtype=int)
        
        # Створюємо стіни
        for i in range(self.size):
            for j in range(self.size):
                if random.random() < self.wall_density:
                    self.grid[i, j] = TerrainType.WALL.value
        
        # Стартова позиція
        if self.start is None:
            self.start = (0, 0)
        if self.goal is None:
            self.goal = (self.size-1, self.size-1)

        # Find a valid start position (not a wall)
        while self.grid[self.start[0], self.start[1]] == TerrainType.WALL.value:
            self.start = (random.randint(0, self.size-1), random.randint(0, self.size-1))
        self.grid[self.start[0], self.start[1]] = TerrainType.START.value
    
        attempts = 0
        # Find a valid goal position (not a wall and not start)
        while (self.grid[self.goal[0], self.goal[1]] == TerrainType.WALL.value or 
               self.goal == self.start) and attempts < 100:
            self.goal = (random.randint(0, self.size-1), random.randint(0, self.size-1))
            attempts += 1
        self.grid[self.goal[0], self.goal[1]] = TerrainType.GOAL.value
        
        # Ensure path exists
        self.ensure_path()
        
        self.reset()
        
    def ensure_path(self):
        # Simple path creation algorithm
        current = self.start
        path_positions = [current]
        
        while current != self.goal:
            dx = self.goal[0] - current[0]
            dy = self.goal[1] - current[1]
            
            possible_moves = []
            if dx > 0:
                possible_moves.append((1, 0))  # down
            elif dx < 0:
                possible_moves.append((-1, 0))  # up
            if dy > 0:
                possible_moves.append((0, 1))  # right
            elif dy < 0:
                possible_moves.append((0, -1))  # left
            
            if not possible_moves:
                break
                
            # Choose a move
            move = random.choice(possible_moves)
            next_cell = (current[0] + move[0], current[1] + move[1])
            
            # Check bounds
            if not (0 <= next_cell[0] < self.size and 0 <= next_cell[1] < self.size):
                break
            
            # Clear wall if needed
            if self.grid[next_cell[0], next_cell[1]] == TerrainType.WALL.value:
                self.grid[next_cell[0], next_cell[1]] = TerrainType.EMPTY.value
            
            current = next_cell
            path_positions.append(current)
            
            # Avoid infinite loops
            if len(path_positions) > self.size * 2:
                break
    
    def get_state(self):
        i, j = self.position
        
        state_features = [
            i,
            j,
            abs(i - self.goal[0]),
            abs(j - self.goal[1]),
        ]
        
        # Check 4 directions for walls
        for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < self.size and 0 <= nj < self.size:
                if self.grid[ni, nj] == TerrainType.WALL.value:
                    state_features.append(1)  # Wall
                else:
                    state_features.append(0)  # Free space
            else:
                state_features.append(1)  # Boundary - treat as wall
        
        return tuple(state_features)
    
    def step(self, action):
        i, j = self.position
        
        old_pos = self.position
        
        # Calculate new position
        new_i, new_j = i, j
        
        if action == 0:  # Up
            new_i = i - 1
        elif action == 1:  # Right
            new_j = j + 1
        elif action == 2:  # Down
            new_i = i + 1
        elif action == 3:  # Left
            new_j = j - 1
        
        # Check bounds and walls
        if (0 <= new_i < self.size and 0 <= new_j < self.size and
            self.grid[new_i, new_j] != TerrainType.WALL.value):
            i, j = new_i, new_j
        else:
            # Penalize trying to move into wall
            return self.get_state(), -2.0, False
        
        self.position = (i, j)
        
        if self.position != old_pos:
            self.trajectory.append(self.position)
            self.evaluation_trajectory.append(self.position)
        
        # Reward
        if self.position == self.goal:
            self.reached_goal = True
            reward = 100.0
            done = True
            return self.get_state(), reward, done
        
        self.reached_goal = False
        
        # Base reward for step
        reward = -0.1
        
        # Bonus for moving toward goal
        old_distance = abs(old_pos[0] - self.goal[0]) + abs(old_pos[1] - self.goal[1])
        new_distance = abs(i - self.goal[0]) + abs(j - self.goal[1])
        
        if new_distance < old_distance:
            reward += 0.3
        elif new_distance > old_distance:
            reward -= 0.2
        
        done = False
        
        return self.get_state(), reward, done