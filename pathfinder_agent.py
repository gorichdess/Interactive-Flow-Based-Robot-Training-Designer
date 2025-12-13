import numpy as np
import heapq
from collections import deque
import time

class PathFinderAgent:
    def __init__(self, algorithm='astar'):
        self.algorithm = algorithm
        self.path = []
        self.current_path_index = 0
        self.name = "PathFinder"
        
        self.stats = {
            'paths_found': 0,
            'average_path_length': 0,
            'search_time': 0,
            'nodes_expanded': 0,
            'max_depth': 0
        }
    
    def find_path(self, grid, start, goal):
        start_time = time.time()
        
        # Reset stats
        self.stats['nodes_expanded'] = 0
        self.stats['max_depth'] = 0
        
        if self.algorithm == 'astar':
            path = self._a_star_search(grid, start, goal)
        elif self.algorithm == 'bfs':
            path = self._bfs_search(grid, start, goal)
        elif self.algorithm == 'dfs':
            path = self._dfs_search(grid, start, goal)
        else:
            path = self._a_star_search(grid, start, goal)
        
        search_time = time.time() - start_time
        self.stats['search_time'] = search_time
        
        if path:
            self.path = path
            self.current_path_index = 0
            self.stats['paths_found'] += 1
            self.stats['average_path_length'] = (
                (self.stats['average_path_length'] * (self.stats['paths_found'] - 1) + len(path)) 
                / self.stats['paths_found'] if self.stats['paths_found'] > 1 else len(path)
            )
            
        return path
    
    def _a_star_search(self, grid, start, goal):
        def heuristic(a, b):
            # Manhattan distance
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        def get_terrain_cost(cell_value):
            # Terrain cost mapping
            if cell_value == 0:  # Normal terrain
                return 1.0
            elif cell_value == 1:  # Difficult terrain
                return 2.0
            elif cell_value == 3:  # Water
                return 3.0
            elif cell_value == 4:  # Swamp
                return 4.0
            elif cell_value == 5:  # Wall (unpassable)
                return float('inf')
            else:
                return 1.0
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        # Possible moves: up, right, down, left
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        while open_set:
            current_f, current = heapq.heappop(open_set)
            self.stats['nodes_expanded'] += 1
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for dx, dy in moves:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check bounds
                if not (0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]):
                    continue
                
                # Check if passable
                if grid[neighbor[0], neighbor[1]] == 5:  # Wall
                    continue
                
                # Calculate cost based on terrain
                terrain_cost = get_terrain_cost(grid[neighbor[0], neighbor[1]])
                if terrain_cost == float('inf'):
                    continue
                
                tentative_g_score = g_score[current] + terrain_cost
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None
    
    def _bfs_search(self, grid, start, goal):
        if grid[start[0], start[1]] == 5 or grid[goal[0], goal[1]] == 5:
            return None
        
        # Стоимость разных типов территории
        def terrain_cost(cell):
            if cell == 5:  # Стена
                return float('inf')
            elif cell == 1:  # Rough
                return 3
            elif cell == 3:  # Mountain
                return 5
            elif cell == 4:  # Swamp
                return 4
            else:
                return 1
        
        # Приоритетная очередь для лучшего пути
        pq = []
        heapq.heappush(pq, (0, start))
        
        cost_so_far = {start: 0}
        came_from = {start: None}
        
        directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        
        while pq:
            current_cost, current = heapq.heappop(pq)
            self.stats['nodes_expanded'] += 1
            
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            
            for dx, dy in directions:
                nx, ny = current[0] + dx, current[1] + dy
                neighbor = (nx, ny)
                
                if 0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1]:
                    cell_cost = terrain_cost(grid[nx, ny])
                    if cell_cost == float('inf'):
                        continue
                    
                    new_cost = current_cost + cell_cost
                    
                    if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                        cost_so_far[neighbor] = new_cost
                        came_from[neighbor] = current
                        heapq.heappush(pq, (new_cost, neighbor))
        
        return None
    
    def _dfs_search(self, grid, start, goal):
        if grid[start[0], start[1]] == 5 or grid[goal[0], goal[1]] == 5:
            return None
        
        stack = [(start, [start])]
        visited = np.zeros(grid.shape, dtype=bool)
        visited[start] = True
        
        # Функция для получения стоимости клетки
        def get_cost(x, y):
            cell = grid[x, y]
            if cell == 5:  # Стена
                return 1000
            elif cell == 1:  # Rough
                return 5
            elif cell == 3:  # Mountain
                return 10
            elif cell == 4:  # Swamp
                return 8
            else:  # Пустая
                return 1
        
        max_depth = grid.shape[0] * grid.shape[1]
        
        while stack:
            current, path = stack.pop()
            self.stats['nodes_expanded'] += 1
            self.stats['max_depth'] = max(self.stats['max_depth'], len(path))
            
            if len(path) > max_depth * 2:  # Ограничиваем глубину
                continue
            
            if current == goal:
                return path
            
            x, y = current
            
            # Получаем всех соседей с их стоимостью
            neighbors = []
            for dx, dy in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
                nx, ny = x + dx, y + dy
                
                if (0 <= nx < grid.shape[0] and 0 <= ny < grid.shape[1] and
                    not visited[nx, ny]):
                    
                    # Проверяем, не стена ли
                    if grid[nx, ny] != 5:
                        cost = get_cost(nx, ny)
                        
                        # Эвристика: расстояние до цели
                        dist_to_goal = abs(nx - goal[0]) + abs(ny - goal[1])
                        
                        # Приоритет: сначала дешевые клетки, ближе к цели
                        priority = cost * 0.7 + dist_to_goal * 0.3
                        
                        neighbors.append((priority, (nx, ny)))
            
            # Сортируем соседей по приоритету (лучшие первыми)
            neighbors.sort(reverse=True)  # reverse=True потому что pop() берет последний
            
            for _, neighbor in neighbors:
                visited[neighbor] = True
                stack.append((neighbor, path + [neighbor]))
        
        return None
    
    def get_next_action(self, current_position):
        if not self.path or self.current_path_index >= len(self.path) - 1:
            return None
        
        # Find the closest point in the path to current position
        min_dist = float('inf')
        closest_idx = self.current_path_index
        
        # Start searching from current index forward
        for i in range(self.current_path_index, len(self.path)):
            pos = self.path[i]
            dist = abs(pos[0] - current_position[0]) + abs(pos[1] - current_position[1])
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        self.current_path_index = closest_idx
        
        if self.current_path_index >= len(self.path) - 1:
            return None
        
        current = self.path[self.current_path_index]
        next_pos = self.path[self.current_path_index + 1]
        
        dx = next_pos[0] - current[0]
        dy = next_pos[1] - current[1]
        
        # Map movement to action
        if dx == -1 and dy == 0:
            action = 0  # Up
        elif dx == 0 and dy == 1:
            action = 1  # Right
        elif dx == 1 and dy == 0:
            action = 2  # Down
        elif dx == 0 and dy == -1:
            action = 3  # Left
        else:
            # If not adjacent, try to move toward the next position
            if dx < 0:
                action = 0  # Up
            elif dx > 0:
                action = 2  # Down
            elif dy > 0:
                action = 1  # Right
            elif dy < 0:
                action = 3  # Left
            else:
                action = None
        
        if action is not None:
            self.current_path_index += 1
        
        return action
    
    def predict_action(self, state):
        if isinstance(state, tuple) and len(state) >= 2:
            # Extract current position from state
            current_pos = (state[-2], state[-1])
            return self.get_next_action(current_pos)
        return None
    
    def reset(self):
        self.path = []
        self.current_path_index = 0
        self.stats['nodes_expanded'] = 0
        self.stats['max_depth'] = 0
    
    def get_statistics(self):
        return self.stats.copy()
    
    def visualize_path(self, grid):
        if not self.path:
            return grid.copy()
        
        visualization = grid.copy()
        for i, (x, y) in enumerate(self.path):
            if i == 0:
                visualization[x, y] = 6  # Start
            elif i == len(self.path) - 1:
                visualization[x, y] = 2  # Goal
            else:
                visualization[x, y] = 7  # Path
        
        return visualization
    
    def get_path_length(self):
        return len(self.path) if self.path else 0