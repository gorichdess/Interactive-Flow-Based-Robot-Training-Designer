import numpy as np
import heapq
from collections import deque

class PathFinderAgent:
    
    def __init__(self, algorithm='astar'):
        self.algorithm = algorithm  # 'astar', 'bfs', 'dfs'
        self.path = []
        self.current_path_index = 0
        self.name = "PathFinder"
        
        # Статистика
        self.stats = {
            'paths_found': 0,
            'average_path_length': 0,
            'search_time': 0
        }
    
    def find_path(self, grid, start, goal):
        import time
        start_time = time.time()
        
        if self.algorithm == 'astar':
            path = self._a_star_search(grid, start, goal)
        elif self.algorithm == 'bfs':
            path = self._bfs_search(grid, start, goal)
        elif self.algorithm == 'dfs':
            path = self._dfs_search(grid, start, goal)
        else:
            path = self._a_star_search(grid, start, goal)
        
        search_time = time.time() - start_time
        
        if path:
            self.path = path
            self.current_path_index = 0
            self.stats['paths_found'] += 1
            self.stats['average_path_length'] = (
                (self.stats['average_path_length'] * (self.stats['paths_found'] - 1) + len(path)) 
                / self.stats['paths_found'] if self.stats['paths_found'] > 1 else len(path)
            )
            self.stats['search_time'] = search_time
            
        return path
    
    def _a_star_search(self, grid, start, goal):
        def heuristic(a, b):
            return abs(a[0] - b[0]) + abs(a[1] - b[1])
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {start: 0}
        f_score = {start: heuristic(start, goal)}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # Реконструкція шляху
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Перевірка меж і перешкод (0 - вільна клітинка, 5 - стіна)
                if (0 <= neighbor[0] < grid.shape[0] and 
                    0 <= neighbor[1] < grid.shape[1] and
                    grid[neighbor[0], neighbor[1]] != 5):  # 5 = WALL
                    
                    tentative_g_score = g_score[current] + 1
                    
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return None  # Шлях не знайдено
    
    def _bfs_search(self, grid, start, goal):
        queue = deque([start])
        visited = {start}
        came_from = {start: None}
        
        while queue:
            current = queue.popleft()
            
            if current == goal:
                # Реконструкція шляху
                path = []
                while current is not None:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if (0 <= neighbor[0] < grid.shape[0] and 
                    0 <= neighbor[1] < grid.shape[1] and
                    grid[neighbor[0], neighbor[1]] != 5 and  # 5 = WALL
                    neighbor not in visited):
                    
                    visited.add(neighbor)
                    came_from[neighbor] = current
                    queue.append(neighbor)
        
        return None
    
    def _dfs_search(self, grid, start, goal, max_depth=100):
        stack = [(start, [start])]
        visited = {start}
        
        while stack:
            current, path = stack.pop()
            
            if current == goal:
                return path
            
            if len(path) > max_depth:
                continue
            
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if (0 <= neighbor[0] < grid.shape[0] and 
                    0 <= neighbor[1] < grid.shape[1] and
                    grid[neighbor[0], neighbor[1]] != 5 and  # 5 = WALL
                    neighbor not in visited):
                    
                    visited.add(neighbor)
                    stack.append((neighbor, path + [neighbor]))
        
        return None
    
    def get_next_action(self, current_position):
        if not self.path or self.current_path_index >= len(self.path) - 1:
            return None
        
        current = self.path[self.current_path_index]
        next_pos = self.path[self.current_path_index + 1]
        
        # Визначаємо напрямок руху
        dx = next_pos[0] - current[0]
        dy = next_pos[1] - current[1]
        
        if dx == -1 and dy == 0:
            action = 0  # Вверх
        elif dx == 0 and dy == 1:
            action = 1  # Вправо
        elif dx == 1 and dy == 0:
            action = 2  # Вниз
        elif dx == 0 and dy == -1:
            action = 3  # Вліво
        else:
            action = None
        
        if action is not None:
            self.current_path_index += 1
        
        return action
    
    def predict_action(self, state):
        # Для PathFinder, state містить поточну позицію
        if isinstance(state, tuple) and len(state) >= 2:
            current_pos = (state[-2], state[-1])
            return self.get_next_action(current_pos)
        return None
    
    def reset(self):
        self.path = []
        self.current_path_index = 0
    
    def get_statistics(self):
        return self.stats.copy()
    
    def visualize_path(self, grid):
        if not self.path:
            return grid.copy()
        
        visualization = grid.copy()
        for i, (x, y) in enumerate(self.path):
            if i == 0:
                visualization[x, y] = 6  # START
            elif i == len(self.path) - 1:
                visualization[x, y] = 2  # GOAL
            else:
                visualization[x, y] = 7  # PATH (спеціальне значення)
        
        return visualization