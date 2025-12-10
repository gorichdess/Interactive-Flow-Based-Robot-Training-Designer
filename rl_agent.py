# general_rl_agent.py
import numpy as np
import random
from collections import defaultdict

class GeneralQLearningAgent:
    
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=0.2):
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        
        # Використовуємо словник для гнучкості
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])  # 4 actions
        
        # Статистика тренування
        self.training_stats = {
            'episodes_trained': 0,
            'unique_states_seen': 0,
            'avg_reward': 0.0
        }
        
    def get_state_key(self, state):
        if isinstance(state, tuple):
            return state
        elif isinstance(state, np.ndarray):
            return tuple(state.flatten())
        elif isinstance(state, list):
            return tuple(state)
        else:
            return str(state)
    
    def choose_action(self, state, training=True):
        state_key = self.get_state_key(state)
        
        # Ініціалізуємо стан, якщо він новий
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0, 0.0, 0.0, 0.0]
        
        # Exploration під час тренування
        if training and random.random() < self.epsilon:
            return random.randint(0, 3)
        
        # Exploitation: вибираємо найкращу дію
        q_values = self.q_table[state_key]
        max_q = np.max(q_values)
        
        # Якщо кілька дій мають однакове Q-значення, вибираємо випадково
        best_actions = np.where(np.array(q_values) == max_q)[0]
        return random.choice(best_actions)
    
    def learn(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Ініціалізуємо стани, якщо потрібно
        if state_key not in self.q_table:
            self.q_table[state_key] = [0.0, 0.0, 0.0, 0.0]
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = [0.0, 0.0, 0.0, 0.0]
        
        # Q-learning формула
        current_q = self.q_table[state_key][action]
        
        if done:
            target_q = reward
        else:
            next_max_q = np.max(self.q_table[next_state_key])
            target_q = reward + self.gamma * next_max_q
        
        # Оновлюємо Q-значення
        self.q_table[state_key][action] = current_q + self.alpha * (target_q - current_q)
    
    def update_epsilon(self, episode, total_episodes):
        decay_rate = 0.999
        self.epsilon = max(0.01, self.epsilon * decay_rate)
    
    def predict_action(self, state):
        return self.choose_action(state, training=False)
    
    def get_action_distribution(self, state):
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            return [0.25, 0.25, 0.25, 0.25]  # Рівномірний розподіл
        
        q_values = np.array(self.q_table[state_key])
        # Softmax для отримання ймовірностей
        exp_q = np.exp(q_values - np.max(q_values))  # Для стабільності
        probs = exp_q / exp_q.sum()
        return probs
    
    def save_model(self, filepath):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load_model(self, filepath):
        import pickle
        with open(filepath, 'rb') as f:
            q_table_dict = pickle.load(f)
            self.q_table.update(q_table_dict)
    
    def get_statistics(self):
        self.training_stats['unique_states_seen'] = len(self.q_table)
        return self.training_stats