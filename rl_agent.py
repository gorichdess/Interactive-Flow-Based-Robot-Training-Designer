import numpy as np
import random
from collections import defaultdict
import time

class QLearningAgent:
    def __init__(self, env, alpha=0.5, gamma=0.95, epsilon=0.2):
        self.env = env
        self.q_table = defaultdict(lambda: np.zeros(4))  # 4 actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.training_stats = {
            'episodes_trained': 0,
            'total_reward': 0,
            'successful_episodes': 0
        }
        
    def get_state_key(self, state):
        if isinstance(state, tuple):
            return state
        elif isinstance(state, np.ndarray):
            return tuple(state.flatten())
        return state
    
    def choose_action(self, state, training=True):
        state_key = self.get_state_key(state)
        
        # Exploration during training
        if training and random.random() < self.epsilon:
            return random.randint(0, 3)
        
        # Exploitation: choose best action
        q_values = self.q_table[state_key]
        max_q = np.max(q_values)
        # If multiple actions have same Q-value, choose randomly among them
        best_actions = np.where(q_values == max_q)[0]
        return np.random.choice(best_actions)
    
    def learn(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        
        if done:
            target = reward
        else:
            # Maximum Q-value for next state
            next_max_q = np.max(self.q_table[next_state_key])
            target = reward + self.gamma * next_max_q
        
        # Update Q-value
        self.q_table[state_key][action] += self.alpha * (target - current_q)
    
    def update_epsilon(self, episode, total_episodes):
        decay_rate = 0.99
        self.epsilon = max(0.01, self.epsilon * decay_rate)
    
    def save_model(self, filepath):
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
    
    def load_model(self, filepath):
        import pickle
        with open(filepath, 'rb') as f:
            q_table_dict = pickle.load(f)
            self.q_table.update(q_table_dict)
    
    def get_action_distribution(self, state):
        state_key = self.get_state_key(state)
        q_values = self.q_table[state_key]
        
        # Softmax over Q-values
        exp_q = np.exp(q_values - np.max(q_values))  # For numerical stability
        probs = exp_q / exp_q.sum()
        return probs
    
    def get_value_function(self, state):
        state_key = self.get_state_key(state)
        return np.max(self.q_table[state_key])