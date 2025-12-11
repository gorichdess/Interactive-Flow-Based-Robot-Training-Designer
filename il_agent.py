import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier

class ImitationLearner:
    def __init__(self, env=None, config=None):
        self.env = env
        self.config = config or {}
        n_estimators = self.config.get('n_estimators', 100)
        max_depth = self.config.get('max_depth', 20)
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        self.training_stats = {
            'demonstrations_collected': 0,
            'train_accuracy': 0,
            'training_time': 0,
            'total_time': 0,
            'terrains_used': 0
        }

        self.is_trained = False

    def _process_state_for_training(self, state):
        if isinstance(state, (tuple, list)):
            return list(state)
        elif isinstance(state, np.ndarray):
            return state.flatten().tolist()
        elif isinstance(state, (int, float)):
            return [state]
        else:
            try:
                return np.array(state).flatten().tolist()
            except Exception:
                return [state]

    def _process_state_for_prediction(self, state):
        return self._process_state_for_training(state)
    
    def train(self, expert_agent, env, num_episodes=100, max_steps=100):
        start_time = time.time()
        demonstrations = []
        terrains_used = 0
        print(f"Training on {num_episodes} episodes...")

        for episode in range(num_episodes):
            if episode % 20 == 0:
                env.generate_random()
                terrains_used += 1
            state = env.reset()
            done = False
            steps = 0

            while not done and steps < max_steps:
                try:
                    if hasattr(expert_agent, 'choose_action'):
                        action = expert_agent.choose_action(state, training=False)
                    elif hasattr(expert_agent, 'predict_action'):
                        action = expert_agent.predict_action(state)
                    else:
                        action = np.random.randint(0, 4)
                    
                    state_vector = self._process_state_for_training(state)
                    demonstrations.append((state_vector, action))
                    next_state, _, done = env.step(action)
                    state = next_state
                    steps += 1
                except Exception as e:
                    print(f"Error in episode {episode}: {e}")
                    break
        
        if not demonstrations:
            print("No demonstrations collected!")
            return False, time.time() - start_time, 0
        
        print(f"Collected {len(demonstrations)} demonstrations from {terrains_used} terrains")
        
        X = np.array([d[0] for d in demonstrations])
        y = np.array([d[1] for d in demonstrations])
        print(f"Training model on {len(X)} samples...")
        
        train_start = time.time()
        self.model.fit(X, y)
        train_time = time.time() - train_start
        
        train_predictions = self.model.predict(X)
        train_accuracy = np.mean(train_predictions == y)
        
        self.training_stats = {
            'demonstrations_collected': len(demonstrations),
            'train_accuracy': train_accuracy,
            'training_time': train_time,
            'total_time': time.time() - start_time,
            'terrains_used': terrains_used
        }
        
        self.is_trained = True
        print(f"Training complete!")
        print(f"  Accuracy: {train_accuracy:.3f}")
        print(f"  Training time: {train_time:.2f}s")
        print(f"  Total time: {self.training_stats['total_time']:.2f}s")
        print(f"  Terrains used: {terrains_used}")
        
        return True, self.training_stats['total_time'], train_time

    def predict_action(self, state):
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")
        
        state_vector = self._process_state_for_prediction(state)

        if isinstance(state_vector, list):
            state_vector = np.array(state_vector).reshape(1, -1)
        elif state_vector.ndim == 1:
            state_vector = state_vector.reshape(1, -1)

        return self.model.predict(state_vector)[0]

    def predict_action_proba(self, state):
        if not self.is_trained:
            raise RuntimeError("Model not trained yet. Call train() first.")

        state_vector = self._process_state_for_prediction(state)

        if isinstance(state_vector, list):
            state_vector = np.array(state_vector).reshape(1, -1)
        elif state_vector.ndim == 1:
            state_vector = state_vector.reshape(1, -1)

        return self.model.predict_proba(state_vector)[0]

    def get_statistics(self):
        return self.training_stats
    
    def save_model(self, filepath):
        import joblib
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        import joblib
        self.model = joblib.load(filepath)
        self.is_trained = True