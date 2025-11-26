class QLearningAgent:
    def __init__(self, env):
        self.env = env
        print("RL agent initialized")

    def choose_action(self, state):
        print(f"Choosing action for state {state}")
        return 0 

    def learn(self, s, a, r, ns):
        print("Updating Q-table from experience")
