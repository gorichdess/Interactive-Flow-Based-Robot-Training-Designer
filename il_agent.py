class ImitationLearner:
    def __init__(self, env):
        self.env = env
        print("IL model created")

    def train(self, expert_agent, episodes=50):
        print("Collecting demonstrations from expert")

    def predict_action(self, state):
        print(f"Predicting action for {state}")
        return 0
