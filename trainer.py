def train_rl_agent(env, episodes):
    print(f"Training RL for {episodes} episodes")
    return None, [], True, 0.0 


def train_il_agent(env, expert, episodes):
    print(f"Training IL for {episodes} episodes")
    return None, True, 0.0


def evaluate_agent(env, agent, agent_type):
    print(f"Evaluating {agent_type} agent")
    return 100, 0, 10, 0.1
