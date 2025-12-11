import numpy as np
import time
import random
from sklearn.ensemble import RandomForestClassifier

def train_rl_agent(env, episodes, max_steps=1000, timeout=300):
    start_time = time.time()
    
    from rl_agent import GeneralQLearningAgent
    agent = GeneralQLearningAgent(alpha=0.1, gamma=0.99, epsilon=0.3)
    
    rewards_history = []
    success_history = []
    terrains_trained = 0
    
    print(f"\n=== Training General RL Agent ===")
    print(f"Episodes: {episodes}, Max steps: {max_steps}")
    
    try:
        for episode in range(episodes):
            if episode % 50 == 0:
                env.generate_random()
                terrains_trained += 1
            
            state = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            while not done and steps < max_steps:
                action = agent.choose_action(state, training=True)
                next_state, reward, done = env.step(action)
                agent.learn(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                steps += 1
            
            rewards_history.append(episode_reward)
            success_history.append(1 if env.reached_goal else 0)
            agent.update_epsilon(episode, episodes)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards_history[-100:]) if len(rewards_history) >= 100 else np.mean(rewards_history)
                success_rate = np.mean(success_history[-100:]) * 100 if success_history else 0
                
                print(f"Episode {episode + 1}/{episodes}: "
                      f"Avg reward: {avg_reward:.2f}, "
                      f"Success: {success_rate:.1f}%, "
                      f"Epsilon: {agent.epsilon:.3f}, "
                      f"Terrains: {terrains_trained}")
        
        train_time = time.time() - start_time
        avg_reward = np.mean(rewards_history)
        success_rate = np.mean(success_history) * 100
        
        agent.training_stats['episodes_trained'] = episodes
        agent.training_stats['avg_reward'] = avg_reward
        agent.training_stats['unique_states_seen'] = len(agent.q_table)
        
        print(f"\n=== Training Complete ===")
        print(f"Total episodes: {episodes}")
        print(f"Terrains used: {terrains_trained}")
        print(f"Unique states learned: {len(agent.q_table)}")
        print(f"Average reward: {avg_reward:.2f}")
        print(f"Success rate: {success_rate:.1f}%")
        print(f"Training time: {train_time:.2f}s")
        
        return agent, rewards_history, True, train_time
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return agent, rewards_history, False, time.time() - start_time

def train_il_agent(env, expert_agent, episodes, timeout=300):
    start_time = time.time()
    
    from il_agent import ImitationLearner
    il_agent = ImitationLearner(env)
    
    print(f"\n=== Training General IL Agent ===")
    print(f"Collecting demonstrations from expert...")
    
    demonstrations = []
    terrains_used = 0
    max_demonstrations = 5000
    
    try:
        for episode in range(min(episodes, 50)):
            if episode % 5 == 0:
                env.generate_random()
                terrains_used += 1
            
            state = env.reset()
            done = False
            steps = 0
            max_demo_steps = 50
            
            while not done and steps < max_demo_steps and len(demonstrations) < max_demonstrations:
                try:
                    if hasattr(expert_agent, 'choose_action'):
                        action = expert_agent.choose_action(state, training=False)
                    elif hasattr(expert_agent, 'predict_action'):
                        action = expert_agent.predict_action(state)
                    else:
                        action = random.randint(0, 3)
                    
                    demonstrations.append((state, action))
                    next_state, _, done = env.step(action)
                    state = next_state
                    steps += 1
                    
                except Exception as e:
                    print(f"  Error in episode {episode}, step {steps}: {e}")
                    break
            
            if len(demonstrations) >= max_demonstrations:
                print(f"  Reached maximum demonstrations ({max_demonstrations})")
                break
        
        print(f"Collected {len(demonstrations)} demonstrations from {terrains_used} terrains")
        
        if demonstrations:
            X = []
            y = []
            
            max_samples = min(2000, len(demonstrations))
            demonstrations_subset = random.sample(demonstrations, max_samples) if len(demonstrations) > max_samples else demonstrations
            
            for state, action in demonstrations_subset:
                if isinstance(state, tuple):
                    state_vector = list(state)
                elif isinstance(state, np.ndarray):
                    state_vector = state.flatten().tolist()
                else:
                    state_vector = [state]
                
                X.append(state_vector)
                y.append(action)
            
            X = np.array(X)
            y = np.array(y)
            
            print(f"Training IL model on {len(X)} samples...")
            train_start = time.time()
            
            il_agent.model = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=-1
            )
            
            il_agent.model.fit(X, y)
            train_time = time.time() - train_start
            
            train_predictions = il_agent.model.predict(X)
            accuracy = np.mean(train_predictions == y)
            
            il_agent.is_trained = True
            il_agent.training_stats = {
                'demonstrations_collected': len(demonstrations),
                'train_accuracy': accuracy,
                'training_time': train_time,
                'total_time': time.time() - start_time,
                'terrains_used': terrains_used,
                'samples_used': len(X)
            }
            
            print(f"IL Training Complete!")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  Training time: {train_time:.2f}s")
            print(f"  Total time: {il_agent.training_stats['total_time']:.2f}s")
            print(f"  Terrains used: {terrains_used}")
            print(f"  Samples used: {len(X)}")
            
            return il_agent, True, il_agent.training_stats['total_time'], train_time
        else:
            print("No demonstrations collected!")
            return il_agent, False, time.time() - start_time, 0
            
    except Exception as e:
        print(f"Error during IL training: {e}")
        import traceback
        traceback.print_exc()
        return il_agent, False, time.time() - start_time, 0

def evaluate_agent(env, agent, agent_type="RL", num_episodes=5, max_steps=100):
    start_time = time.time()
    
    total_rewards = []
    success_count = 0
    trajectories = []
    
    print(f"\n=== Evaluating {agent_type} Agent ===")
    print(f"Testing on {num_episodes} different terrains...")
    
    try:
        for episode in range(num_episodes):
            env.generate_random()
            
            if agent_type == "PathFinder":
                if hasattr(agent, 'find_path'):
                    grid = env.get_grid()
                    path = agent.find_path(grid, env.start, env.goal)
                    state = env.reset()
                    
                    if path:
                        steps = 0
                        done = False
                        episode_reward = 0
                        episode_trajectory = [env.position]
                        
                        for pos in path[1:]:
                            dx = pos[0] - env.position[0]
                            dy = pos[1] - env.position[1]
                            
                            if dx == -1 and dy == 0:
                                action = 0
                            elif dx == 0 and dy == 1:
                                action = 1
                            elif dx == 1 and dy == 0:
                                action = 2
                            elif dx == 0 and dy == -1:
                                action = 3
                            else:
                                continue
                            
                            next_state, reward, done = env.step(action)
                            episode_reward += reward
                            steps += 1
                            episode_trajectory.append(env.position)
                            
                            if done:
                                break
                        
                        trajectories = episode_trajectory
                        total_rewards.append(episode_reward)
                        if env.reached_goal:
                            success_count += 1
                            print(f"  Terrain {episode+1}: SUCCESS in {steps} steps")
                        else:
                            print(f"  Terrain {episode+1}: FAILED after {steps} steps")
                    else:
                        print(f"  Terrain {episode+1}: NO PATH FOUND")
                        total_rewards.append(-100)
                        env.reset()
                else:
                    continue
            else:
                state = env.reset()
                episode_reward = 0
                done = False
                steps = 0
                episode_trajectory = [env.position]
                
                while not done and steps < max_steps:
                    try:
                        if agent_type == "RL":
                            if hasattr(agent, 'choose_action'):
                                action = agent.choose_action(state, training=False)
                            elif hasattr(agent, 'predict_action'):
                                action = agent.predict_action(state)
                            else:
                                action = random.randint(0, 3)
                        else:
                            if hasattr(agent, 'predict_action'):
                                action = agent.predict_action(state)
                            else:
                                action = random.randint(0, 3)
                    except Exception as e:
                        print(f"  Error getting action: {e}")
                        action = random.randint(0, 3)
                    
                    next_state, reward, done = env.step(action)
                    state = next_state
                    episode_reward += reward
                    steps += 1
                    episode_trajectory.append(env.position)
                
                total_rewards.append(episode_reward)
                if env.reached_goal:
                    success_count += 1
                    print(f"  Terrain {episode+1}: SUCCESS in {steps} steps")
                else:
                    print(f"  Terrain {episode+1}: FAILED after {steps} steps")
                
                if episode == 0 and episode_trajectory:
                    trajectories = episode_trajectory
        
        eval_time = time.time() - start_time
        success_rate = (success_count / num_episodes) * 100 if num_episodes > 0 else 0
        avg_reward = np.mean(total_rewards) if total_rewards else 0
        
        print(f"\nEvaluation Results:")
        print(f"  Success rate: {success_rate:.1f}% ({success_count}/{num_episodes})")
        print(f"  Average reward: {avg_reward:.2f}")
        print(f"  Evaluation time: {eval_time:.2f}s")
        
        if not trajectories:
            trajectories = [env.position] if hasattr(env, 'position') else [(0, 0)]
        
        return success_rate, avg_reward, 0, eval_time, trajectories
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        
        default_trajectory = [(0, 0)] if hasattr(env, 'position') else [(0, 0)]
        return 0, 0, 0, 0, default_trajectory