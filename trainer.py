import numpy as np
import time
from tqdm import tqdm
from collections import defaultdict

# Import the agent classes
from rl_agent import QLearningAgent
from il_agent import ImitationLearner

def train_rl_agent(env, episodes, max_steps=1000, timeout=300):
    start_time = time.time()
    agent = QLearningAgent(env)
    
    rewards = []
    steps_per_episode = []
    success_rate = []
    
    # Use tqdm for progress bar if available
    try:
        progress_bar = tqdm(range(episodes), desc="Training RL Agent")
    except:
        # Fallback if tqdm is not available
        progress_bar = range(episodes)
        print(f"Training RL Agent for {episodes} episodes...")
    
    try:
        for episode in progress_bar:
            episode_start_time = time.time()
            
            # Check timeout
            if time.time() - start_time > timeout:
                print(f"Training timeout after {timeout} seconds")
                return agent, rewards, False, time.time() - start_time
            
            # Reset environment
            state = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            # Run episode
            while not done and steps < max_steps:
                # Choose action
                action = agent.choose_action(state, training=True)
                
                # Take action
                next_state, reward, done = env.step(action)
                
                # Learn from experience
                agent.learn(state, action, reward, next_state, done)
                
                # Update state and counters
                state = next_state
                episode_reward += reward
                steps += 1
            
            # Update statistics
            rewards.append(episode_reward)
            steps_per_episode.append(steps)
            
            # Check if goal was reached
            if hasattr(env, 'reached_goal'):
                success_rate.append(1 if env.reached_goal else 0)
            
            # Update epsilon (exploration rate)
            agent.update_epsilon(episode, episodes)
            
            # Update progress if using tqdm
            if hasattr(progress_bar, 'set_postfix'):
                avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                progress_bar.set_postfix({
                    'avg_reward': f'{avg_reward:.2f}',
                    'epsilon': f'{agent.epsilon:.3f}',
                    'steps': steps
                })
            
            # Print progress every 100 episodes
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(rewards[-100:]) if len(rewards) >= 100 else np.mean(rewards)
                success_percentage = np.mean(success_rate[-100:]) * 100 if success_rate else 0
                print(f"Episode {episode + 1}/{episodes}: "
                      f"Avg Reward (last 100): {avg_reward:.2f}, "
                      f"Success Rate: {success_percentage:.1f}%")
        
        # Calculate final statistics
        train_time = time.time() - start_time
        
        print(f"\nRL Training Completed:")
        print(f"  Total episodes: {episodes}")
        print(f"  Average reward: {np.mean(rewards):.2f}")
        print(f"  Average steps: {np.mean(steps_per_episode):.1f}")
        if success_rate:
            print(f"  Success rate: {np.mean(success_rate)*100:.1f}%")
        print(f"  Training time: {train_time:.2f}s")
        
        return agent, rewards, True, train_time
        
    except Exception as e:
        print(f"Error during RL training: {e}")
        import traceback
        traceback.print_exc()
        return agent, rewards, False, time.time() - start_time

def train_il_agent(env, expert_agent, episodes, timeout=300):
    start_time = time.time()
    
    # Create IL agent
    il_agent = ImitationLearner(env)
    
    try:
        # Train IL agent using expert demonstrations
        success, total_time, train_time = il_agent.train(
            expert_agent=expert_agent,
            num_episodes=episodes
        )
        
        if success:
            print(f"\nIL Training Completed:")
            print(f"  Demonstrations collected: {il_agent.training_stats['demonstrations_collected']}")
            print(f"  Training accuracy: {il_agent.training_stats.get('train_accuracy', 0):.3f}")
            print(f"  Validation accuracy: {il_agent.training_stats.get('val_accuracy', 0):.3f}")
            print(f"  Model training time: {train_time:.2f}s")
            print(f"  Total time: {total_time:.2f}s")
        else:
            print("IL Training failed or timed out")
        
        return il_agent, success, total_time, train_time
        
    except Exception as e:
        print(f"Error during IL training: {e}")
        import traceback
        traceback.print_exc()
        total_time = time.time() - start_time
        return il_agent, False, total_time, 0

def evaluate_agent(env, agent, agent_type="RL", num_episodes=10, max_steps=1000):
    start_time = time.time()
    
    total_rewards = []
    steps_per_episode = []
    success_count = 0
    
    print(f"\nEvaluating {agent_type} agent over {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Choose action based on agent type
            if agent_type == "RL":
                action = agent.choose_action(state, training=False)
            else:  # IL agent
                action = agent.predict_action(state)
            
            # Take action
            next_state, reward, done = env.step(action)
            
            episode_reward += reward
            state = next_state
            steps += 1
        
        total_rewards.append(episode_reward)
        steps_per_episode.append(steps)
        
        # Check if goal was reached
        if hasattr(env, 'reached_goal'):
            if env.reached_goal:
                success_count += 1
    
    # Calculate metrics
    eval_time = time.time() - start_time
    success_rate = (success_count / num_episodes) * 100
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(steps_per_episode)
    
    print(f"Evaluation Results for {agent_type}:")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Average reward: {avg_reward:.2f}")
    print(f"  Average steps: {avg_steps:.1f}")
    print(f"  Evaluation time: {eval_time:.2f}s")
    
    return success_rate, avg_reward, avg_steps, eval_time

def compare_agents(env, rl_agent, il_agent, num_episodes=10):
    print("\n" + "="*50)
    print("COMPARING RL vs IL AGENTS")
    print("="*50)
    
    results = {}
    
    # Evaluate RL agent
    if rl_agent:
        rl_success, rl_reward, rl_steps, rl_time = evaluate_agent(
            env, rl_agent, "RL", num_episodes
        )
        results['RL'] = {
            'success_rate': rl_success,
            'avg_reward': rl_reward,
            'avg_steps': rl_steps,
            'eval_time': rl_time
        }
    
    # Evaluate IL agent
    if il_agent:
        il_success, il_reward, il_steps, il_time = evaluate_agent(
            env, il_agent, "IL", num_episodes
        )
        results['IL'] = {
            'success_rate': il_success,
            'avg_reward': il_reward,
            'avg_steps': il_steps,
            'eval_time': il_time
        }
    
    # Print comparison
    if rl_agent and il_agent:
        print("\nComparison Summary:")
        print("-"*40)
        
        metrics = ['success_rate', 'avg_reward', 'avg_steps', 'eval_time']
        for metric in metrics:
            rl_val = results['RL'][metric]
            il_val = results['IL'][metric]
            
            if metric == 'success_rate':
                print(f"{metric.replace('_', ' ').title()}: RL: {rl_val:.1f}% vs IL: {il_val:.1f}%")
            elif metric == 'eval_time':
                print(f"{metric.replace('_', ' ').title()}: RL: {rl_val:.2f}s vs IL: {il_val:.2f}s")
            else:
                print(f"{metric.replace('_', ' ').title()}: RL: {rl_val:.2f} vs IL: {il_val:.2f}")
    
    return results