import dearpygui.dearpygui as dpg
import numpy as np
from node_editor import NodeEditorApp
from rl_agent import GeneralQLearningAgent 
from environment import TerrainEnv, LabyrinthEnv
from pathfinder_agent import PathFinderAgent
from il_agent import ImitationLearner
from trainer import train_rl_agent, train_il_agent, evaluate_agent
from settings import GRID_SIZE, RL_EPISODES, IL_EPISODES, MAX_STEPS, TIMEOUT, START_POSITION, GOAL_POSITION

class GlobalSettings:
    def __init__(self):
        self.GRID_SIZE = GRID_SIZE
        self.RL_EPISODES = RL_EPISODES
        self.IL_EPISODES = IL_EPISODES
        self.MAX_STEPS = MAX_STEPS
        self.TIMEOUT = TIMEOUT
        self.START_POSITION = START_POSITION
        self.GOAL_POSITION = GOAL_POSITION

SETTINGS = GlobalSettings()

class RobotSimulatorApp:
    def __init__(self):
        self.environments = {} 
        self.rl_agents = {}
        self.il_agents = {}
        self.pathfinder_agents = {}
        self.settings = SETTINGS
        
        self.node_editor = NodeEditorApp(self)
        self.node_editor_window_tag = "NodeEditorWindow"
        self.is_training = False

    def setup(self):
        dpg.create_context()
        dpg.create_viewport(title="Robot Terrain Learning Node Editor", width=1920, height=1080, decorated=False)
        self.create_menu_bar()
        self.node_editor.create_node_editor_window()

    def create_menu_bar(self):
        with dpg.viewport_menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Toggle Fullscreen", 
                                callback=lambda: dpg.toggle_viewport_fullscreen())  
                dpg.add_separator()
                dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())
                
            with dpg.menu(label="Windows"):
                dpg.add_menu_item(label="Node Editor Window", callback=lambda: dpg.show_item(self.node_editor_window_tag))
            
            with dpg.menu(label="Node Editor"):
                dpg.add_menu_item(label="Add Environment Node", callback=lambda: self.node_editor.add_environment_node())
                dpg.add_separator()
                dpg.add_menu_item(label="Add RL Agent Node", callback=lambda: self.node_editor.add_rl_agent_node())
                dpg.add_menu_item(label="Add IL Agent Node", callback=lambda: self.node_editor.add_il_agent_node())
                dpg.add_separator()
                dpg.add_menu_item(label="Add Visualizer Node", callback=lambda: self.node_editor.add_visualizer_node())
                dpg.add_menu_item(label="Add Settings Node", callback=lambda: self.node_editor.add_settings_node())
                dpg.add_menu_item(label="Add Results Node", callback=lambda: self.node_editor.add_results_node())
                dpg.add_separator()
                dpg.add_menu_item(label="Clear All Nodes", callback=self.node_editor.clear_all_nodes)
    
    def generate_terrain(self, node_id, grid_size=None):
        if node_id not in self.environments:
            print(f"Error: Environment {node_id} not found")
            return
        
        env = self.environments[node_id]
        
        # Store current positions before generating
        current_start = env.start if hasattr(env, 'start') else None
        current_goal = env.goal if hasattr(env, 'goal') else None
        
        # Update size if provided
        if grid_size:
            env.set_size(grid_size)
        
        # Generate terrain
        env.generate_random()
        
        # Restore custom positions if they were set
        if current_start and current_goal:
            env.set_pos(current_start, current_goal)

    def train_rl(self, env_id, agent_id, settings_config=None):
        if self.is_training: return
        self.is_training = True
        
        env = self.environments.get(env_id)
        if not env:
            self.is_training = False
            return
            
        episodes = settings_config.get('rl_episodes', self.settings.RL_EPISODES) if settings_config else self.settings.RL_EPISODES
        max_steps = settings_config.get('max_steps', self.settings.MAX_STEPS) if settings_config else self.settings.MAX_STEPS
        timeout = settings_config.get('timeout', self.settings.TIMEOUT) if settings_config else self.settings.TIMEOUT
        
        try:
            # Використовуємо оновлену функцію тренування
            rl_agent, rewards, success, train_time = train_rl_agent(env, episodes, max_steps, timeout)
            if success:
                self.rl_agents[agent_id] = rl_agent
                print(f"[Simulator] General RL Agent {agent_id} trained successfully")
                print(f"  Episodes: {episodes}")
                print(f"  Unique states learned: {len(rl_agent.q_table)}")
            return rl_agent
        except Exception as e:
            print(f"Error training RL agent: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self.is_training = False


    def train_rl_with_terrains(self, env_id, agent_id, num_terrains=10, settings_config=None):
        if self.is_training: 
            return None
        
        self.is_training = True
        
        env = self.environments.get(env_id)
        if not env:
            self.is_training = False
            print(f"Error: Environment {env_id} not found.")
            return None
        
        try:
            print(f"\n=== Training RL Agent on {num_terrains} different terrains ===")
            
            # Отримуємо налаштування
            episodes = settings_config.get('rl_episodes', self.settings.RL_EPISODES) if settings_config else self.settings.RL_EPISODES
            max_steps = settings_config.get('max_steps', self.settings.MAX_STEPS) if settings_config else self.settings.MAX_STEPS
            timeout = settings_config.get('timeout', self.settings.TIMEOUT) if settings_config else self.settings.TIMEOUT
            
            # Створюємо агента
            agent = GeneralQLearningAgent(alpha=0.1, gamma=0.99, epsilon=0.3)
            
            total_episodes = 0
            rewards_history = []
            success_history = []
            
            # Тренуємо на різних територіях
            for terrain_idx in range(num_terrains):
                print(f"\nTerrain {terrain_idx + 1}/{num_terrains}")
                
                # Генеруємо нову територію
                env.generate_random()
                
                # Тренуємо на цій території
                terrain_episodes = episodes // num_terrains if num_terrains > 0 else episodes
                if terrain_episodes < 10:
                    terrain_episodes = 10  # Мінімум 10 епізодів на територію
                
                for episode in range(terrain_episodes):
                    state = env.reset()
                    episode_reward = 0
                    done = False
                    steps = 0
                    
                    while not done and steps < max_steps:
                        # Вибираємо дію
                        action = agent.choose_action(state, training=True)
                        
                        # Виконуємо дію
                        next_state, reward, done = env.step(action)
                        
                        # Вчимося
                        agent.learn(state, action, reward, next_state, done)
                        
                        # Оновлюємо
                        state = next_state
                        episode_reward += reward
                        steps += 1
                    
                    # Записуємо статистику
                    rewards_history.append(episode_reward)
                    success_history.append(1 if env.reached_goal else 0)
                    total_episodes += 1
                    
                    # Оновлюємо epsilon
                    agent.update_epsilon(total_episodes, episodes)
                    
                    # Прогрес кожні 50 епізодів
                    if total_episodes % 50 == 0:
                        avg_reward = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history)
                        success_rate = np.mean(success_history[-50:]) * 100 if success_history else 0
                        print(f"  Episode {total_episodes}: Avg reward: {avg_reward:.2f}, Success: {success_rate:.1f}%")
            
            # Фінальна статистика
            avg_reward = np.mean(rewards_history)
            success_rate = np.mean(success_history) * 100 if success_history else 0
            
            # Оновлюємо статистику агента
            agent.training_stats['episodes_trained'] = total_episodes
            agent.training_stats['avg_reward'] = avg_reward
            agent.training_stats['unique_states_seen'] = len(agent.q_table)
            agent.training_stats['terrains_used'] = num_terrains
            
            print(f"\n=== Training Complete ===")
            print(f"Total episodes: {total_episodes}")
            print(f"Terrains used: {num_terrains}")
            print(f"Unique states learned: {len(agent.q_table)}")
            print(f"Average reward: {avg_reward:.2f}")
            print(f"Success rate: {success_rate:.1f}%")
            
            # Зберігаємо агента
            self.rl_agents[agent_id] = agent
            return agent
            
        except Exception as e:
            print(f"Error training RL agent on multiple terrains: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self.is_training = False

    def train_il(self, env_id, il_agent_id, teacher_agent_id, settings_config=None):
        if self.is_training: 
            return None
        
        self.is_training = True

        env = self.environments.get(env_id)
        teacher_agent = self.rl_agents.get(teacher_agent_id)
        
        if not env or not teacher_agent:
            self.is_training = False
            print(f"Missing environment or teacher agent. Env: {env_id}, Teacher: {teacher_agent_id}")
            return None
        
        episodes = settings_config.get('il_episodes', self.settings.IL_EPISODES) if settings_config else self.settings.IL_EPISODES
        timeout = settings_config.get('timeout', self.settings.TIMEOUT) if settings_config else self.settings.TIMEOUT

        try:
            # Використовуємо оновлену функцію тренування
            il_agent, success, total_time, train_time = train_il_agent(
                env, teacher_agent, episodes, timeout
            )
            if success:
                self.il_agents[il_agent_id] = il_agent
                print(f"[Simulator] General IL Agent {il_agent_id} trained successfully")
                stats = il_agent.get_statistics()
                print(f"  Demonstrations: {stats.get('demonstrations_collected', 0)}")
                print(f"  Accuracy: {stats.get('train_accuracy', 0):.3f}")
                print(f"  Terrains used: {stats.get('terrains_used', 0)}")
            else:
                print(f"[Simulator] IL Agent {il_agent_id} training failed")
            return il_agent
        except Exception as e:
            print(f"Error training IL agent: {e}")
            import traceback
            traceback.print_exc()
            return None
        finally:
            self.is_training = False
    
    def evaluate(self, env_id, agent_id, agent_type, num_runs=5):
        env = self.environments.get(env_id)
        agent = self.rl_agents.get(agent_id) if agent_type == "RL" else self.il_agents.get(agent_id)

        if not env or not agent:
            print(f"Missing environment or agent. Env: {env_id}, Agent: {agent_id}, Type: {agent_type}")
            return None, None
        
        try:
            print(f"\n[Simulator] Evaluating {agent_type} agent {agent_id}...")
            print(f"Testing on {num_runs} different terrains...")
            
            # Використовуємо оновлену функцію оцінки
            success_rate, avg_reward, avg_steps, eval_time, trajectory = evaluate_agent(
                env, agent, agent_type, num_runs
            )
            
            results = {
                'success_rate': success_rate,
                'avg_reward': avg_reward,
                'avg_steps': avg_steps,
                'eval_time': eval_time
            }
            
            print(f"\n[Simulator] Evaluation complete for {agent_type} agent")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Average reward: {avg_reward:.2f}")
            print(f"  Trajectory length: {len(trajectory) if trajectory else 0}")
            
            return results, trajectory
        except Exception as e:
            print(f"Evaluation error: {e}")
            import traceback
            traceback.print_exc()
            return None, None
        
    def find_path(self, env_id, agent_id):
        env = self.environments.get(env_id)
        agent = self.pathfinder_agents.get(agent_id)
        
        if not env or not agent:
            print(f"Missing environment or pathfinder agent. Env: {env_id}, Agent: {agent_id}")
            return None
        
        try:
            print(f"\n[Simulator] PathFinder agent {agent_id} finding path...")
            grid = env.get_grid()
            path = agent.find_path(grid, env.start, env.goal)
            
            if path:
                print(f"Path found! Length: {len(path)}")
                print(f"Search time: {agent.stats['search_time']:.3f}s")
            else:
                print("No path found!")
            
            return path
        except Exception as e:
            print(f"Error finding path: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def get_agent(self, agent_id):
        return (self.rl_agents.get(agent_id) or 
                self.il_agents.get(agent_id) or 
                self.pathfinder_agents.get(agent_id))
        
    def get_environment(self, env_id):
        return self.environments.get(env_id)

    def run(self):
        self.setup()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.maximize_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()

if __name__ == "__main__":
    app = RobotSimulatorApp()
    app.run()