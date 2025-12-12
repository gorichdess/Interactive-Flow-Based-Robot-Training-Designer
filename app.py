import dearpygui.dearpygui as dpg
from node_editor import NodeEditorApp
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

class RobotSimulatorApp:
    def __init__(self):
        self.environments = {}
        self.rl_agents = {}
        self.il_agents = {}
        self.pathfinder_agents = {}
        self.settings = GlobalSettings()
        self.node_editor = NodeEditorApp(self)
        self.node_editor_window_tag = "NodeEditorWindow"
        self.is_training = False
        self.help_window_tag = "HelpWindow"

    def setup(self):
        dpg.create_context()
        dpg.create_viewport(title="Robot Terrain Learning Node Editor", width=1920, height=1080, decorated=False)
        self.create_menu_bar()
        self.node_editor.create_node_editor_window()
        self.create_help_window()

    def create_menu_bar(self):
        with dpg.viewport_menu_bar():
            with dpg.menu(label="File"):
                dpg.add_menu_item(label="Toggle Fullscreen", callback=lambda: dpg.toggle_viewport_fullscreen())
                dpg.add_separator()
                dpg.add_menu_item(label="Exit", callback=lambda: dpg.stop_dearpygui())
            
            with dpg.menu(label="Windows"):
                dpg.add_menu_item(label="Node Editor Window", callback=lambda: dpg.show_item(self.node_editor_window_tag))
                dpg.add_menu_item(label="Help Guide", callback=lambda: dpg.show_item(self.help_window_tag))
            
            with dpg.menu(label="Node Editor"):
                dpg.add_menu_item(label="Add Environment Node", callback=lambda: self.node_editor.add_environment_node())
                dpg.add_menu_item(label="Add Labyrinth Node", callback=lambda: self.node_editor.add_labyrinth_node())
                dpg.add_separator()
                dpg.add_menu_item(label="Add RL Agent Node", callback=lambda: self.node_editor.add_rl_agent_node())
                dpg.add_menu_item(label="Add IL Agent Node", callback=lambda: self.node_editor.add_il_agent_node())
                dpg.add_menu_item(label="Add PathFinder Node", callback=lambda: self.node_editor.add_pathfinder_node())
                dpg.add_separator()
                dpg.add_menu_item(label="Add Visualizer Node", callback=lambda: self.node_editor.add_visualizer_node())
                dpg.add_menu_item(label="Add Settings Node", callback=lambda: self.node_editor.add_settings_node())
                dpg.add_menu_item(label="Add Results Node", callback=lambda: self.node_editor.add_results_node())
                dpg.add_separator()
                dpg.add_menu_item(label="Clear All Nodes", callback=self.node_editor.clear_all_nodes)
            
            with dpg.menu(label="Help"):
                dpg.add_menu_item(label="Quick Start Guide", callback=lambda: dpg.show_item(self.help_window_tag))
                dpg.add_menu_item(label="Node Connection Guide", callback=self.show_connection_guide)
                dpg.add_menu_item(label="Troubleshooting", callback=self.show_troubleshooting)
                dpg.add_separator()
                dpg.add_menu_item(label="About", callback=self.show_about)

    def create_help_window(self):
        if dpg.does_item_exist(self.help_window_tag):
            dpg.delete_item(self.help_window_tag)
        
        with dpg.window(label="Robot Simulator Help Guide", tag=self.help_window_tag, 
                       width=800, height=600, show=False, pos=[100, 100]):
            
            with dpg.tab_bar():
                # Quick Start Tab
                with dpg.tab(label="Quick Start"):
                    with dpg.group():
                        dpg.add_text("Robot Simulator Node Editor - Quick Start", color=(0, 255, 255))
                        dpg.add_separator()
                        
                        with dpg.collapsing_header(label="Getting Started", default_open=True):
                            dpg.add_text("1. Add nodes from the 'Node Editor' menu or toolbar buttons")
                            dpg.add_text("2. Connect nodes by dragging from outputs (right) to inputs (left)")
                            dpg.add_text("3. Configure settings in Settings node if needed")
                            dpg.add_text("4. Click buttons on nodes to execute actions")
                            dpg.add_text("5. Visualize results in Visualizer node")
                        
                        with dpg.collapsing_header(label="Basic Workflow"):
                            dpg.add_text("For RL Agent Training:")
                            dpg.add_text("    * Environment Node -> RL Agent Node")
                            dpg.add_text("    * (Optional) Settings Node -> Environment Node")
                            dpg.add_text("    * Click 'Generate Terrain' then 'Train RL Agent'")
                            
                            dpg.add_spacer(height=10)
                            dpg.add_text("For IL Agent Training:")
                            dpg.add_text("    * Environment Node -> IL Agent Node")
                            dpg.add_text("    * RL Agent Node -> IL Agent Node (as teacher)")
                            dpg.add_text("    * Click 'Train IL Agent'")
                            
                            dpg.add_spacer(height=10)
                            dpg.add_text("For Path Finding:")
                            dpg.add_text("    * Environment Node -> PathFinder Node")
                            dpg.add_text("    * Click 'Find Path'")
                        
                        with dpg.collapsing_header(label="Navigation"):
                            dpg.add_text("  * Left-click and drag: Move nodes")
                            dpg.add_text("  * Right-Alt + click link: Delete connection")
                            dpg.add_text("  * 'X Delete' button: Remove node")
                
                # Node Guide Tab
                with dpg.tab(label="Node Guide"):
                    with dpg.group():
                        dpg.add_text("Node Types and Usage", color=(0, 255, 255))
                        dpg.add_separator()
                        
                        with dpg.collapsing_header(label="Environment Node", default_open=True):
                            dpg.add_text("Purpose: Creates terrain for robots to navigate")
                            dpg.add_text("Inputs: Config (from Settings node)")
                            dpg.add_text("Outputs: Grid Data, Env Instance")
                            dpg.add_text("Actions: Generate Terrain")
                            dpg.add_text("Connect To: All agent nodes, visualizer")
                        
                        with dpg.collapsing_header(label="Labyrinth Node"):
                            dpg.add_text("Purpose: Creates maze with walls for navigation")
                            dpg.add_text("Inputs: Config (from Settings node)")
                            dpg.add_text("Outputs: Grid Data, Env Instance")
                            dpg.add_text("Actions: Generate Labyrinth")
                            dpg.add_text("Connect To: All agent nodes, visualizer")
                        
                        with dpg.collapsing_header(label="RL Agent Node"):
                            dpg.add_text("Purpose: Trains Q-learning agent")
                            dpg.add_text("Inputs: Environment, Config")
                            dpg.add_text("Outputs: Trained Agent")
                            dpg.add_text("Actions: Train RL Agent")
                            dpg.add_text("Connect From: Environment node")
                            dpg.add_text("Connect To: IL Agent (as teacher), Results")
                        
                        with dpg.collapsing_header(label="IL Agent Node"):
                            dpg.add_text("Purpose: Trains imitation learning agent")
                            dpg.add_text("Inputs: Environment, Teacher Agent, Config")
                            dpg.add_text("Outputs: Trained Agent")
                            dpg.add_text("Actions: Train IL Agent")
                            dpg.add_text("Connect From: Environment + RL Agent")
                            dpg.add_text("Connect To: Results")
                        
                        with dpg.collapsing_header(label="PathFinder Node"):
                            dpg.add_text("Purpose: Finds optimal paths (A*, BFS, DFS)")
                            dpg.add_text("Inputs: Environment, Grid Data")
                            dpg.add_text("Outputs: PathFinder Agent")
                            dpg.add_text("Actions: Find Path, Reset Path")
                            dpg.add_text("Connect From: Environment node")
                            dpg.add_text("Algorithms: A* (default), BFS, DFS")
                        
                        with dpg.collapsing_header(label="Visualizer Node"):
                            dpg.add_text("Purpose: Visualizes terrain and agent paths")
                            dpg.add_text("Inputs: Environment, Agent")
                            dpg.add_text("Outputs: None")
                            dpg.add_text("Actions: Visualize Terrain, Evaluate Agent")
                            dpg.add_text("Connect From: Environment + Agent nodes")
                        
                        with dpg.collapsing_header(label="Settings Node"):
                            dpg.add_text("Purpose: Configure simulation parameters")
                            dpg.add_text("Inputs: None")
                            dpg.add_text("Outputs: Config Data")
                            dpg.add_text("Parameters: Grid size, positions, episodes, etc.")
                            dpg.add_text("Connect To: Environment and Agent nodes")
                        
                        with dpg.collapsing_header(label="Results Node"):
                            dpg.add_text("Purpose: Displays evaluation metrics")
                            dpg.add_text("Inputs: Agent, Environment")
                            dpg.add_text("Outputs: None")
                            dpg.add_text("Actions: Evaluate Agent")
                            dpg.add_text("Connect From: Agent + Environment nodes")
                
                # Connection Guide Tab
                with dpg.tab(label="Connections"):
                    with dpg.group():
                        dpg.add_text("Node Connection Guide", color=(0, 255, 255))
                        dpg.add_separator()
                        
                        dpg.add_text("CRITICAL: Always connect 'env_instance' to agents, NOT 'grid_data'!", color=(255, 100, 100))
                        dpg.add_separator()
                        
                        with dpg.collapsing_header(label="Essential Connections", default_open=True):
                            dpg.add_text("   Environment/Labyrinth -> RL Agent:", color=(0, 255, 0))
                            dpg.add_text("   Connect: env_instance -> env_instance")
                            
                            dpg.add_text("   Environment/Labyrinth -> IL Agent:", color=(0, 255, 0))
                            dpg.add_text("   Connect: env_instance -> env_instance")
                            
                            dpg.add_text("   Environment/Labyrinth -> PathFinder:", color=(0, 255, 0))
                            dpg.add_text("   Connect: env_instance -> env_instance")
                            
                            dpg.add_text("   Environment/Labyrinth -> Visualizer:", color=(0, 255, 0))
                            dpg.add_text("   Connect: env_instance -> env_instance")
                            
                            dpg.add_text("   RL Agent -> IL Agent:", color=(0, 255, 0))
                            dpg.add_text("   Connect: output_agent -> teacher_agent")
                            
                            dpg.add_text("   Settings -> Environment/Agent:", color=(0, 255, 0))
                            dpg.add_text("   Connect: output_config -> config")
                        
                        with dpg.collapsing_header(label="Common Mistakes", default_open=True):
                            dpg.add_text("  WRONG: grid_data -> RL Agent", color=(255, 0, 0))
                            dpg.add_text("   Agents need env_instance, not just grid_data")
                            
                            dpg.add_text("  WRONG: Forgetting to connect Environment", color=(255, 0, 0))
                            dpg.add_text("   All agents need an Environment connection")
                            
                            dpg.add_text("  WRONG: IL Agent without Teacher", color=(255, 0, 0))
                            dpg.add_text("   IL Agent needs RL Agent as teacher")
                        
                        with dpg.collapsing_header(label="Workflow Examples"):
                            dpg.add_text("RL Training Workflow:")
                            dpg.add_text("  1. Settings -> Environment (config)")
                            dpg.add_text("  2. Environment -> RL Agent (env_instance)")
                            dpg.add_text("  3. Click 'Generate Terrain' on Environment")
                            dpg.add_text("  4. Click 'Train RL Agent' on RL Agent")
                            
                            dpg.add_spacer(height=10)
                            dpg.add_text("IL Training Workflow:")
                            dpg.add_text("  1. Environment -> RL Agent (env_instance)")
                            dpg.add_text("  2. Environment -> IL Agent (env_instance)")
                            dpg.add_text("  3. RL Agent -> IL Agent (teacher_agent)")
                            dpg.add_text("  4. Click 'Train IL Agent' on IL Agent")
                            
                            dpg.add_spacer(height=10)
                            dpg.add_text("Evaluation Workflow:")
                            dpg.add_text("  1. Environment -> Visualizer (env_instance)")
                            dpg.add_text("  2. Agent -> Visualizer (agent_instance)")
                            dpg.add_text("  3. Click 'Evaluate Agent' on Visualizer")
                
                # Troubleshooting Tab
                with dpg.tab(label="Troubleshooting"):
                    with dpg.group():
                        dpg.add_text("Common Issues and Solutions", color=(0, 255, 255))
                        dpg.add_separator()
                        
                        with dpg.collapsing_header(label="Agent Training Issues", default_open=True):
                            dpg.add_text("Problem: 'Error: Connect Environment first!'", color=(255, 100, 100))
                            dpg.add_text("Solution: Connect env_instance from Environment to agent")
                            
                            dpg.add_spacer(height=5)
                            dpg.add_text("Problem: Training fails or gets stuck", color=(255, 100, 100))
                            dpg.add_text("Solution: Reduce grid size or increase episodes")
                            
                            dpg.add_spacer(height=5)
                            dpg.add_text("Problem: IL Agent says 'Teacher agent not trained!'", color=(255, 100, 100))
                            dpg.add_text("Solution: Train RL Agent first, then connect to IL Agent")
                        
                        with dpg.collapsing_header(label="Visualization Issues"):
                            dpg.add_text("Problem: Visualizer shows blank/black screen", color=(255, 100, 100))
                            dpg.add_text("Solution: Generate terrain first, then visualize")
                            
                            dpg.add_spacer(height=5)
                            dpg.add_text("Problem: No path shown on visualization", color=(255, 100, 100))
                            dpg.add_text("Solution: Train/evaluate agent first or find path")
                        
                        with dpg.collapsing_header(label="Connection Issues"):
                            dpg.add_text("Problem: Can't connect nodes", color=(255, 100, 100))
                            dpg.add_text("Solution: Drag from output (right) to input (left)")
                            
                            dpg.add_spacer(height=5)
                            dpg.add_text("Problem: Connection disappears", color=(255, 100, 100))
                            dpg.add_text("Solution: Right-Alt + click to delete, reconnect")
                            
                            dpg.add_spacer(height=5)
                            dpg.add_text("Problem: Wrong connection type", color=(255, 100, 100))
                            dpg.add_text("Solution: Check 'Connections' tab for correct mappings")
                        
                        with dpg.collapsing_header(label="Performance Issues"):
                            dpg.add_text("Problem: Training takes too long", color=(255, 100, 100))
                            dpg.add_text("Solution: Reduce grid size, episodes, or terrains")
                            
                            dpg.add_spacer(height=5)
                            dpg.add_text("Problem: App is slow/laggy", color=(255, 100, 100))
                            dpg.add_text("Solution: Close unused nodes, reduce grid complexity")

    def show_connection_guide(self):
        if dpg.does_item_exist("ConnectionGuideWindow"):
            dpg.show_item("ConnectionGuideWindow")
            return
        
        with dpg.window(label="Node Connection Guide", tag="ConnectionGuideWindow", 
                       width=700, height=500, show=True, pos=[200, 150]):
            
            dpg.add_text("Essential Node Connections", color=(0, 255, 255))
            dpg.add_separator()
            
            with dpg.table(header_row=True, borders_innerH=True, borders_outerH=True, 
                          borders_innerV=True, borders_outerV=True):
                dpg.add_table_column(label="From Node")
                dpg.add_table_column(label="Output")
                dpg.add_table_column(label="To Node")
                dpg.add_table_column(label="Input")
                dpg.add_table_column(label="Purpose")
                
                # Environment connections
                with dpg.table_row():
                    dpg.add_text("Environment")
                    dpg.add_text("env_instance", color=(0, 255, 0))
                    dpg.add_text("RL Agent")
                    dpg.add_text("env_instance", color=(0, 255, 0))
                    dpg.add_text("Training environment")
                
                with dpg.table_row():
                    dpg.add_text("Environment")
                    dpg.add_text("env_instance", color=(0, 255, 0))
                    dpg.add_text("IL Agent")
                    dpg.add_text("env_instance", color=(0, 255, 0))
                    dpg.add_text("Training environment")
                
                with dpg.table_row():
                    dpg.add_text("Environment")
                    dpg.add_text("env_instance", color=(0, 255, 0))
                    dpg.add_text("PathFinder")
                    dpg.add_text("env_instance", color=(0, 255, 0))
                    dpg.add_text("Path finding")
                
                with dpg.table_row():
                    dpg.add_text("Environment")
                    dpg.add_text("env_instance", color=(0, 255, 0))
                    dpg.add_text("Visualizer")
                    dpg.add_text("env_instance", color=(0, 255, 0))
                    dpg.add_text("Visualization")
                
                # Agent connections
                with dpg.table_row():
                    dpg.add_text("RL Agent")
                    dpg.add_text("output_agent", color=(255, 255, 0))
                    dpg.add_text("IL Agent")
                    dpg.add_text("teacher_agent", color=(255, 255, 0))
                    dpg.add_text("Teacher for imitation")
                
                with dpg.table_row():
                    dpg.add_text("Any Agent")
                    dpg.add_text("output_agent", color=(255, 255, 0))
                    dpg.add_text("Visualizer")
                    dpg.add_text("agent_instance", color=(255, 255, 0))
                    dpg.add_text("Evaluation")
                
                # Settings connections
                with dpg.table_row():
                    dpg.add_text("Settings")
                    dpg.add_text("output_config", color=(0, 200, 255))
                    dpg.add_text("Environment")
                    dpg.add_text("config", color=(0, 200, 255))
                    dpg.add_text("Configuration")
                
                with dpg.table_row():
                    dpg.add_text("Settings")
                    dpg.add_text("output_config", color=(0, 200, 255))
                    dpg.add_text("RL/IL Agent")
                    dpg.add_text("config", color=(0, 200, 255))
                    dpg.add_text("Training settings")
            
            dpg.add_separator()
            dpg.add_text(" IMPORTANT: Always connect env_instance to agents, NOT grid_data!", color=(255, 100, 100))
            dpg.add_text("env_instance contains: grid, positions, trajectory, and all methods")
            dpg.add_text("grid_data is only the raw grid array (for special cases)")

    def show_troubleshooting(self):
        if dpg.does_item_exist("TroubleshootingWindow"):
            dpg.show_item("TroubleshootingWindow")
            return
        
        with dpg.window(label="Troubleshooting Guide", tag="TroubleshootingWindow", 
                       width=600, height=400, show=True, pos=[250, 200]):
            
            dpg.add_text("Quick Fixes for Common Problems", color=(0, 255, 255))
            dpg.add_separator()
            
            with dpg.tree_node(label="Connection Problems", default_open=True):
                dpg.add_text("  'Connect Environment first!'", color=(255, 100, 100))
                dpg.add_text("Fix: Connect env_instance from Environment to agent")
                dpg.add_separator()
                
                dpg.add_text("  'Connect Teacher Agent first!'", color=(255, 100, 100))
                dpg.add_text("Fix: Connect trained RL Agent to IL Agent as teacher")
                dpg.add_separator()
                
                dpg.add_text("  Can't connect nodes", color=(255, 100, 100))
                dpg.add_text("Fix: Drag from output (right side) to input (left side)")
            
            with dpg.tree_node(label="Training Problems"):
                dpg.add_text("  Training fails immediately", color=(255, 100, 100))
                dpg.add_text("Fix: Check all required connections are made")
                dpg.add_separator()
                
                dpg.add_text("  Training takes forever", color=(255, 100, 100))
                dpg.add_text("Fix: Reduce grid size or training episodes")
                dpg.add_separator()
                
                dpg.add_text("  Poor training results", color=(255, 100, 100))
                dpg.add_text("Fix: Increase episodes or use simpler terrain")
            
            with dpg.tree_node(label="Visualization Problems"):
                dpg.add_text("  Blank/black visualization", color=(255, 100, 100))
                dpg.add_text("Fix: Generate terrain first, then click Visualize")
                dpg.add_separator()
                
                dpg.add_text("  No path shown", color=(255, 100, 100))
                dpg.add_text("Fix: Train agent or find path before visualizing")
            
            with dpg.tree_node(label="General Tips"):
                dpg.add_text("Start simple: Small grid, default settings")
                dpg.add_text("Follow workflow: Generate -> Train -> Visualize")
                dpg.add_text("Check status messages on nodes")
                dpg.add_text("Use 'Clear All Nodes' to start fresh if stuck")

    def show_about(self):
        if dpg.does_item_exist("AboutWindow"):
            dpg.show_item("AboutWindow")
            return
        
        with dpg.window(label="About Robot Simulator", tag="AboutWindow", 
                       width=500, height=300, show=True, pos=[350, 250]):
            
            dpg.add_text("Robot Terrain Learning Node Editor", color=(0, 255, 255))
            dpg.add_text("Version 1.0", color=(200, 200, 200))
            dpg.add_separator()
            
            dpg.add_text("A node-based simulator for training and evaluating")
            dpg.add_text("robotic navigation algorithms.")
            dpg.add_spacer(height=10)
            
            dpg.add_text("Features:", color=(0, 255, 0))
            dpg.add_text("  * Q-Learning Reinforcement Learning")
            dpg.add_text("  * Imitation Learning from expert agents")
            dpg.add_text("  * Path Finding (A*, BFS, DFS)")
            dpg.add_text("  * Terrain and Labyrinth generation")
            dpg.add_text("  * Visual node-based interface")
            
            dpg.add_spacer(height=10)
            dpg.add_text("Created with DearPyGui", color=(200, 200, 200))
            dpg.add_text("For educational and research purposes")

    def generate_terrain(self, node_id, grid_size=None):
        if node_id not in self.environments:
            print(f"Error: Environment {node_id} not found")
            return
        
        env = self.environments[node_id]
        current_start = env.start if hasattr(env, 'start') else None
        current_goal = env.goal if hasattr(env, 'goal') else None
        
        if grid_size:
            env.set_size(grid_size)
        
        env.generate_random()
        
        if current_start and current_goal:
            env.set_pos(current_start, current_goal)

    def train_rl(self, env_id, agent_id, settings_config=None):
        if self.is_training:
            return
        
        self.is_training = True
        env = self.environments.get(env_id)
        
        if not env:
            self.is_training = False
            return
        
        from trainer import train_rl_agent
        
        episodes = settings_config.get('rl_episodes', self.settings.RL_EPISODES) if settings_config else self.settings.RL_EPISODES
        max_steps = settings_config.get('max_steps', self.settings.MAX_STEPS) if settings_config else self.settings.MAX_STEPS
        timeout = settings_config.get('timeout', self.settings.TIMEOUT) if settings_config else self.settings.TIMEOUT
        
        try:
            rl_agent, rewards, success, train_time = train_rl_agent(env, episodes, max_steps, timeout)
            if success:
                self.rl_agents[agent_id] = rl_agent
                print(f"General RL Agent {agent_id} trained successfully")
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
            
            from rl_agent import GeneralQLearningAgent
            import numpy as np
            
            episodes = settings_config.get('rl_episodes', self.settings.RL_EPISODES) if settings_config else self.settings.RL_EPISODES
            max_steps = settings_config.get('max_steps', self.settings.MAX_STEPS) if settings_config else self.settings.MAX_STEPS
            
            agent = GeneralQLearningAgent(alpha=0.1, gamma=0.99, epsilon=0.3)
            total_episodes = 0
            rewards_history = []
            success_history = []
            
            for terrain_idx in range(num_terrains):
                print(f"\nTerrain {terrain_idx + 1}/{num_terrains}")
                env.generate_random()
                
                terrain_episodes = episodes // num_terrains if num_terrains > 0 else episodes
                if terrain_episodes < 10:
                    terrain_episodes = 10
                
                for episode in range(terrain_episodes):
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
                    total_episodes += 1
                    agent.update_epsilon(total_episodes, episodes)
                    
                    if total_episodes % 50 == 0:
                        avg_reward = np.mean(rewards_history[-50:]) if len(rewards_history) >= 50 else np.mean(rewards_history)
                        success_rate = np.mean(success_history[-50:]) * 100 if success_history else 0
                        print(f"  Episode {total_episodes}: Avg reward: {avg_reward:.2f}, Success: {success_rate:.1f}%")
            
            avg_reward = np.mean(rewards_history)
            success_rate = np.mean(success_history) * 100 if success_history else 0
            
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
        
        from trainer import train_il_agent
        
        episodes = settings_config.get('il_episodes', self.settings.IL_EPISODES) if settings_config else self.settings.IL_EPISODES
        timeout = settings_config.get('timeout', self.settings.TIMEOUT) if settings_config else self.settings.TIMEOUT

        try:
            il_agent, success, total_time, train_time = train_il_agent(env, teacher_agent, episodes, timeout)
            if success:
                self.il_agents[il_agent_id] = il_agent
                print(f"General IL Agent {il_agent_id} trained successfully")
                stats = il_agent.get_statistics()
                print(f"  Demonstrations: {stats.get('demonstrations_collected', 0)}")
                print(f"  Accuracy: {stats.get('train_accuracy', 0):.3f}")
                print(f"  Terrains used: {stats.get('terrains_used', 0)}")
            else:
                print(f"IL Agent {il_agent_id} training failed")
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
            print(f"\nEvaluating {agent_type} agent {agent_id}...")
            print(f"Testing on {num_runs} different terrains...")
            
            from trainer import evaluate_agent
            success_rate, avg_reward, avg_steps, eval_time, trajectory = evaluate_agent(env, agent, agent_type, num_runs)
            
            results = {
                'success_rate': success_rate,
                'avg_reward': avg_reward,
                'avg_steps': avg_steps,
                'eval_time': eval_time
            }
            
            print(f"\nEvaluation complete for {agent_type} agent")
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
            print(f"\nPathFinder agent {agent_id} finding path...")
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