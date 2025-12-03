import dearpygui.dearpygui as dpg
from node_editor import NodeEditorApp
from rl_agent import QLearningAgent
from il_agent import ImitationLearner
from trainer import train_rl_agent, train_il_agent, evaluate_agent
from settings import GRID_SIZE, RL_EPISODES, IL_EPISODES, MAX_STEPS, TIMEOUT

class GlobalSettings:
    def __init__(self):
        self.GRID_SIZE = GRID_SIZE
        self.RL_EPISODES = RL_EPISODES
        self.IL_EPISODES = IL_EPISODES
        self.MAX_STEPS = MAX_STEPS
        self.TIMEOUT = TIMEOUT

SETTINGS = GlobalSettings()

class RobotSimulatorApp:
    def __init__(self):
        self.environments = {}
        self.rl_agents = {}
        self.il_agents = {}
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
    
    def generate_terrain(self, env_id, grid_size=None):
        env = self.environments.get(env_id)
        if not env:
            print(f"Error: Environment {env_id} not found.")
            return None
            
        if grid_size is not None and env.size != grid_size:
             env.set_size(size=grid_size)

        env.generate_random()
        return env.get_grid()

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
            rl_agent, rewards, success, train_time = train_rl_agent(env, episodes, max_steps, timeout)
            if success:
                self.rl_agents[agent_id] = rl_agent
            return rl_agent
        except Exception:
            return None
        finally:
            self.is_training = False

    def train_il(self, env_id, il_agent_id, teacher_agent_id, settings_config=None):
        if self.is_training: return
        self.is_training = True

        env = self.environments.get(env_id)
        teacher_agent = self.rl_agents.get(teacher_agent_id)
        
        if not env or not teacher_agent:
            self.is_training = False
            return
            
        episodes = settings_config.get('il_episodes', self.settings.IL_EPISODES) if settings_config else self.settings.IL_EPISODES
        timeout = settings_config.get('timeout', self.settings.TIMEOUT) if settings_config else self.settings.TIMEOUT

        try:
            il_agent, success, total_time, train_time = train_il_agent(env, teacher_agent, episodes, timeout)
            if success:
                self.il_agents[il_agent_id] = il_agent
            return il_agent
        except Exception:
            return None
        finally:
            self.is_training = False
    
    def evaluate(self, env_id, agent_id, agent_type, num_runs=1):
        env = self.environments.get(env_id)
        agent = self.rl_agents.get(agent_id) if agent_type == "RL" else self.il_agents.get(agent_id)

        if not env or not agent:
            return None, None
        
        try:
            success_rate, avg_reward, avg_steps, eval_time = evaluate_agent(env, agent, agent_type)
            results = {
                'success_rate': success_rate,
                'avg_reward': avg_reward,
                'avg_steps': avg_steps,
                'eval_time': eval_time
            }
            trajectory = getattr(env, 'trajectory', [])
            return results, trajectory
        except Exception:
            return None, None

    def get_agent(self, agent_id):
        return self.rl_agents.get(agent_id) or self.il_agents.get(agent_id)
        
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