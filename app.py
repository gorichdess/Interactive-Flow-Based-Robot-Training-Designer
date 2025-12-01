import dearpygui.dearpygui as dpg
import numpy as np
import time
from collections import defaultdict

from node_editor import NodeEditorApp
from environment import TerrainEnv
from rl_agent import QLearningAgent
from il_agent import ImitationLearner
from trainer import train_rl_agent, train_il_agent, evaluate_agent
from renderer import grid_to_image, render_trajectory
from utils import set_global_seed
from settings import GRID_SIZE, RL_EPISODES, IL_EPISODES, GLOBAL_SEED, MAX_STEPS, TIMEOUT, WINDOW_SIZE
   
class RobotSimulatorApp:
    def __init__(self):
        set_global_seed(GLOBAL_SEED)
        self.env = TerrainEnv()
        self.rl_agent = None
        self.il_agent = None
        self.current_trajectory = None
        self.evaluation_results = {}
        
        self.node_editor = NodeEditorApp(self)

        # Texture tags
        self.texture_tag_terrain = "texture_terrain"
        self.texture_tag_output = "texture_output"
        
        # Window tags
        self.controls_window_tag = "ControlsWindow"
        self.settings_window_tag = "SettingsWindow"
        self.output_window_tag = "OutputWindow"
        self.main_window_tag = "MainWindow"
        self.node_editor_window_tag = "NodeEditorWindow"
        self.results_window_tag = "ResultsWindow"
        
        # Status variables
        self.is_training = False
        self.training_progress = 0

    def setup(self):
        dpg.create_context()
        dpg.create_viewport(title="Robot Terrain Learning", width=1200, height=900)
        
        self.create_menu_bar()
        self.create_main_window()
        self.create_output_window()
        self.create_controls_window()
        self.create_settings_window()
        self.create_results_window() 

        self.node_editor.create_node_editor_window()

    def create_main_window(self):
        with dpg.window(label="Terrain Visualization", tag=self.main_window_tag, 
                       pos=(0, 30), width=400, height=400):
            with dpg.texture_registry():
                dpg.add_raw_texture(
                    width=WINDOW_SIZE, 
                    height=WINDOW_SIZE, 
                    default_value=np.ones((WINDOW_SIZE * WINDOW_SIZE * 3)) * 0.8, 
                    format=dpg.mvFormat_Float_rgb,
                    tag=self.texture_tag_terrain
                )
            dpg.add_image(self.texture_tag_terrain)
            dpg.add_text("Initial terrain map", tag="terrain_status")

    def create_output_window(self):
        with dpg.window(label="Agent Path Visualization", tag=self.output_window_tag,
                       pos=(0, 440), width=400, height=400):
            with dpg.texture_registry():
                dpg.add_raw_texture(
                    width=WINDOW_SIZE, 
                    height=WINDOW_SIZE, 
                    default_value=np.ones((WINDOW_SIZE * WINDOW_SIZE * 3)) * 0.8,
                    format=dpg.mvFormat_Float_rgb,
                    tag=self.texture_tag_output
                )
            dpg.add_image(self.texture_tag_output)
            dpg.add_text("Agent path will appear here", tag="output_status")

    def create_controls_window(self):
        with dpg.window(label="Controls", tag=self.controls_window_tag,
                       pos=(410, 30), width=350, height=250):
            dpg.add_button(label="Generate Random Terrain", 
                          callback=self.generate_random_terrain, width=300)
            dpg.add_separator()
            dpg.add_button(label="Train RL Agent", 
                          callback=self.train_rl, width=300)
            dpg.add_button(label="Train IL Agent", 
                          callback=self.train_il, width=300)
            dpg.add_separator()
            dpg.add_button(label="Evaluate RL Agent", 
                          callback=lambda: self.evaluate("RL"), width=300)
            dpg.add_button(label="Evaluate IL Agent", 
                          callback=lambda: self.evaluate("IL"), width=300)
            
            # Progress bar
            dpg.add_separator()
            dpg.add_text("Training Progress:")
            dpg.add_progress_bar(default_value=0.0, tag="progress_bar", width=300)
            dpg.add_text("Ready", tag="status_text")

    def create_settings_window(self):
        with dpg.window(label="Settings", tag=self.settings_window_tag,
                       pos=(410, 300), width=350, height=300):
            self.grid_size_input = dpg.add_input_int(
                label="Grid Size", 
                default_value=GRID_SIZE,
                min_value=5,
                min_clamped=True,
                max_value=50,
                max_clamped=True,
                width=150
            )
            self.max_steps_input = dpg.add_input_int(
                label="Max Steps", 
                default_value=MAX_STEPS,
                width=150
            )
            self.rl_episodes_input = dpg.add_input_int(
                label="RL Episodes", 
                default_value=RL_EPISODES,
                width=150
            )
            self.il_episodes_input = dpg.add_input_int(
                label="IL Episodes", 
                default_value=IL_EPISODES,
                width=150
            )
            self.timeout_input = dpg.add_input_int(
                label="Timeout (s)", 
                default_value=TIMEOUT,
                width=150
            )
            self.seed_input = dpg.add_input_int(
                label="Global Seed", 
                default_value=GLOBAL_SEED,
                width=150
            )
            
            dpg.add_spacer(height=10)
            dpg.add_button(label="Apply Settings", 
                          callback=self.apply_settings, width=150)

    def create_results_window(self):
        with dpg.window(label="Results", tag=self.results_window_tag,
                       pos=(770, 30), width=400, height=400):
            dpg.add_text("Evaluation Results:", tag="results_title")
            dpg.add_separator()
            
            with dpg.table(header_row=True, tag="results_table", borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True):
                dpg.add_table_column(label="Metric")
                dpg.add_table_column(label="RL Agent")
                dpg.add_table_column(label="IL Agent")
                
                # Add rows for different metrics
                metrics = ["Success_Rate", "Avg_Reward", "Avg_Steps", "Eval_Time"]
                for metric in metrics:
                    with dpg.table_row():
                        dpg.add_text(metric.replace('_', ' '))
                        dpg.add_text("-", tag=f"rl_{metric}")
                        dpg.add_text("-", tag=f"il_{metric}")

    def create_menu_bar(self):
        with dpg.viewport_menu_bar():
            with dpg.menu(label="Windows"):
                dpg.add_menu_item(label="Terrain Visualization Window", callback=self.show_main_window)
                dpg.add_menu_item(label="Controls Window", callback=self.show_controls_window)
                dpg.add_menu_item(label="Agent Path Visualization Window", callback=self.show_output_window)
                dpg.add_menu_item(label="Settings Window", callback=self.show_settings_window)
                dpg.add_menu_item(label="Result Window", callback=self.show_results_window)
                dpg.add_menu_item(label="Node Edit Window", callback=self.show_node_editor_window)

                dpg.add_separator()
                dpg.add_menu_item(label="Show All Windows", callback=self.show_all_windows)
                dpg.add_menu_item(label="Hide All Windows", callback=self.hide_all_windows)
                dpg.add_separator()
                dpg.add_menu_item(label="Reset Layout", callback=self.reset_layout)

            with dpg.menu(label="Node Editor"):
                dpg.add_menu_item(label="Add Environment Node", 
                                callback=lambda: self.node_editor.add_environment_node())
                dpg.add_menu_item(label="Add RL Agent Node", 
                                callback=lambda: self.node_editor.add_rl_agent_node())
                dpg.add_menu_item(label="Add IL Agent Node", 
                                callback=lambda: self.node_editor.add_il_agent_node())
                dpg.add_menu_item(label="Add Visualizer Node", 
                                callback=lambda: self.node_editor.add_visualizer_node())
                dpg.add_separator()
                dpg.add_menu_item(label="Clear All Nodes", 
                                callback=self.node_editor.clear_all_nodes)
            
            with dpg.menu(label="Help"):
                dpg.add_menu_item(label="About", callback=self.show_about)

    def show_all_windows(self):
        windows = [self.main_window_tag, self.output_window_tag, self.controls_window_tag,
                  self.settings_window_tag, self.results_window_tag, self.node_editor_window_tag]
        for window in windows:
            dpg.show_item(window)

    def hide_all_windows(self):
        windows = [self.main_window_tag, self.output_window_tag, self.controls_window_tag,
                  self.settings_window_tag, self.results_window_tag, self.node_editor_window_tag]
        for window in windows:
            dpg.hide_item(window)
    
    def show_main_window(self):
        dpg.show_item(self.main_window_tag)

    def show_controls_window(self):
        dpg.show_item(self.controls_window_tag)

    def show_output_window(self):
        dpg.show_item(self.output_window_tag)

    def show_settings_window(self):
        dpg.show_item(self.settings_window_tag)

    def show_results_window(self):
        dpg.show_item(self.results_window_tag)

    def show_node_editor_window(self):
        dpg.show_item(self.node_editor_window_tag)

    def reset_layout(self):
        # Reset window positions to default
        dpg.set_item_pos(self.main_window_tag, [0, 30])
        dpg.set_item_pos(self.output_window_tag, [0, 440])
        dpg.set_item_pos(self.controls_window_tag, [410, 30])
        dpg.set_item_pos(self.settings_window_tag, [410, 300])
        dpg.set_item_pos(self.results_window_tag, [770, 30])
        dpg.set_item_pos(self.node_editor_window_tag, [410, 620])

    def show_about(self):
        with dpg.window(label="About", width=400, height=300):
            dpg.add_text("Robot Terrain Learning Simulator")
            dpg.add_text("Reinforcement Learning and Imitation Learning")
            dpg.add_text("for Terrain Navigation")

    def update_progress(self, value, status=""):
        dpg.set_value("progress_bar", value)
        if status:
            dpg.set_value("status_text", status)

    def generate_random_terrain(self):
        self.update_progress(0, "Generating terrain...")
        self.env.generate_random()
        grid = self.env.get_grid()
        
        # Update terrain visualization
        scale = WINDOW_SIZE // GRID_SIZE
        image_data = grid_to_image(grid, scale=scale, target_size=WINDOW_SIZE)
        dpg.set_value(self.texture_tag_terrain, image_data.flatten())
        
        # Clear output
        dpg.set_value(self.texture_tag_output, np.ones((WINDOW_SIZE * WINDOW_SIZE * 3)) * 0.8)
        dpg.set_value("terrain_status", f"Terrain generated: {self.env.size}x{self.env.size}")
        dpg.set_value("output_status", "Ready for training and evaluation")
        
        self.update_progress(1.0, "Terrain generated!")
        time.sleep(0.5)
        self.update_progress(0.0, "Ready")

    def train_rl(self):
        if self.is_training:
            return
            
        self.is_training = True
        try:
            self.update_progress(0, "RL Training started...")
            
            # Get settings
            episodes = dpg.get_value(self.rl_episodes_input)
            max_steps = dpg.get_value(self.max_steps_input)
            timeout = dpg.get_value(self.timeout_input)
            
            self.rl_agent, rewards, success, train_time = train_rl_agent(
                self.env, episodes, max_steps, timeout
            )
            
            if success:
                self.update_progress(1.0, f"RL Training completed in {train_time:.2f}s")
                dpg.set_value("status_text", f"RL trained: {len(rewards)} episodes")
            else:
                self.update_progress(0, f"RL Training failed: timeout")
                
        except Exception as e:
            self.update_progress(0, f"RL Training error: {str(e)}")
        finally:
            self.is_training = False

    def train_il(self):
        if not self.rl_agent:
            self.update_progress(0, "Error: Train RL agent first!")
            return
            
        if self.is_training:
            return
            
        self.is_training = True
        try:
            self.update_progress(0, "IL Training started...")
            
            # Get settings
            episodes = dpg.get_value(self.il_episodes_input)
            timeout = dpg.get_value(self.timeout_input)
            
            self.il_agent, success, total_time, train_time = train_il_agent(
                self.env, self.rl_agent, episodes, timeout
            )
            
            if success:
                self.update_progress(1.0, f"IL Training completed in {total_time:.2f}s")
                dpg.set_value("status_text", f"IL trained in {train_time:.2f}s")
            else:
                self.update_progress(0, f"IL Training failed: timeout")
                
        except Exception as e:
            self.update_progress(0, f"IL Training error: {str(e)}")
        finally:
            self.is_training = False

    def evaluate_all(self):
        if self.rl_agent:
            self.evaluate("RL")
        if self.il_agent:
            self.evaluate("IL")

    def update_results_display(self):
        for agent_type in ["RL", "IL"]:
            if agent_type in self.evaluation_results:
                results = self.evaluation_results[agent_type]
                prefix = agent_type.lower()
                
                dpg.set_value(f"{prefix}_Success_Rate", f"{results['success_rate']:.1f}%")
                dpg.set_value(f"{prefix}_Avg_Reward", f"{results['avg_reward']:.2f}")
                dpg.set_value(f"{prefix}_Avg_Steps", f"{results['avg_steps']:.1f}")
                dpg.set_value(f"{prefix}_Eval_Time", f"{results['eval_time']:.2f}s")

    def visualize_agent_path(self, agent, agent_type):
        try:
            steps = 0
            
            if hasattr(self.env, 'trajectory'):
                trajectory = self.env.trajectory
            else:
                # Fallback: create a simple trajectory from start to current position
                trajectory = [self.env.start]
                if steps > 0:
                    # Try to get current position
                    if hasattr(self.env, 'position'):
                        trajectory.append(self.env.position)
            
            print(f"Visualizing {agent_type} agent trajectory: {len(trajectory)} steps")
            print(f"Start: {trajectory[0] if trajectory else 'None'}")
            print(f"End: {trajectory[-1] if trajectory else 'None'}")
            
            # Get terrain grid
            grid = self.env.get_grid()
            
            # Render trajectory on terrain
            scale = WINDOW_SIZE // self.env.size
            image_data = render_trajectory(
                grid, 
                trajectory, 
                scale=scale, 
                target_size=WINDOW_SIZE,
                line_width=3
            )
            
            # Update texture
            dpg.set_value(self.texture_tag_output, image_data.flatten())
            
            dpg.set_value("output_status", 
                        f"{agent_type} Agent Path: {len(trajectory)} steps, "
                        f"Reached goal: {hasattr(self.env, 'reached_goal') and self.env.reached_goal}")
                            
        except Exception as e:
            print(f"Visualization error: {e}")
            import traceback
            traceback.print_exc()
            dpg.set_value("output_status", f"Visualization error: {str(e)}")

    # Also update the evaluate method to store trajectory
    def evaluate(self, agent_type):
        agent = self.rl_agent if agent_type == "RL" else self.il_agent
        if not agent:
            self.update_progress(0, f"Error: {agent_type} agent not trained!")
            return

        self.update_progress(0, f"Evaluating {agent_type} agent...")
        
        try:
            success_rate, avg_reward, avg_steps, eval_time = evaluate_agent(
                self.env, agent, agent_type)
            
            # Store results
            self.evaluation_results[agent_type] = {
                'success_rate': success_rate,
                'avg_reward': avg_reward,
                'avg_steps': avg_steps,
                'eval_time': eval_time
            }
            
            # Update results table
            self.update_results_display()
            
            # Visualize trajectory from the last evaluation run
            self.visualize_agent_path(agent, agent_type)
            
            self.update_progress(1.0, f"{agent_type} evaluation completed!")
            
        except Exception as e:
            self.update_progress(0, f"Evaluation error: {str(e)}")

    def apply_settings(self):
        global GRID_SIZE, RL_EPISODES, IL_EPISODES, MAX_STEPS, TIMEOUT, GLOBAL_SEED

        new_grid_size = dpg.get_value(self.grid_size_input)
        new_max_steps = dpg.get_value(self.max_steps_input)
        new_rl_episodes = dpg.get_value(self.rl_episodes_input)
        new_il_episodes = dpg.get_value(self.il_episodes_input)
        new_timeout = dpg.get_value(self.timeout_input)
        new_seed = dpg.get_value(self.seed_input)

        # Update global settings
        GRID_SIZE = new_grid_size
        MAX_STEPS = new_max_steps
        RL_EPISODES = new_rl_episodes
        IL_EPISODES = new_il_episodes
        TIMEOUT = new_timeout
        GLOBAL_SEED = new_seed

        # Update environment
        self.env.set_size(size=GRID_SIZE)
        self.env.reset()
        set_global_seed(GLOBAL_SEED)
        
        self.update_progress(0, "Settings applied!")
        print(f"Settings updated: GRID_SIZE={GRID_SIZE}, MAX_STEPS={MAX_STEPS}")

    def run(self):
        self.setup()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()


if __name__ == "__main__":
    app = RobotSimulatorApp()
    app.run()