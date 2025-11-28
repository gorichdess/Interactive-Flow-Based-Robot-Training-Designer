import dearpygui.dearpygui as dpg
import numpy as np

from environment import TerrainEnv
from rl_agent import QLearningAgent
from il_agent import ImitationLearner
from trainer import train_rl_agent, train_il_agent, evaluate_agent
from renderer import grid_to_image
from utils import set_global_seed
from settings import GRID_SIZE, RL_EPISODES, IL_EPISODES, GLOBAL_SEED, MAX_STEPS, TIMEOUT, WINDOW_SIZE


class RobotSimulatorApp:
    def __init__(self):
        set_global_seed(GLOBAL_SEED)
        self.env = TerrainEnv()
        self.rl_agent = None
        self.il_agent = None
        self.texture_tag = "texture"

    def setup(self):
        dpg.create_context()
        dpg.create_viewport(title="Robot Terrain Learning", width=1000, height=820)

        self.create_main_window()
        self.create_controls_window()
        self.create_settings_window()
        
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()
        
    def create_main_window(self):
        self.env.generate_random()
        with dpg.window(label="Terrain Visualization", pos=(0, 0), width=610, height=810):
            #Creating a texture for an image
            with dpg.texture_registry():
                #Create an empty texture
                self.texture_tag = dpg.add_raw_texture(
                    width=WINDOW_SIZE, 
                    height=WINDOW_SIZE, 
                    default_value=[],
                    format=dpg.mvFormat_Float_rgb
                )
            
            #Displaying an image
            dpg.add_image(self.texture_tag)

    def create_controls_window(self):
        with dpg.window(label="Controls", pos=(620, 0), width=350, height=350):
            dpg.add_button(label="Generate Random Terrain", callback=self.generate_random_terrain)
            dpg.add_button(label="Train RL", callback=self.train_rl)
            dpg.add_button(label="Train IL", callback=self.train_il)
            dpg.add_button(label="Evaluate", callback=self.evaluate)

    def create_settings_window(self):
        with dpg.window(label="Settings", pos=(620,370), width=350, height=280):
            self.grid_size_input = dpg.add_input_int(
                label="Grid Size", 
                default_value=GRID_SIZE,
                min_value=2,        #The min size is limited
                min_clamped=True,
                max_value=WINDOW_SIZE, #The max size is limited
                max_clamped=True
            )
            self.max_steps_input = dpg.add_input_int(label="Max Steps", default_value=MAX_STEPS)
            self.rl_episodes_input = dpg.add_input_int(label="RL Episodes", default_value=RL_EPISODES)
            self.il_episodes_input = dpg.add_input_int(label="IL Episodes", default_value=IL_EPISODES)
            self.timeout_input = dpg.add_input_int(label="Timeout", default_value=TIMEOUT)
            self.seed_input = dpg.add_input_int(label="Global Seed", default_value=GLOBAL_SEED)

            dpg.add_button(label="Apply Settings", callback=self.apply_settings)

    def generate_random_terrain(self):  
        self.env.generate_random()
        grid = self.env.get_grid()
        scale = WINDOW_SIZE // GRID_SIZE
        
        image_data = grid_to_image(grid,scale=scale)
        
        #Updating the texture
        dpg.set_value(self.texture_tag, image_data.flatten())


    def train_rl(self):
        print("RL training started...")
        self.rl_agent, *_ = train_rl_agent(self.env, RL_EPISODES)

    def train_il(self):
        print("IL training started...")
        if not self.rl_agent:
            print("Train RL first!")
            return
        self.il_agent, *_ = train_il_agent(self.env, self.rl_agent, IL_EPISODES)

    def evaluate(self):
        print("Evaluation button pressed")

    def apply_settings(self):
        global GRID_SIZE, RL_EPISODES, IL_EPISODES, MAX_STEPS, TIMEOUT, GLOBAL_SEED

        GRID_SIZE = dpg.get_value(self.grid_size_input)
        self.env.set_size(size=GRID_SIZE)
        self.env.reset()
        MAX_STEPS = dpg.get_value(self.max_steps_input)
        RL_EPISODES = dpg.get_value(self.rl_episodes_input)
        IL_EPISODES = dpg.get_value(self.il_episodes_input)
        TIMEOUT = dpg.get_value(self.timeout_input)
        GLOBAL_SEED = dpg.get_value(self.seed_input)

        print(f"GRID_SIZE={GRID_SIZE}, MAX_STEPS={MAX_STEPS}, RL_EPISODES={RL_EPISODES}, IL_EPISODES={IL_EPISODES}, TIMEOUT={TIMEOUT}, GLOBAL_SEED={GLOBAL_SEED}")

    def run(self):
        self.setup()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()


if __name__ == "__main__":
    app = RobotSimulatorApp()
    app.run()
