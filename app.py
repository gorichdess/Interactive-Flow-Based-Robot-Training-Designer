import dearpygui.dearpygui as dpg

from environment import TerrainEnv
from rl_agent import QLearningAgent
from il_agent import ImitationLearner
from trainer import train_rl_agent, train_il_agent, evaluate_agent
from renderer import grid_to_image
from utils import set_global_seed
from settings import GRID_SIZE, RL_EPISODES, IL_EPISODES, GLOBAL_SEED


class RobotSimulatorApp:
    def __init__(self):
        set_global_seed(GLOBAL_SEED)
        self.env = TerrainEnv(size=GRID_SIZE)
        self.rl_agent = None
        self.il_agent = None

    def setup(self):
        dpg.create_context()
        dpg.create_viewport(title="Robot Terrain Learning", width=1000, height=700)

        self.create_main_window()
        self.create_controls_window()

    def create_main_window(self):
        with dpg.window(label="Terrain View", width=600, height=600):
            dpg.add_text("Here terrain image will be displayed")

    def create_controls_window(self):
        with dpg.window(label="Controls", pos=(620, 0), width=350, height=600):
            dpg.add_button(label="Generate Terrain", callback=self.generate_terrain)
            dpg.add_button(label="Train RL", callback=self.train_rl)
            dpg.add_button(label="Train IL", callback=self.train_il)
            dpg.add_button(label="Evaluate", callback=self.evaluate)

    def generate_terrain(self):
        print("Generating terrain...")
        self.env.reset()

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

    def run(self):
        self.setup()
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.start_dearpygui()
        dpg.destroy_context()


if __name__ == "__main__":
    app = RobotSimulatorApp()
    app.run()
