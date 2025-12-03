import dearpygui.dearpygui as dpg
import numpy as np
from enum import Enum
from environment import TerrainEnv
from settings import WINDOW_SIZE
from renderer import grid_to_image, render_trajectory

class NodeType(Enum):
    ENVIRONMENT = "Environment"
    RL_AGENT = "RL_Agent"
    IL_AGENT = "IL_Agent"
    SETTINGS = "Settings"
    VISUALIZER = "Visualizer"
    RESULTS = "Results"

class NodeEditorApp:
    def __init__(self, simulator_app):
        self.simulator = simulator_app
        self.nodes = {}
        self.links = {}
        
        self.node_editor_window_tag = "NodeEditorWindow"
        self.node_editor_tag = "NodeEditor"
        self.node_counter = 0
        self.attribute_to_node = {}
        self.node_objects = {}

    def create_node_editor_window(self):
        if dpg.does_item_exist(self.node_editor_window_tag):
            dpg.focus_item(self.node_editor_window_tag)
            return

        with dpg.window(label="Robot Simulator Node Editor", tag=self.node_editor_window_tag,
                         pos=(10, 10), width=1500, height=800, min_size=[600, 600]):
            
            dpg.add_text("Drag from output (Right) to input (Left) to connect. Right alt + click link to delete.")
            dpg.add_separator()
            
            with dpg.group(horizontal=True):
                dpg.add_button(label="Environment Node", callback=self.add_environment_node)
                dpg.add_button(label="RL Agent Node", callback=self.add_rl_agent_node)
                dpg.add_button(label="IL Agent Node", callback=self.add_il_agent_node)
                dpg.add_button(label="Visualizer Node", callback=self.add_visualizer_node)
                dpg.add_button(label="Settings Node", callback=self.add_settings_node)
                dpg.add_button(label="Results Node", callback=self.add_results_node)
                dpg.add_button(label="Clear All Nodes", callback=self.clear_all_nodes)
            
            dpg.add_separator()
            
            with dpg.node_editor(
                tag=self.node_editor_tag, 
                callback=self.link_callback,
                delink_callback=self.delink_callback,
                minimap=True,
                height=900 
            ):
                pass

    def link_callback(self, sender, app_data):
        output_attr_id, input_attr_id = app_data
        
        output_node_id = self.get_node_from_attribute_id(output_attr_id)
        input_node_id = self.get_node_from_attribute_id(input_attr_id)
        
        if not output_node_id or not input_node_id:
            return
        
        output_type = self.nodes[output_node_id]['output_types'].get(output_attr_id)
        input_type = self.nodes[input_node_id]['input_types'].get(input_attr_id)

        if output_type != input_type:
            print(f"Warning: Connecting incompatible types: {output_type} -> {input_type}")

        link_id = dpg.add_node_link(output_attr_id, input_attr_id, parent=sender)
        self.links[link_id] = (output_attr_id, input_attr_id)

    def delink_callback(self, sender, app_data):
        link_id = app_data
        if link_id in self.links:
            del self.links[link_id]
        dpg.delete_item(link_id)

    def get_node_from_attribute_id(self, attr_id):
        if attr_id in self.attribute_to_node:
            return self.attribute_to_node[attr_id]
        
        for node_id, node_data in self.nodes.items():
            for attr in node_data['all_attrs']:
                if attr == attr_id:
                    self.attribute_to_node[attr_id] = node_id
                    return node_id
        return None

    def find_connected_node_and_data(self, target_node_id, input_attribute_name):
        target_node_data = self.nodes.get(target_node_id)
        if not target_node_data: return None, None
        
        target_input_attr = target_node_data['input_attrs'].get(input_attribute_name)
        if not target_input_attr: return None, None
        
        for link_id, (output_attr_id, input_attr_id) in self.links.items():
            if input_attr_id == target_input_attr:
                output_node_id = self.get_node_from_attribute_id(output_attr_id)
                if output_node_id and output_node_id in self.node_objects:
                    return output_node_id, self.node_objects[output_node_id]
        return None, None

    def find_connected_output_value(self, target_node_id, input_attribute_name, output_key=None):
        output_node_id, output_node_obj = self.find_connected_node_and_data(target_node_id, input_attribute_name)
        if output_node_obj:
            if output_key:
                return output_node_obj.get_output_data().get(output_key)
            else:
                return output_node_obj.get_output_data() 
        return None

    def add_node(self, node_class, label, pos):
        node_id = f"{node_class.__name__.lower()}_{self.node_counter}"
        self.node_counter += 1
        
        node_obj = node_class(node_id, self.simulator, self)
        
        with dpg.node(tag=node_id, parent=self.node_editor_tag, label=label, pos=pos):
            input_attrs, output_attrs, static_attrs, all_attrs, input_types, output_types = node_obj.create_attributes()

        self.node_objects[node_id] = node_obj
        self.nodes[node_id] = {
            'type': node_obj.node_type,
            'input_attrs': input_attrs,
            'output_attrs': output_attrs,
            'static_attrs': static_attrs,
            'all_attrs': all_attrs,
            'input_types': input_types,
            'output_types': output_types
        }
        
        for attr in all_attrs:
            self.attribute_to_node[attr] = node_id
            
        return node_id

    def add_environment_node(self):
        self.add_node(EnvironmentNode, "Environment", [50, 50])

    def add_rl_agent_node(self):
        self.add_node(RLAgentNode, "RL Agent (Q-Learner)", [300, 50])

    def add_il_agent_node(self):
        self.add_node(ILAgentNode, "IL Agent (Imitation)", [300, 250])

    def add_visualizer_node(self):
        self.add_node(VisualizerNode, "Visualization", [600, 50])

    def add_settings_node(self):
        self.add_node(SettingsNode, "Settings", [50, 400])

    def add_results_node(self):
        self.add_node(ResultsNode, "Results Table", [850, 50])
        
    def clear_all_nodes(self):
        for link_id in list(self.links.keys()):
            dpg.delete_item(link_id)
        for node_id in list(self.nodes.keys()):
            dpg.delete_item(node_id)
            
        self.nodes = {}
        self.links = {}
        self.attribute_to_node = {}
        self.node_objects = {}
        self.simulator.environments = {}
        self.simulator.rl_agents = {}
        self.simulator.il_agents = {}
        self.node_counter = 0


class BaseNode:
    def __init__(self, node_id, simulator_app, node_editor_app):
        self.node_id = node_id
        self.simulator = simulator_app
        self.editor = node_editor_app
        self.node_type = None
        
    def create_attributes(self):
        raise NotImplementedError

    def get_output_data(self):
        raise NotImplementedError
        
    def get_input_data(self, input_attribute_name, output_key=None):
        return self.editor.find_connected_output_value(self.node_id, input_attribute_name, output_key)


class EnvironmentNode(BaseNode):
    def __init__(self, node_id, simulator_app, node_editor_app):
        super().__init__(node_id, simulator_app, node_editor_app)
        self.node_type = NodeType.ENVIRONMENT
        self.env_instance = TerrainEnv(size=self.simulator.settings.GRID_SIZE)
        self.simulator.environments[self.node_id] = self.env_instance
        self.output_grid_attr = None
        self.output_env_instance_attr = None
        
    def create_attributes(self):
        input_attrs, output_attrs, static_attrs, all_attrs = {}, {}, {}, []
        input_types, output_types = {}, {}

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, label="Config") as input_config_attr:
            dpg.add_text("Config (Optional)")
        input_attrs['config'] = input_config_attr
        input_types[input_config_attr] = 'Settings'
        all_attrs.append(input_config_attr)
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static) as button_attr:
            dpg.add_button(label="Generate Terrain", width=150, callback=self.generate_terrain_callback)
        static_attrs['button'] = button_attr
        all_attrs.append(button_attr)

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, label="Grid Data") as output_grid_attr:
            dpg.add_text("Grid Data", indent=80)
        output_attrs['output_grid'] = output_grid_attr
        output_types[output_grid_attr] = 'Grid Data'
        all_attrs.append(output_grid_attr)
        self.output_grid_attr = output_grid_attr
            
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, label="Env Instance") as output_env_instance_attr:
            dpg.add_text("Env Instance", indent=80)
        output_attrs['output_env_instance'] = output_env_instance_attr
        output_types[output_env_instance_attr] = 'Environment'
        all_attrs.append(output_env_instance_attr)
        self.output_env_instance_attr = output_env_instance_attr

        return input_attrs, output_attrs, static_attrs, all_attrs, input_types, output_types

    def generate_terrain_callback(self):
        config = self.get_input_data('config', 'all')
        grid_size = config.get('grid_size') if config else None
        self.simulator.generate_terrain(self.node_id, grid_size=grid_size)

    def get_output_data(self):
        return {
            self.output_grid_attr: self.env_instance.get_grid(),
            self.output_env_instance_attr: self.env_instance
        }

class RLAgentNode(BaseNode):
    def __init__(self, node_id, simulator_app, node_editor_app):
        super().__init__(node_id, simulator_app, node_editor_app)
        self.node_type = NodeType.RL_AGENT
        self.rl_agent_instance = None
        self.output_agent_attr = None
        
    def create_attributes(self):
        input_attrs, output_attrs, static_attrs, all_attrs = {}, {}, {}, []
        input_types, output_types = {}, {}

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, label="Environment") as input_env_attr:
            dpg.add_text("Environment")
        input_attrs['env_instance'] = input_env_attr
        input_types[input_env_attr] = 'Environment'
        all_attrs.append(input_env_attr)
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, label="Config") as input_config_attr:
            dpg.add_text("Config (Optional)")
        input_attrs['config'] = input_config_attr
        input_types[input_config_attr] = 'Settings'
        all_attrs.append(input_config_attr)

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static) as button_attr:
            dpg.add_button(label="Train RL Agent", width=150, callback=self.train_rl_callback)
        static_attrs['button'] = button_attr
        all_attrs.append(button_attr)

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, label="Trained Agent") as output_agent_attr:
            dpg.add_text("Trained Agent", indent=60)
        output_attrs['output_agent'] = output_agent_attr
        output_types[output_agent_attr] = 'RL Agent'
        all_attrs.append(output_agent_attr)
        self.output_agent_attr = output_agent_attr

        return input_attrs, output_attrs, static_attrs, all_attrs, input_types, output_types

    def train_rl_callback(self):
        env_node_id, _ = self.editor.find_connected_node_and_data(self.node_id, 'env_instance')
        config = self.get_input_data('config', 'all')
        
        if not env_node_id:
            print(f"[{self.node_id}] Error: Connect Environment first!")
            return

        agent = self.simulator.train_rl(env_node_id, self.node_id, settings_config=config)
        self.rl_agent_instance = agent

    def get_output_data(self):
        return { self.output_agent_attr: self.rl_agent_instance }

class ILAgentNode(BaseNode):
    def __init__(self, node_id, simulator_app, node_editor_app):
        super().__init__(node_id, simulator_app, node_editor_app)
        self.node_type = NodeType.IL_AGENT
        self.il_agent_instance = None
        self.output_agent_attr = None
        
    def create_attributes(self):
        input_attrs, output_attrs, static_attrs, all_attrs = {}, {}, {}, []
        input_types, output_types = {}, {}

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, label="Environment") as input_env_attr:
            dpg.add_text("Environment")
        input_attrs['env_instance'] = input_env_attr
        input_types[input_env_attr] = 'Environment'
        all_attrs.append(input_env_attr)
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, label="Teacher Agent") as input_teacher_attr:
            dpg.add_text("Teacher (RL Agent)")
        input_attrs['teacher_agent'] = input_teacher_attr
        input_types[input_teacher_attr] = 'RL Agent'
        all_attrs.append(input_teacher_attr)
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, label="Config") as input_config_attr:
            dpg.add_text("Config (Optional)")
        input_attrs['config'] = input_config_attr
        input_types[input_config_attr] = 'Settings'
        all_attrs.append(input_config_attr)

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static) as button_attr:
            dpg.add_button(label="Train IL Agent", width=150, callback=self.train_il_callback)
        static_attrs['button'] = button_attr
        all_attrs.append(button_attr)

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, label="Trained Agent") as output_agent_attr:
            dpg.add_text("Trained Agent", indent=60)
        output_attrs['output_agent'] = output_agent_attr
        output_types[output_agent_attr] = 'IL Agent'
        all_attrs.append(output_agent_attr)
        self.output_agent_attr = output_agent_attr

        return input_attrs, output_attrs, static_attrs, all_attrs, input_types, output_types

    def train_il_callback(self):
        env_node_id, _ = self.editor.find_connected_node_and_data(self.node_id, 'env_instance')
        teacher_node_id, _ = self.editor.find_connected_node_and_data(self.node_id, 'teacher_agent')
        config = self.get_input_data('config', 'all')
        
        if not env_node_id or not teacher_node_id:
            print(f"[{self.node_id}] Error: Connect Env and Teacher Agent first!")
            return

        agent = self.simulator.train_il(env_node_id, self.node_id, teacher_node_id, settings_config=config)
        self.il_agent_instance = agent

    def get_output_data(self):
        return { self.output_agent_attr: self.il_agent_instance }

class SettingsNode(BaseNode):
    def __init__(self, node_id, simulator_app, node_editor_app):
        super().__init__(node_id, simulator_app, node_editor_app)
        self.node_type = NodeType.SETTINGS
        self.inputs = {}
        self.output_config_attr = None
        self.default_settings = {
            'grid_size': self.simulator.settings.GRID_SIZE,
            'max_steps': self.simulator.settings.MAX_STEPS,
            'rl_episodes': self.simulator.settings.RL_EPISODES,
            'il_episodes': self.simulator.settings.IL_EPISODES,
            'timeout': self.simulator.settings.TIMEOUT,
        }
        
    def create_attributes(self):
        input_attrs, output_attrs, static_attrs, all_attrs = {}, {}, {}, []
        input_types, output_types = {}, {}

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static) as static_attr_settings:
            dpg.add_text("Simulation Settings", color=(255, 255, 0))
            self.inputs['grid_size'] = dpg.add_input_int(label="Grid Size", default_value=self.default_settings['grid_size'], min_value=5, max_value=50, min_clamped=True, max_clamped=True, width=150)
            self.inputs['max_steps'] = dpg.add_input_int(label="Max Steps", default_value=self.default_settings['max_steps'], width=150)
            self.inputs['rl_episodes'] = dpg.add_input_int(label="RL Episodes", default_value=self.default_settings['rl_episodes'], width=150)
            self.inputs['il_episodes'] = dpg.add_input_int(label="IL Episodes", default_value=self.default_settings['il_episodes'], width=150)
            self.inputs['timeout'] = dpg.add_input_int(label="Timeout (s)", default_value=self.default_settings['timeout'], width=150)
        static_attrs['settings'] = static_attr_settings
        all_attrs.append(static_attr_settings)

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, label="Config Output") as output_config_attr:
            dpg.add_text("Config Data", indent=80)
        output_attrs['output_config'] = output_config_attr
        output_types[output_config_attr] = 'Settings'
        all_attrs.append(output_config_attr)
        self.output_config_attr = output_config_attr

        return input_attrs, output_attrs, static_attrs, all_attrs, input_types, output_types

    def get_output_data(self):
        config_data = {key: dpg.get_value(tag) for key, tag in self.inputs.items()}
        return { self.output_config_attr: config_data, 'all': config_data }

class VisualizerNode(BaseNode):
    def __init__(self, node_id, simulator_app, node_editor_app):
        super().__init__(node_id, simulator_app, node_editor_app)
        self.node_type = NodeType.VISUALIZER
        self.texture_tag_terrain = f"texture_{node_id}_terrain"
        self.image_tag = f"image_{node_id}"
        self.status_tag = f"status_{node_id}"
        self.window_size = WINDOW_SIZE
        
    def create_attributes(self):
        input_attrs, output_attrs, static_attrs, all_attrs = {}, {}, {}, []
        input_types, output_types = {}, {}

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, label="Environment") as input_env_attr:
            dpg.add_text("Environment")
        input_attrs['env_instance'] = input_env_attr
        input_types[input_env_attr] = 'Environment'
        all_attrs.append(input_env_attr)
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, label="Agent") as input_agent_attr:
            dpg.add_text("Agent (RL or IL)")
        input_attrs['agent_instance'] = input_agent_attr
        input_types[input_agent_attr] = 'RL Agent'
        all_attrs.append(input_agent_attr)

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static) as button_attr:
            dpg.add_button(label="Visualize Terrain", width=150, callback=self.visualize_terrain_callback)
            dpg.add_button(label="Evaluate Agent", width=150, callback=self.evaluate_agent_callback)
        static_attrs['button'] = button_attr
        all_attrs.append(button_attr)
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static) as vis_output_attr:
            with dpg.texture_registry(show=False):
                dpg.add_raw_texture(
                    width=self.window_size, 
                    height=self.window_size, 
                    default_value=np.ones((self.window_size * self.window_size * 3)) * 0.8, 
                    format=dpg.mvFormat_Float_rgb,
                    tag=self.texture_tag_terrain
                )
            dpg.add_text("Visualization Output", color=(0, 255, 0))
            dpg.add_image(self.texture_tag_terrain, tag=self.image_tag)
            dpg.add_text("Awaiting data...", tag=self.status_tag)
        static_attrs['vis_output'] = vis_output_attr
        all_attrs.append(vis_output_attr)

        return input_attrs, output_attrs, static_attrs, all_attrs, input_types, output_types

    def visualize_terrain_callback(self):
        env_node_id, env_node_obj = self.editor.find_connected_node_and_data(self.node_id, 'env_instance')
        
        if not env_node_id:
            dpg.set_value(self.status_tag, "Error: Connect Environment first!")
            return
            
        env_instance = self.simulator.get_environment(env_node_id)
        agent_node_id, _ = self.editor.find_connected_node_and_data(self.node_id, 'agent_instance')
        trajectory = None
        agent_type = "Base"

        if agent_node_id:
            agent_type = "RL" if agent_node_id in self.simulator.rl_agents else "IL"
            trajectory = getattr(env_instance, 'trajectory', [])

        grid = env_instance.get_grid()
        scale = self.window_size // grid.shape[0]
        
        if trajectory:
            image_data = render_trajectory(grid, trajectory, scale=scale, target_size=self.window_size, line_width=3)
            status_text = f"{agent_type} Path: {len(trajectory)} steps. Goal: {getattr(env_instance, 'reached_goal', 'N/A')}"
        else:
            image_data = grid_to_image(grid, scale=scale, target_size=self.window_size)
            status_text = f"Terrain: {env_instance.size}x{env_instance.size}"

        dpg.set_value(self.texture_tag_terrain, image_data.flatten())
        dpg.set_value(self.status_tag, status_text)

    def evaluate_agent_callback(self):
        env_node_id, _ = self.editor.find_connected_node_and_data(self.node_id, 'env_instance')
        agent_node_id, _ = self.editor.find_connected_node_and_data(self.node_id, 'agent_instance')

        if not env_node_id or not agent_node_id:
            dpg.set_value(self.status_tag, "Error: Connect Environment AND Agent first!")
            return
            
        agent_type = "RL" if agent_node_id in self.simulator.rl_agents else "IL"
        results, trajectory = self.simulator.evaluate(env_node_id, agent_node_id, agent_type)

        if results:
            dpg.set_value(self.status_tag, f"Eval Success: {results['success_rate']:.1f}%, Avg. Reward: {results['avg_reward']:.2f}")
            self.visualize_terrain_callback()

    def get_output_data(self):
        return {}

class ResultsNode(BaseNode):
    def __init__(self, node_id, simulator_app, node_editor_app):
        super().__init__(node_id, simulator_app, node_editor_app)
        self.node_type = NodeType.RESULTS
        self.metrics = ["Success_Rate", "Avg_Reward", "Avg_Steps", "Eval_Time"]
        self.tags = {m: f"{node_id}_val_{m}" for m in self.metrics}
        self.results_table_tag = f"{node_id}_results_table"

    def create_attributes(self):
        input_attrs, output_attrs, static_attrs, all_attrs = {}, {}, {}, []
        input_types, output_types = {}, {}

        # Generic Agent Input
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, label="Agent") as input_agent_attr:
            dpg.add_text("Agent (RL or IL)")
        input_attrs['agent'] = input_agent_attr
        input_types[input_agent_attr] = 'Agent'
        all_attrs.append(input_agent_attr)
        
        # Environment Input
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, label="Environment") as input_env_attr:
            dpg.add_text("Environment")
        input_attrs['env_instance'] = input_env_attr
        input_types[input_env_attr] = 'Environment'
        all_attrs.append(input_env_attr)

        # Action Button
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static) as button_attr:
            dpg.add_button(label="Evaluate Agent", width=150, callback=self.evaluate_callback)
        static_attrs['button'] = button_attr
        all_attrs.append(button_attr)

        # Results Table
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static) as table_attr:
            dpg.add_text("Evaluation Results:", color=(0, 255, 255))
            with dpg.table(header_row=True, tag=self.results_table_tag, borders_innerH=True, borders_outerH=True, borders_innerV=True, borders_outerV=True, width=220):
                dpg.add_table_column(label="Metric")
                dpg.add_table_column(label="Value")
                
                for metric in self.metrics:
                    with dpg.table_row():
                        dpg.add_text(metric.replace('_', ' '))
                        dpg.add_text("-", tag=self.tags[metric])
        static_attrs['table'] = table_attr
        all_attrs.append(table_attr)
        
        return input_attrs, output_attrs, static_attrs, all_attrs, input_types, output_types

    def evaluate_callback(self):
        env_node_id, _ = self.editor.find_connected_node_and_data(self.node_id, 'env_instance')
        agent_node_id, _ = self.editor.find_connected_node_and_data(self.node_id, 'agent')

        if not env_node_id:
            print(f"[{self.node_id}] Error: Connect Environment first!")
            return
            
        if not agent_node_id:
            print(f"[{self.node_id}] Error: Connect an Agent first!")
            return

        # Determine agent type based on where it is stored in simulator
        agent_type = "RL" if agent_node_id in self.simulator.rl_agents else "IL"
        
        results, _ = self.simulator.evaluate(env_node_id, agent_node_id, agent_type)
        self.update_results_display(results)

    def update_results_display(self, results):
        if results:
            dpg.set_value(self.tags["Success_Rate"], f"{results['success_rate']:.1f}%")
            dpg.set_value(self.tags["Avg_Reward"], f"{results['avg_reward']:.2f}")
            dpg.set_value(self.tags["Avg_Steps"], f"{results['avg_steps']:.1f}")
            dpg.set_value(self.tags["Eval_Time"], f"{results['eval_time']:.2f}s")
        else:
            for tag in self.tags.values():
                dpg.set_value(tag, "-")

    def get_output_data(self):
        return {}