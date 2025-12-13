import dearpygui.dearpygui as dpg
import numpy as np
from enum import Enum
from settings import WINDOW_SIZE
from renderer import grid_to_image, render_trajectory
from environment import TerrainEnv, LabyrinthEnv
from pathfinder_agent import PathFinderAgent

class NodeType(Enum):
    ENVIRONMENT = "Environment"
    LABYRINTH = "Labyrinth"
    RL_AGENT = "RL_Agent"
    IL_AGENT = "IL_Agent"
    PATHFINDER_AGENT = "PathFinder_Agent"
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
                dpg.add_button(label="Labyrinth Node", callback=self.add_labyrinth_node)
                dpg.add_button(label="RL Agent Node", callback=self.add_rl_agent_node)
                dpg.add_button(label="IL Agent Node", callback=self.add_il_agent_node)
                dpg.add_button(label="PathFinder Node", callback=self.add_pathfinder_node)
                dpg.add_button(label="Visualizer Node", callback=self.add_visualizer_node)
                dpg.add_button(label="Settings Node", callback=self.add_settings_node)
                dpg.add_button(label="Results Node", callback=self.add_results_node)
                dpg.add_button(label="Clear All Nodes", callback=self.clear_all_nodes)
            
            dpg.add_separator()
            
            with dpg.node_editor(tag=self.node_editor_tag, 
                                callback=self.link_callback,
                                delink_callback=self.delink_callback,
                                minimap=True,
                                height=900):
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
        if not target_node_data:
            return None, None
        
        target_input_attr = target_node_data['input_attrs'].get(input_attribute_name)
        if not target_input_attr:
            return None, None
        
        for link_id, (output_attr_id, input_attr_id) in self.links.items():
            if input_attr_id == target_input_attr:
                output_node_id = self.get_node_from_attribute_id(output_attr_id)
                if output_node_id and output_node_id in self.node_objects:
                    return output_node_id, self.node_objects[output_node_id]
        return None, None

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

    def add_labyrinth_node(self):
        self.add_node(LabyrinthNode, "Labyrinth Environment", [100, 150])

    def add_rl_agent_node(self):
        self.add_node(RLAgentNode, "RL Agent (Q-Learner)", [300, 50])

    def add_il_agent_node(self):
        self.add_node(ILAgentNode, "IL Agent (Imitation)", [300, 250])

    def add_pathfinder_node(self):
        self.add_node(PathFinderNode, "PathFinder Agent", [550, 50])

    def add_visualizer_node(self):
        self.add_node(VisualizerNode, "Visualization", [600, 50])

    def add_settings_node(self):
        self.add_node(SettingsNode, "Settings", [50, 400])

    def add_results_node(self):
        self.add_node(ResultsNode, "Results Table", [850, 50])

    def delete_node(self, node_id):
        if node_id not in self.nodes:
            print(f"Node {node_id} not found")
            return
        
        try:
            links_to_delete = []
            for link_id, (output_attr_id, input_attr_id) in list(self.links.items()):
                output_node_id = self.get_node_from_attribute_id(output_attr_id)
                input_node_id = self.get_node_from_attribute_id(input_attr_id)
                
                if output_node_id == node_id or input_node_id == node_id:
                    links_to_delete.append(link_id)
            
            for link_id in links_to_delete:
                if link_id in self.links:
                    del self.links[link_id]
                try:
                    dpg.delete_item(link_id)
                except:
                    pass
            
            node_type = self.nodes[node_id]['type']
            
            if node_type == NodeType.ENVIRONMENT:
                if node_id in self.simulator.environments:
                    del self.simulator.environments[node_id]
            elif node_type == NodeType.RL_AGENT:
                if node_id in self.simulator.rl_agents:
                    del self.simulator.rl_agents[node_id]
            elif node_type == NodeType.IL_AGENT:
                if node_id in self.simulator.il_agents:
                    del self.simulator.il_agents[node_id]
            elif node_type == NodeType.PATHFINDER_AGENT:
                if node_id in self.simulator.pathfinder_agents:
                    del self.simulator.pathfinder_agents[node_id]
            
            for attr in self.nodes[node_id]['all_attrs']:
                if attr in self.attribute_to_node:
                    del self.attribute_to_node[attr]
            
            if node_id in self.node_objects:
                del self.node_objects[node_id]
            
            del self.nodes[node_id]
            dpg.delete_item(node_id)
            
            print(f"Successfully deleted node: {node_id}")
            
        except Exception as e:
            print(f"Error deleting node {node_id}: {e}")

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
        self.simulator.pathfinder_agents = {}
        self.node_counter = 0

class BaseNode:
    def __init__(self, node_id, simulator_app, node_editor_app):
        self.node_id = node_id
        self.simulator = simulator_app
        self.editor = node_editor_app
        self.node_type = None
        self.status = "Ready for use"
        self.status_tag = None

    def update_status(self, new_status, color=None):
        self.status = new_status
        if self.status_tag and dpg.does_item_exist(self.status_tag):
            dpg.set_value(self.status_tag, f"Status: {new_status}")
            if color:
                dpg.configure_item(self.status_tag, color=color)
            elif "Error" in new_status:
                dpg.configure_item(self.status_tag, color=(255, 0, 0))
            elif "Generating" in new_status or "Training" in new_status:
                dpg.configure_item(self.status_tag, color=(255, 255, 0))
            elif "Ready" in new_status or "Generated" in new_status or "Trained" in new_status:
                dpg.configure_item(self.status_tag, color=(0, 255, 0))
            else:
                dpg.configure_item(self.status_tag, color=(255, 255, 255))

    def create_attributes(self):
        raise NotImplementedError

    def create_delete_button(self):
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static) as delete_attr:
            dpg.add_button(label="X Delete", width=100, callback=lambda: self.editor.delete_node(self.node_id))
        return delete_attr

    def get_output_data(self):
        raise NotImplementedError

    def get_input_data(self, input_attribute_name, output_key=None):
        output_node_id, output_node_obj = self.editor.find_connected_node_and_data(self.node_id, input_attribute_name)
        
        if not output_node_obj:
            return None
        
        output_data = output_node_obj.get_output_data()
        
        if not output_data:
            return None
        
        if output_key:
            result = output_data.get(output_key)
            return result
        
        return output_data

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

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static) as static_attr:
            self.status_tag = f"{self.node_id}_status"
            dpg.add_text(f"Status: {self.status}", tag=self.status_tag)
        static_attrs['status'] = static_attr
        all_attrs.append(static_attr)

        delete_attr = self.create_delete_button()
        static_attrs['delete'] = delete_attr
        all_attrs.append(delete_attr)

        return input_attrs, output_attrs, static_attrs, all_attrs, input_types, output_types

    def generate_terrain_callback(self):
        self.update_status("Generating...")
        try:
            config_data = self.get_input_data('config')
            
            if config_data and isinstance(config_data, dict):
                grid_size = config_data.get('grid_size') 
                start_pos = config_data.get('start_position')
                goal_pos = config_data.get('goal_position')
                
                if grid_size:
                    self.env_instance.set_size(grid_size)
                if start_pos and goal_pos:
                    self.env_instance.set_pos(start=start_pos, goal=goal_pos)
            
            self.simulator.generate_terrain(self.node_id, grid_size=config_data.get('grid_size') if config_data else None)
            
            self.update_status("Generated")
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_output_data(self):
        return {
            self.output_grid_attr: self.env_instance.get_grid(),
            self.output_env_instance_attr: self.env_instance
        }

class LabyrinthNode(BaseNode):
    def __init__(self, node_id, simulator_app, node_editor_app):
        super().__init__(node_id, simulator_app, node_editor_app)
        self.node_type = NodeType.LABYRINTH
        self.env_instance = LabyrinthEnv(size=self.simulator.settings.GRID_SIZE)
        self.env_instance.generate_random()
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
            dpg.add_button(label="Generate Labyrinth", width=150, callback=self.generate_labyrinth_callback)
            dpg.add_slider_float(label="Wall Density", default_value=0.2, min_value=0.1, max_value=0.4,
                                width=150, callback=self.update_wall_density, tag=f"{self.node_id}_density")
        static_attrs['button'] = button_attr
        all_attrs.append(button_attr)

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, label="Grid Data") as output_grid_attr:
            dpg.add_text("Grid Data", indent=80)
        output_attrs['output_grid'] = output_grid_attr
        output_types[output_grid_attr] = 'Grid Data'
        all_attrs.append(output_grid_attr)
        self.output_grid_attr = output_grid_attr
            
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, label="Env Instance") as output_env_instance_attr:
            dpg.add_text("Labyrinth Instance", indent=80)
        output_attrs['output_env_instance'] = output_env_instance_attr
        output_types[output_env_instance_attr] = 'Environment'
        all_attrs.append(output_env_instance_attr)
        self.output_env_instance_attr = output_env_instance_attr

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static) as static_attr:
            self.status_tag = f"{self.node_id}_status"
            dpg.add_text(f"Status: {self.status}", tag=self.status_tag)
        static_attrs['status'] = static_attr
        all_attrs.append(static_attr)

        delete_attr = self.create_delete_button()
        static_attrs['delete'] = delete_attr
        all_attrs.append(delete_attr)

        return input_attrs, output_attrs, static_attrs, all_attrs, input_types, output_types

    def generate_labyrinth_callback(self):
        self.update_status("Generating labyrinth...")
        try:
            config_data = self.get_input_data('config')
            
            if config_data and isinstance(config_data, dict):
                grid_size = config_data.get('grid_size')
                start_pos = config_data.get('start_position')
                goal_pos = config_data.get('goal_position')
                
                if grid_size:
                    self.env_instance.set_size(grid_size)
                    self.env_instance.generate_random()
                if start_pos and goal_pos:
                    self.env_instance.set_pos(start=start_pos, goal=goal_pos)
            
            self.env_instance.generate_random()
            
            self.update_status("Generated")
        except Exception as e:
            self.update_status(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_wall_density(self, sender, app_data):
        self.env_instance.wall_density = app_data

    def get_output_data(self):
        return {
            self.output_grid_attr: self.env_instance.get_grid(),
            self.output_env_instance_attr: self.env_instance
        }

class PathFinderNode(BaseNode):
    def __init__(self, node_id, simulator_app, node_editor_app):
        super().__init__(node_id, simulator_app, node_editor_app)
        self.node_type = NodeType.PATHFINDER_AGENT
        self.pathfinder_instance = PathFinderAgent(algorithm='astar')
        self.simulator.pathfinder_agents[self.node_id] = self.pathfinder_instance
        self.output_agent_attr = None
        self.output_path_attr = None
        self.algorithm_var = 'astar'
        self.found_path = None
        
    def create_attributes(self):
        input_attrs, output_attrs, static_attrs, all_attrs = {}, {}, {}, []
        input_types, output_types = {}, {}

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, label="Environment") as input_env_attr:
            dpg.add_text("Environment")
        input_attrs['env_instance'] = input_env_attr
        input_types[input_env_attr] = 'Environment'
        all_attrs.append(input_env_attr)
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, label="Grid Data") as input_grid_attr:
            dpg.add_text("Grid Data")
        input_attrs['grid_data'] = input_grid_attr
        input_types[input_grid_attr] = 'Grid Data'
        all_attrs.append(input_grid_attr)

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static) as button_attr:
            dpg.add_text("PathFinder Settings:")
            
            dpg.add_radio_button(items=["A*", "BFS", "DFS"], default_value="A*",
                               callback=self.update_algorithm, tag=f"{self.node_id}_algorithm")
            
            dpg.add_button(label="Find Path", width=150, callback=self.find_path_callback)
            dpg.add_button(label="Reset Path", width=150, callback=self.reset_path_callback)
        static_attrs['button'] = button_attr
        all_attrs.append(button_attr)

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, label="PathFinder Agent") as output_agent_attr:
            dpg.add_text("PathFinder Agent", indent=60)
        output_attrs['output_agent'] = output_agent_attr
        output_types[output_agent_attr] = 'PathFinder Agent'
        all_attrs.append(output_agent_attr)
        self.output_agent_attr = output_agent_attr

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static) as static_attr:
            self.status_tag = f"{self.node_id}_status"
            dpg.add_text(f"Status: {self.status}", tag=self.status_tag)
            
            self.path_info_tag = f"{self.node_id}_path_info"
            dpg.add_text("No path found yet", tag=self.path_info_tag)
        static_attrs['status'] = static_attr
        all_attrs.append(static_attr)

        delete_attr = self.create_delete_button()
        static_attrs['delete'] = delete_attr
        all_attrs.append(delete_attr)

        return input_attrs, output_attrs, static_attrs, all_attrs, input_types, output_types

    def update_algorithm(self, sender, app_data):
        algorithm_map = {"A*": "astar", "BFS": "bfs", "DFS": "dfs"}
        self.algorithm_var = algorithm_map.get(app_data, "astar")
        self.pathfinder_instance.algorithm = self.algorithm_var
    
    def find_path_callback(self):
        self.update_status("Finding path...", color=(255, 255, 0))
        
        try:
            env_node_id, _ = self.editor.find_connected_node_and_data(self.node_id, 'env_instance')
            
            if not env_node_id:
                self.update_status("Error: Connect Environment first!", color=(255, 0, 0))
                return
            
            env = self.simulator.get_environment(env_node_id)
            if not env:
                self.update_status("Error: Environment not found!", color=(255, 0, 0))
                return
            
            grid_data = self.get_input_data('grid_data')
            if grid_data is None:
                grid = env.get_grid()
            else:
                grid = grid_data
            
            env.trajectory = []
            env.evaluation_trajectory = []
            
            self.found_path = self.pathfinder_instance.find_path(grid, env.start, env.goal)
            
            if self.found_path:
                env.trajectory = self.found_path
                env.evaluation_trajectory = self.found_path
                
                self.update_status(f"Path found! Length: {len(self.found_path)}", color=(0, 255, 0))
                dpg.set_value(self.path_info_tag, 
                            f"Algorithm: {self.algorithm_var.upper()}, "
                            f"Path length: {len(self.found_path)}, "
                            f"Time: {self.pathfinder_instance.stats['search_time']:.3f}s")
            else:
                self.update_status("No path found!", color=(255, 0, 0))
                dpg.set_value(self.path_info_tag, "No path found")
                
        except Exception as e:
            error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
            self.update_status(f"Error: {error_msg}", color=(255, 0, 0))
    
    def reset_path_callback(self):
        self.pathfinder_instance.reset()
        self.found_path = None
        
        env_node_id, _ = self.editor.find_connected_node_and_data(self.node_id, 'env_instance')
        if env_node_id:
            env = self.simulator.get_environment(env_node_id)
            if env:
                env.trajectory = []
                env.evaluation_trajectory = []
        
        self.update_status("Path reset", color=(0, 255, 0))
        dpg.set_value(self.path_info_tag, "Path reset")

    def get_output_data(self):
        return {
            self.output_agent_attr: self.pathfinder_instance,
            self.output_path_attr: self.found_path
        }

class RLAgentNode(BaseNode):
    def __init__(self, node_id, simulator_app, node_editor_app):
        super().__init__(node_id, simulator_app, node_editor_app)
        self.node_type = NodeType.RL_AGENT
        self.rl_agent_instance = None
        self.output_agent_attr = None
        self.training_terrains_input = None
        
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
            dpg.add_text("Training Settings:")
            self.training_terrains_input = dpg.add_input_int(label="Training Terrains", default_value=200,
                                                           min_value=1, max_value=100, width=150)
            
            dpg.add_button(label="Train RL Agent on Multiple Terrains", width=200,
                         callback=self.train_rl_callback)
        static_attrs['button'] = button_attr
        all_attrs.append(button_attr)

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, label="Trained Agent") as output_agent_attr:
            dpg.add_text("General RL Agent", indent=60)
        output_attrs['output_agent'] = output_agent_attr
        output_types[output_agent_attr] = 'RL Agent'
        all_attrs.append(output_agent_attr)
        self.output_agent_attr = output_agent_attr

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static) as static_attr:
            self.status_tag = f"{self.node_id}_status"
            dpg.add_text(f"Status: {self.status}", tag=self.status_tag)
            
            self.training_info_tag = f"{self.node_id}_training_info"
            dpg.add_text("Ready for training", tag=self.training_info_tag)
        static_attrs['status'] = static_attr
        all_attrs.append(static_attr)

        delete_attr = self.create_delete_button()
        static_attrs['delete'] = delete_attr
        all_attrs.append(delete_attr)

        return input_attrs, output_attrs, static_attrs, all_attrs, input_types, output_types

    def train_rl_callback(self):
        self.update_status("Starting training...", color=(255, 255, 0))
        
        try:
            env_node_id, _ = self.editor.find_connected_node_and_data(self.node_id, 'env_instance')
            config = self.get_input_data('config', 'all')
            
            if not env_node_id:
                self.update_status("Error: Connect Environment first!", color=(255, 0, 0))
                return

            training_terrains = 200
            if self.training_terrains_input and dpg.does_item_exist(self.training_terrains_input):
                try:
                    training_terrains = dpg.get_value(self.training_terrains_input)
                except:
                    pass
            
            if not config or not isinstance(config, dict):
                config = None
            
            self.update_status(f"Training on {training_terrains} terrains...", color=(255, 255, 0))
            dpg.set_value(self.training_info_tag, f"Training on {training_terrains} terrains...")
            
            agent = self.simulator.train_rl_with_terrains(env_node_id, self.node_id, 
                                                        num_terrains=training_terrains,
                                                        settings_config=config)
            
            if agent:
                self.rl_agent_instance = agent
                
                if hasattr(agent, 'get_statistics'):
                    stats = agent.get_statistics()
                    unique_states = stats.get('unique_states_seen', 'N/A')
                    avg_reward = stats.get('avg_reward', 0)
                    
                    self.update_status(f"Trained successfully!", color=(0, 255, 0))
                    dpg.set_value(self.training_info_tag, 
                                f"States learned: {unique_states}, Avg reward: {avg_reward:.2f}")
                else:
                    self.update_status(f"Trained on {training_terrains} terrains", color=(0, 255, 0))
                    dpg.set_value(self.training_info_tag, f"Trained on {training_terrains} terrains")
            else:
                self.update_status("Training failed!", color=(255, 0, 0))
                dpg.set_value(self.training_info_tag, "Training failed")
                
        except Exception as e:
            error_msg = str(e)[:50] + "..." if len(str(e)) > 50 else str(e)
            self.update_status(f"Error: {error_msg}", color=(255, 0, 0))
            import traceback
            traceback.print_exc()

    def get_output_data(self):
        return {self.output_agent_attr: self.rl_agent_instance}

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

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static) as static_attr:
            self.status_tag = f"{self.node_id}_status"
            dpg.add_text(f"Status: {self.status}", tag=self.status_tag)
        static_attrs['status'] = static_attr
        all_attrs.append(static_attr)

        delete_attr = self.create_delete_button()
        static_attrs['delete'] = delete_attr
        all_attrs.append(delete_attr)

        return input_attrs, output_attrs, static_attrs, all_attrs, input_types, output_types

    def train_il_callback(self):
        self.update_status("Initializing...")
        
        env_node_id, _ = self.editor.find_connected_node_and_data(self.node_id, 'env_instance')
        teacher_node_id, _ = self.editor.find_connected_node_and_data(self.node_id, 'teacher_agent')
        config = self.get_input_data('config', 'all')
        
        if not env_node_id:
            self.update_status("Error: Connect Environment first!")
            return
            
        if not teacher_node_id:
            self.update_status("Error: Connect Teacher Agent first!")
            return
        
        teacher_agent = self.simulator.rl_agents.get(teacher_node_id)
        if not teacher_agent:
            self.update_status("Error: Teacher agent not trained!")
            return

        try:
            self.update_status("Training IL Agent...")
            agent = self.simulator.train_il(env_node_id, self.node_id, teacher_node_id, settings_config=config)
            
            if agent:
                self.il_agent_instance = agent
                self.update_status("Trained")
            else:
                self.update_status("Training failed")
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            if len(error_msg) > 50:
                error_msg = error_msg[:50] + "..."
            self.update_status(error_msg)
            import traceback
            traceback.print_exc()

    def get_output_data(self):
        return {self.output_agent_attr: self.il_agent_instance}

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
            'start_position': self.simulator.settings.START_POSITION,
            'goal_position': self.simulator.settings.GOAL_POSITION
        }
        
    def create_attributes(self):
        input_attrs, output_attrs, static_attrs, all_attrs = {}, {}, {}, []
        input_types, output_types = {}, {}

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static) as static_attr_settings:
            dpg.add_text("Simulation Settings", color=(255, 255, 0))
            
            self.inputs['grid_size'] = dpg.add_input_int(label="Grid Size", 
                default_value=self.default_settings['grid_size'], min_value=5, max_value=50, width=150)
            
            self.inputs['max_steps'] = dpg.add_input_int(label="Max Steps", 
                default_value=self.default_settings['max_steps'], width=150)
            self.inputs['rl_episodes'] = dpg.add_input_int(label="RL Episodes", 
                default_value=self.default_settings['rl_episodes'], width=150)
            self.inputs['il_episodes'] = dpg.add_input_int(label="IL Episodes", 
                default_value=self.default_settings['il_episodes'], width=150)
            self.inputs['timeout'] = dpg.add_input_int(label="Timeout (s)", 
                default_value=self.default_settings['timeout'], width=150)
            
            dpg.add_text("Start Position:")
            with dpg.group(horizontal=True):
                self.inputs['start_x'] = dpg.add_input_int(label="X", 
                    default_value=self.default_settings['start_position'][0], min_value=0, max_value=49, width=70)
                self.inputs['start_y'] = dpg.add_input_int(label="Y", 
                    default_value=self.default_settings['start_position'][1], min_value=0, max_value=49, width=70)
            
            dpg.add_text("Goal Position:")
            with dpg.group(horizontal=True):
                self.inputs['goal_x'] = dpg.add_input_int(label="X", 
                    default_value=self.default_settings['goal_position'][0], min_value=0, max_value=49, width=70)
                self.inputs['goal_y'] = dpg.add_input_int(label="Y", 
                    default_value=self.default_settings['goal_position'][1], min_value=0, max_value=49, width=70)
                
        static_attrs['settings'] = static_attr_settings
        all_attrs.append(static_attr_settings)

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output, label="Config Output") as output_config_attr:
            dpg.add_text("Config Data", indent=80)
        output_attrs['output_config'] = output_config_attr
        output_types[output_config_attr] = 'Settings'
        all_attrs.append(output_config_attr)
        self.output_config_attr = output_config_attr

        delete_attr = self.create_delete_button()
        static_attrs['delete'] = delete_attr
        all_attrs.append(delete_attr)

        return input_attrs, output_attrs, static_attrs, all_attrs, input_types, output_types

    def get_output_data(self):
        config_data = {}
        
        simple_keys = ['grid_size', 'max_steps', 'rl_episodes', 'il_episodes', 'timeout']
        for key in simple_keys:
            if key in self.inputs:
                try:
                    config_data[key] = dpg.get_value(self.inputs[key])
                except:
                    config_data[key] = self.default_settings.get(key)
        
        try:
            start_x = dpg.get_value(self.inputs['start_x'])
            start_y = dpg.get_value(self.inputs['start_y'])
            config_data['start_position'] = (start_x, start_y)
        except:
            config_data['start_position'] = self.default_settings['start_position']
        
        try:
            goal_x = dpg.get_value(self.inputs['goal_x'])
            goal_y = dpg.get_value(self.inputs['goal_y'])
            config_data['goal_position'] = (goal_x, goal_y)
        except:
            config_data['goal_position'] = self.default_settings['goal_position']
        
        return { 
            self.output_config_attr: config_data, 
            'all': config_data,
            'grid_size': config_data.get('grid_size'),
            'max_steps': config_data.get('max_steps'),
            'rl_episodes': config_data.get('rl_episodes'),
            'il_episodes': config_data.get('il_episodes'),
            'timeout': config_data.get('timeout'),
            'start_position': config_data.get('start_position'),
            'goal_position': config_data.get('goal_position')
        }

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
            dpg.add_text("Agent (RL, IL or PathFinder)")
        input_attrs['agent_instance'] = input_agent_attr
        input_types[input_agent_attr] = 'Agent'
        all_attrs.append(input_agent_attr)

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static) as button_attr:
            dpg.add_button(label="Visualize Terrain", width=150, callback=self.visualize_terrain_callback)
            dpg.add_button(label="Evaluate Agent", width=150, callback=self.evaluate_agent_callback)
        static_attrs['button'] = button_attr
        all_attrs.append(button_attr)
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static) as vis_output_attr:
            with dpg.texture_registry(show=False):
                dpg.add_raw_texture(width=self.window_size, height=self.window_size,
                    default_value=np.ones((self.window_size * self.window_size * 3)) * 0.8,
                    format=dpg.mvFormat_Float_rgb, tag=self.texture_tag_terrain)
            dpg.add_text("Visualization Output", color=(0, 255, 0))
            dpg.add_image(self.texture_tag_terrain, tag=self.image_tag)
            
            self.status_tag = f"{self.node_id}_status"
            dpg.add_text(f"Status: {self.status}", tag=self.status_tag)
            
            self.info_tag = f"{self.node_id}_info"
            dpg.add_text("Ready for visualization", tag=self.info_tag)
        static_attrs['vis_output'] = vis_output_attr
        all_attrs.append(vis_output_attr)

        delete_attr = self.create_delete_button()
        static_attrs['delete'] = delete_attr
        all_attrs.append(delete_attr)

        return input_attrs, output_attrs, static_attrs, all_attrs, input_types, output_types

    def visualize_terrain_callback(self):
        self.update_status("Loading terrain...", color=(255, 255, 0))
        
        env_node_id, env_node_obj = self.editor.find_connected_node_and_data(self.node_id, 'env_instance')
        
        if not env_node_id:
            self.update_status("Error: Connect Environment first!", color=(255, 0, 0))
            return
            
        env_instance = self.simulator.get_environment(env_node_id)
        agent_node_id, _ = self.editor.find_connected_node_and_data(self.node_id, 'agent_instance')
        trajectory = None
        agent_type = "Base"

        if agent_node_id:
            if agent_node_id in self.simulator.rl_agents:
                agent_type = "RL"
                trajectory = getattr(env_instance, 'trajectory', [])
            elif agent_node_id in self.simulator.il_agents:
                agent_type = "IL"
                trajectory = getattr(env_instance, 'trajectory', [])
            elif agent_node_id in self.simulator.pathfinder_agents:
                agent_type = "PathFinder"
                pathfinder_node = self.editor.node_objects.get(agent_node_id)
                if pathfinder_node and hasattr(pathfinder_node, 'found_path'):
                    trajectory = pathfinder_node.found_path
                else:
                    trajectory = getattr(env_instance, 'trajectory', [])

        grid = env_instance.get_grid()
        scale = self.window_size // grid.shape[0]
        
        if trajectory:
            if agent_type == "PathFinder":
                image_data = self.render_pathfinder_path(grid, trajectory, scale)
                status_text = f"PathFinder Path: {len(trajectory)} steps"
            else:
                image_data = render_trajectory(grid, trajectory, scale=scale, 
                                              target_size=self.window_size, line_width=3)
                status_text = f"{agent_type} Path: {len(trajectory)} steps"
            self.update_status(f"Showing {agent_type} path", color=(0, 255, 0))
        else:
            image_data = grid_to_image(grid, scale=scale, target_size=self.window_size)
            status_text = f"Terrain: {env_instance.size}x{env_instance.size}"
            self.update_status("Showing terrain only", color=(0, 255, 0))
        
        try:
            dpg.set_value(self.texture_tag_terrain, image_data.flatten())
            dpg.set_value(self.info_tag, status_text)
        except Exception as e:
            self.update_status(f"Error rendering: {str(e)[:30]}...", color=(255, 0, 0))

    def render_pathfinder_path(self, grid, path, scale):
        base_image = grid_to_image(grid, scale, self.window_size)
        
        if not path or len(path) < 2:
            return base_image
        
        img_uint8 = (base_image * 255).astype(np.uint8)
        h, w = grid.shape
        scaled_h = h * scale
        scaled_w = w * scale
        pad_h = (self.window_size - scaled_h) // 2
        pad_w = (self.window_size - scaled_w) // 2
        
        for k in range(1, len(path)):
            prev_pos = path[k-1]
            curr_pos = path[k]
            
            prev_x = pad_w + (prev_pos[1] + 0.5) * scale
            prev_y = pad_h + (prev_pos[0] + 0.5) * scale
            curr_x = pad_w + (curr_pos[1] + 0.5) * scale
            curr_y = pad_h + (curr_pos[0] + 0.5) * scale
            
            num_points = max(abs(int(curr_x - prev_x)), abs(int(curr_y - prev_y))) + 1
            for t in np.linspace(0, 1, num_points):
                x = int(prev_x * (1-t) + curr_x * t)
                y = int(prev_y * (1-t) + curr_y * t)
                
                line_width = 3
                for dx in range(-line_width//2, line_width//2 + 1):
                    for dy in range(-line_width//2, line_width//2 + 1):
                        px = x + dx
                        py = y + dy
                        if 0 <= px < self.window_size and 0 <= py < self.window_size:
                            img_uint8[py, px] = [0, 255, 255]
        
        return img_uint8.astype(np.float32) / 255.0

    def evaluate_agent_callback(self):
        self.update_status("Evaluating agent...", color=(255, 255, 0))
    
        env_node_id, _ = self.editor.find_connected_node_and_data(self.node_id, 'env_instance')
        agent_node_id, _ = self.editor.find_connected_node_and_data(self.node_id, 'agent_instance')

        if not env_node_id or not agent_node_id:
            self.update_status("Error: Connect Environment AND Agent first!", color=(255, 0, 0))
            return
            
        agent_type = "RL"
        if agent_node_id in self.simulator.rl_agents:
            agent_type = "RL"
        elif agent_node_id in self.simulator.il_agents:
            agent_type = "IL"
        elif agent_node_id in self.simulator.pathfinder_agents:
            agent_type = "PathFinder"
        
        env = self.simulator.get_environment(env_node_id)
        if env:
            env.reset()
            env.evaluation_trajectory = []
        
        if agent_type == "PathFinder":
            results = self.evaluate_pathfinder(env_node_id, agent_node_id)
            trajectory = getattr(env, 'evaluation_trajectory', getattr(env, 'trajectory', []))
        else:
            results, trajectory = self.simulator.evaluate(env_node_id, agent_node_id, agent_type)
        
        
        self.visualize_terrain_callback()

    
    def evaluate_pathfinder(self, env_id, agent_id):
        env = self.simulator.get_environment(env_id)
        agent = self.simulator.pathfinder_agents.get(agent_id)
        
        if not env or not agent:
            return {'success': False, 'error': 'Environment or agent not found'}
        
        try:
            grid = env.get_grid()
            path = agent.find_path(grid, env.start, env.goal)
            
            if path:
                env.trajectory = path
                env.evaluation_trajectory = path
                
                return {
                    'success': True,
                    'path_length': len(path),
                    'search_time': agent.stats['search_time']
                }
            else:
                return {'success': False, 'error': 'No path found'}
        except Exception as e:
            return {'success': False, 'error': str(e)[:50]}

    def update_status(self, new_status, color=None):
        self.status = new_status
        if self.status_tag and dpg.does_item_exist(self.status_tag):
            try:
                dpg.set_value(self.status_tag, f"Status: {new_status}")
                if color:
                    dpg.configure_item(self.status_tag, color=color)
                elif "Error" in new_status:
                    dpg.configure_item(self.status_tag, color=(255, 0, 0))
                elif "Generating" in new_status or "Training" in new_status or "Loading" in new_status or "Evaluating" in new_status:
                    dpg.configure_item(self.status_tag, color=(255, 255, 0))
                elif "Ready" in new_status or "Showing" in new_status or "Complete" in new_status:
                    dpg.configure_item(self.status_tag, color=(0, 255, 0))
                else:
                    dpg.configure_item(self.status_tag, color=(255, 255, 255))
            except Exception as e:
                print(f"Error updating status in VisualizerNode: {e}")

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

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, label="Agent") as input_agent_attr:
            dpg.add_text("Agent (RL, IL or PathFinder)")
        input_attrs['agent'] = input_agent_attr
        input_types[input_agent_attr] = 'Agent'
        all_attrs.append(input_agent_attr)
        
        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input, label="Environment") as input_env_attr:
            dpg.add_text("Environment")
        input_attrs['env_instance'] = input_env_attr
        input_types[input_env_attr] = 'Environment'
        all_attrs.append(input_env_attr)

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static) as button_attr:
            dpg.add_button(label="Evaluate Agent", width=150, callback=self.evaluate_callback)
        static_attrs['button'] = button_attr
        all_attrs.append(button_attr)

        with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static) as table_attr:
            dpg.add_text("Evaluation Results:", color=(0, 255, 255))
            with dpg.table(header_row=True, tag=self.results_table_tag, borders_innerH=True,
                          borders_outerH=True, borders_innerV=True, borders_outerV=True, width=220):
                dpg.add_table_column(label="Metric")
                dpg.add_table_column(label="Value")
                
                for metric in self.metrics:
                    with dpg.table_row():
                        dpg.add_text(metric.replace('_', ' '))
                        dpg.add_text("-", tag=self.tags[metric])
        static_attrs['table'] = table_attr
        all_attrs.append(table_attr)

        delete_attr = self.create_delete_button()
        static_attrs['delete'] = delete_attr
        all_attrs.append(delete_attr)
        
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

        # Determine agent type
        if agent_node_id in self.simulator.rl_agents:
            agent_type = "RL"
            results, _ = self.simulator.evaluate(env_node_id, agent_node_id, agent_type)
        elif agent_node_id in self.simulator.il_agents:
            agent_type = "IL"
            results, _ = self.simulator.evaluate(env_node_id, agent_node_id, agent_type)
        elif agent_node_id in self.simulator.pathfinder_agents:
            agent_type = "PathFinder"
            results = self.evaluate_pathfinder(env_node_id, agent_node_id)
        else:
            print(f"[{self.node_id}] Error: Unknown agent type!")
            return
            
        self.update_results_display(results)

    def evaluate_pathfinder(self, env_id, agent_id):
        env = self.simulator.get_environment(env_id)
        agent = self.simulator.pathfinder_agents.get(agent_id)
        
        if not env or not agent:
            return {'success_rate': 0.0, 'avg_reward': 0.0, 'avg_steps': 0.0, 'eval_time': 0.0}
        
        try:
            import time
            start_time = time.time()
            
            grid = env.get_grid()
            path = agent.find_path(grid, env.start, env.goal)
            
            eval_time = time.time() - start_time
            
            if path:
                # Calculate metrics similar to other agents
                success_rate = 100.0
                avg_reward = 1.0  # PathFinder always gets maximum reward if path found
                avg_steps = len(path)
                
                return {
                    'success_rate': success_rate,
                    'avg_reward': avg_reward,
                    'avg_steps': avg_steps,
                    'eval_time': eval_time
                }
            else:
                return {
                    'success_rate': 0.0,
                    'avg_reward': 0.0,
                    'avg_steps': 0.0,
                    'eval_time': eval_time
                }
        except Exception as e:
            print(f"Error evaluating PathFinder: {e}")
            return {
                'success_rate': 0.0,
                'avg_reward': 0.0,
                'avg_steps': 0.0,
                'eval_time': 0.0
            }

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