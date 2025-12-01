import dearpygui.dearpygui as dpg
import numpy as np
from enum import Enum

class NodeType(Enum):
    ENVIRONMENT = "Environment"
    RL_AGENT = "RL_Agent"
    IL_AGENT = "IL_Agent"
    TRAINER = "Trainer"
    EVALUATOR = "Evaluator"
    VISUALIZER = "Visualizer"

class NodeEditorApp:
    def __init__(self, simulator_app):
        self.simulator = simulator_app
        self.nodes = {}
        
        # Store links as {link_id: (output_attr_id, input_attr_id)}
        self.links = {}
        
        # Node editor window tag
        self.node_editor_window_tag = "NodeEditorWindow"
        self.node_editor_tag = "NodeEditor"
        
        # Node IDs counter
        self.node_counter = 0
        
        # Store attribute mappings separately
        self.attribute_to_node = {}
        
    def create_node_editor_window(self):
        # If window already exists
        if dpg.does_item_exist(self.node_editor_window_tag):
            dpg.focus_item(self.node_editor_window_tag)
            return

        with dpg.window(label="Node Editor", tag=self.node_editor_window_tag,
                        pos=(410, 620), width=800, height=600):
            
            dpg.add_text("How to use: Drag from output (Right) to input (Left) to connect nodes")
            dpg.add_text("Right alt + left click a link to delete it.", color=(150, 150, 150))
            dpg.add_separator()
            
            # Toolbar
            with dpg.group(horizontal=True):
                dpg.add_button(label="Add Environment", callback=self.add_environment_node)
                dpg.add_button(label="Add RL Agent", callback=self.add_rl_agent_node)
                dpg.add_button(label="Add IL Agent", callback=self.add_il_agent_node)
                dpg.add_button(label="Add Visualizer", callback=self.add_visualizer_node)
                dpg.add_button(label="Clear All", callback=self.clear_all_nodes)
            
            dpg.add_separator()
            
            # Node Editor
            with dpg.node_editor(
                tag=self.node_editor_tag, 
                callback=self.link_callback,
                delink_callback=self.delink_callback,
                minimap=True,
                minimap_location=dpg.mvNodeMiniMap_Location_BottomRight
            ):
                pass
            
            # Execution controls
            dpg.add_separator()
            with dpg.group(horizontal=True):
                dpg.add_button(label="Test Connection", callback=self.test_connection)
                dpg.add_button(label="Print Node Info", callback=self.print_node_info)

    def link_callback(self, sender, app_data):
        output_attr_id, input_attr_id = app_data
        
        # Get the node IDs for these attributes
        output_node_id = self.get_node_from_attribute_id(output_attr_id)
        input_node_id = self.get_node_from_attribute_id(input_attr_id)
        
        if not output_node_id or not input_node_id:
            print(f"Error: Could not find source or target node. Output: {output_attr_id}, Input: {input_attr_id}")
            return
        
        # Draw the link
        link_id = dpg.add_node_link(output_attr_id, input_attr_id, parent=sender)
        
        # Store the link
        self.links[link_id] = (output_attr_id, input_attr_id)
        
        print(f"Connected: {self.nodes[output_node_id]['type'].value} -> {self.nodes[input_node_id]['type'].value}")
        print(f"Link ID: {link_id}, Output Attr: {output_attr_id}, Input Attr: {input_attr_id}")

    def delink_callback(self, sender, app_data):
        link_id = app_data
        
        if link_id in self.links:
            del self.links[link_id]
            print(f"Disconnected link: {link_id}")
        
        dpg.delete_item(link_id)

    def get_node_from_attribute_id(self, attr_id):
        # First check our mapping
        if attr_id in self.attribute_to_node:
            return self.attribute_to_node[attr_id]
        
        #search through nodes
        for node_id, node_data in self.nodes.items():
            for attribute in node_data['attributes']:
                if hasattr(attribute, 'tag'):
                    if attribute.tag == attr_id:
                        self.attribute_to_node[attr_id] = node_id
                        return node_id
                elif isinstance(attribute, str) and attribute == attr_id:
                    self.attribute_to_node[attr_id] = node_id
                    return node_id
        return None

    def add_environment_node(self):
        node_id = f"env_node_{self.node_counter}"
        self.node_counter += 1
        
        with dpg.node(tag=node_id, parent=self.node_editor_tag, label="Environment", pos=[50, 50]):
            # Input
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as input_attr:
                dpg.add_text("Config Input")
            
            # Actions
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_button(label="Generate Terrain", width=150,
                             callback=self.simulator.generate_random_terrain)

            # Outputs
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as output_grid_attr:
                dpg.add_text("Grid Data", indent=80)
            
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as output_state_attr:
                dpg.add_text("State", indent=110)
        
        # Store the actual attribute objects
        self.nodes[node_id] = {
            'type': NodeType.ENVIRONMENT,
            'attributes': [input_attr, output_grid_attr, output_state_attr],
            'input_attrs': {'input': input_attr},
            'output_attrs': {'output_grid': output_grid_attr, 'output_state': output_state_attr}
        }
        
        # Update mapping
        self.attribute_to_node[input_attr] = node_id
        self.attribute_to_node[output_grid_attr] = node_id
        self.attribute_to_node[output_state_attr] = node_id

    def add_rl_agent_node(self):
        node_id = f"rl_node_{self.node_counter}"
        self.node_counter += 1
        
        with dpg.node(tag=node_id, parent=self.node_editor_tag, label="RL Agent", pos=[300, 50]):
            # Inputs
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as input_env_attr:
                dpg.add_text("Environment")
            
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as input_config_attr:
                dpg.add_text("Config")
            
            # Actions
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_button(label="Train RL", width=150,
                             callback=lambda: self.rl_train_callback(node_id))

            # Outputs
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as output_agent_attr:
                dpg.add_text("Trained Agent", indent=60)
        
        self.nodes[node_id] = {
            'type': NodeType.RL_AGENT,
            'attributes': [input_env_attr, input_config_attr, output_agent_attr],
            'input_attrs': {'input_env': input_env_attr, 'input_config': input_config_attr},
            'output_attrs': {'output_agent': output_agent_attr}
        }
        
        # Update mapping
        self.attribute_to_node[input_env_attr] = node_id
        self.attribute_to_node[input_config_attr] = node_id
        self.attribute_to_node[output_agent_attr] = node_id

    def add_il_agent_node(self):
        node_id = f"il_node_{self.node_counter}"
        self.node_counter += 1
        
        with dpg.node(tag=node_id, parent=self.node_editor_tag, label="IL Agent", pos=[300, 250]):
            # Inputs
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as input_env_attr:
                dpg.add_text("Environment")
            
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as input_teacher_attr:
                dpg.add_text("Teacher (Agent)")
            
            # Actions
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_button(label="Train IL", width=150,
                             callback=lambda: self.il_train_callback(node_id))

            # Outputs
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Output) as output_agent_attr:
                dpg.add_text("Trained Agent", indent=60)
        
        self.nodes[node_id] = {
            'type': NodeType.IL_AGENT,
            'attributes': [input_env_attr, input_teacher_attr, output_agent_attr],
            'input_attrs': {'input_env': input_env_attr, 'input_teacher': input_teacher_attr},
            'output_attrs': {'output_agent': output_agent_attr}
        }
        
        # Update mapping
        self.attribute_to_node[input_env_attr] = node_id
        self.attribute_to_node[input_teacher_attr] = node_id
        self.attribute_to_node[output_agent_attr] = node_id

    def add_visualizer_node(self):
        node_id = f"vis_node_{self.node_counter}"
        self.node_counter += 1
        
        with dpg.node(tag=node_id, parent=self.node_editor_tag, label="Visualizer", pos=[600, 50]):
            # Inputs
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as input_env_attr:
                dpg.add_text("Environment Data")
            
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Input) as input_agent_attr:
                dpg.add_text("Agent Policy")
            
            # Actions
            with dpg.node_attribute(attribute_type=dpg.mvNode_Attr_Static):
                dpg.add_button(label="Show Path", width=150,
                             callback=lambda: self.visualize_callback(node_id))
                dpg.add_button(label="Evaluate RL", width=150,
                             callback=lambda: self.rl_evaluate_callback(node_id))
                dpg.add_button(label="Evaluate IL", width=150,
                             callback=lambda: self.il_evaluate_callback(node_id))
        
        self.nodes[node_id] = {
            'type': NodeType.VISUALIZER,
            'attributes': [input_env_attr, input_agent_attr],
            'input_attrs': {'input_env': input_env_attr, 'input_agent': input_agent_attr},
            'output_attrs': {}
        }
        
        # Update mapping
        self.attribute_to_node[input_env_attr] = node_id
        self.attribute_to_node[input_agent_attr] = node_id

    def find_connected_node(self, target_node_id, input_attribute_name):
        target_node = self.nodes[target_node_id]
        
        # Get the actual attribute ID for this input
        target_input_attr = target_node['input_attrs'].get(input_attribute_name)
        if not target_input_attr:
            return None
        
        # Search through links to find a connection to this attribute
        for link_id, (output_attr_id, input_attr_id) in self.links.items():
            if input_attr_id == target_input_attr:
                # Find which node owns this output attribute
                return self.get_node_from_attribute_id(output_attr_id)
        
        return None

    def rl_train_callback(self, node_id):
        env_node = self.find_connected_node(node_id, 'input_env')
        if env_node:
            print(f"[{node_id}] Starting RL Training using {env_node}...")
            if hasattr(self.simulator, 'train_rl'):
                self.simulator.train_rl()
            else:
                print("Simulator doesn't have train_rl method")
        else:
            print(f"[{node_id}] Error: Connect Environment first!")

    def rl_evaluate_callback(self, node_id):
        if hasattr(self.simulator, 'rl_agent') and self.simulator.rl_agent:
            if hasattr(self.simulator, 'evaluate'):
                self.simulator.evaluate("RL")
            else:
                print("Simulator doesn't have evaluate method")
        else:
            print(f"[{node_id}] Error: Train RL agent first!")

    def il_train_callback(self, node_id):
        env_node = self.find_connected_node(node_id, 'input_env')
        teacher_node = self.find_connected_node(node_id, 'input_teacher')
        
        if env_node and teacher_node:
            print(f"[{node_id}] Starting IL Training...")
            if hasattr(self.simulator, 'train_il'):
                if hasattr(self.simulator, 'rl_agent') and self.simulator.rl_agent:
                    self.simulator.train_il()
                else:
                    print("Error: Teacher (RL Agent) is not trained yet!")
            else:
                print("Simulator doesn't have train_il method")
        else:
            missing = []
            if not env_node:
                missing.append("Environment")
            if not teacher_node:
                missing.append("Teacher")
            print(f"[{node_id}] Error: Connect {', '.join(missing)}!")

    def il_evaluate_callback(self, node_id):
        if hasattr(self.simulator, 'il_agent') and self.simulator.il_agent:
            if hasattr(self.simulator, 'evaluate'):
                self.simulator.evaluate("IL")
            else:
                print("Simulator doesn't have evaluate method")
        else:
            print("Error: Train IL agent first!")

    def visualize_callback(self, node_id):
        env_node = self.find_connected_node(node_id, 'input_env')
        agent_node = self.find_connected_node(node_id, 'input_agent')
        
        if env_node and agent_node:
            agent_type = self.nodes[agent_node]['type']
            
            if agent_type == NodeType.RL_AGENT and hasattr(self.simulator, 'rl_agent') and self.simulator.rl_agent:
                if hasattr(self.simulator, 'visualize_agent_path'):
                    self.simulator.visualize_agent_path(self.simulator.rl_agent, "RL")
                else:
                    print("Simulator doesn't have visualize_agent_path method")
            elif agent_type == NodeType.IL_AGENT and hasattr(self.simulator, 'il_agent') and self.simulator.il_agent:
                if hasattr(self.simulator, 'visualize_agent_path'):
                    self.simulator.visualize_agent_path(self.simulator.il_agent, "IL")
                else:
                    print("Simulator doesn't have visualize_agent_path method")
            else:
                print(f"[{node_id}] Error: Connected agent is not trained or unknown type")
        else:
            missing = []
            if not env_node:
                missing.append("Environment Data")
            if not agent_node:
                missing.append("Agent Policy")
            print(f"[{node_id}] Error: Connect {', '.join(missing)}!")

    def test_connection(self):
        print("\nActive Connections")
        if not self.links:
            print("No connections.")
        for link_id, (output_attr_id, input_attr_id) in self.links.items():
            output_node_id = self.get_node_from_attribute_id(output_attr_id)
            input_node_id = self.get_node_from_attribute_id(input_attr_id)
            
            if output_node_id and input_node_id:
                output_node_name = self.nodes[output_node_id]['type'].value
                input_node_name = self.nodes[input_node_id]['type'].value
                print(f"Link {link_id}: {output_node_name} -> {input_node_name}")
            else:
                print(f"Link {link_id}: Unknown -> Unknown (Attr IDs: {output_attr_id}, {input_attr_id})")

    def print_node_info(self):
        print("\nNode Information")
        for node_id, node_data in self.nodes.items():
            print(f"\nNode: {node_id} ({node_data['type'].value})")
            
            print("Input attributes:")
            for name, attr in node_data['input_attrs'].items():
                print(f"  {name}: {attr}")
            
            print("Output attributes:")
            for name, attr in node_data['output_attrs'].items():
                print(f"  {name}: {attr}")

    def clear_all_nodes(self):
        # Delete all links
        for link_id in list(self.links.keys()):
            dpg.delete_item(link_id)
        self.links.clear()
        
        # Delete all nodes
        for node_id in list(self.nodes.keys()):
            dpg.delete_item(node_id)
        self.nodes.clear()
        
        # Clear attribute mapping
        self.attribute_to_node.clear()
        
        self.node_counter = 0
        print("Cleared all nodes and links.")