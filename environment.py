class TerrainEnv:
    def __init__(self, size=20, terrain_map=None):
        self.size = size
        self.grid = terrain_map
        print("Env created — will store terrain & agent position")

    def reset(self):
        print("Reset environment — generate / load map")
        return (0, 0)  # placeholder state

    def step(self, action):
        print(f"Agent performs action: {action}")
        return (0, 0), -1, False  # next_state, reward, done

    def get_grid(self):
        print("return numpy grid for rendering")
        return self.grid