import numpy as np

def grid_to_image(grid, scale=20):
    h, w = grid.shape
    img = np.zeros((h*scale, w*scale, 3), dtype=np.uint8)  # Используем uint8

    colors = {
        0: [255, 255, 255],     # Empty - white
        1: [204, 128, 51],      # Rough - tan  
        2: [0, 255, 0],         # Goal - green
        3: [128, 128, 128],     # Mountain - gray
        4: [0, 0, 255],         # Swamp - blue
    }

    for i in range(h):
        for j in range(w):
            color = colors.get(grid[i, j], [255, 0, 0])  # Красный для неизвестных значений
            img[i*scale:(i+1)*scale, j*scale:(j+1)*scale] = color

    return img