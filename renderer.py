import numpy as np

def grid_to_image(grid, scale=20, target_size=600):
    h, w = grid.shape
    
    img_content = np.zeros((h*scale, w*scale, 3), dtype=np.uint8) 

    colors = {
        0: [250, 250, 250],      # Empty - white
        1: [204, 128, 51],       # Rough - tan 
        2: [0, 255, 0],          # Goal - green
        3: [128, 128, 128],      # Mountain - gray
        4: [0, 0, 255],          # Swamp - blue
    }

    for i in range(h):
        for j in range(w):
            color = colors.get(grid[i, j], [255, 0, 0]) 
            img_content[i*scale:(i+1)*scale, j*scale:(j+1)*scale] = color
    
    #Fill with balck
    final_img = np.full((target_size, target_size, 3), 0, dtype=np.uint8)
    
    #We insert generated content in the upper left corner
    content_H, content_W, _ = img_content.shape
    
    # We check that instead of exceeding the target size
    if content_H <= target_size and content_W <= target_size:
        final_img[:content_H, :content_W] = img_content
    else:
        # For whatever reason, the content is larger than 600x600, 
        final_img = img_content[:target_size, :target_size]
        
    
    return final_img.astype(np.float32) / 255.0 #DearPyGui expects a flat array of normalized values ​​[0, 1]