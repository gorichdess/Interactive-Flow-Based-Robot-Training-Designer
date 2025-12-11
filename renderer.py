import numpy as np

def grid_to_image(grid, scale=20, target_size=600):
    h, w = grid.shape
    
    scaled_h = h * scale
    scaled_w = w * scale
    
    img_content = np.zeros((scaled_h, scaled_w, 3), dtype=np.uint8)
    
    # Updated colors with support for labyrinth and pathfinder
    colors = {
        0: [250, 250, 250],      # Empty - white
        1: [204, 128, 51],       # Rough - tan/brown
        2: [0, 255, 0],          # Goal - green
        3: [128, 128, 128],      # Mountain - gray
        4: [0, 0, 255],          # Swamp - blue
        5: [50, 50, 50],         # Wall - dark gray (для лабіринту)
        6: [255, 0, 0],          # Start - red
        7: [255, 255, 0],        # Path - yellow
        8: [0, 255, 255],        # PathFinder path - cyan
        -1:[255, 20, 147]        # Start (резервний)
    }
    
    for i in range(h):
        for j in range(w):
            cell_value = grid[i, j]
            color = colors.get(cell_value, [255, 20, 147] )
            img_content[i*scale:(i+1)*scale, j*scale:(j+1)*scale] = color
    
    if scaled_h != target_size or scaled_w != target_size:
        final_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        
        pad_h = (target_size - scaled_h) // 2
        pad_w = (target_size - scaled_w) // 2
        
        pad_h = max(0, pad_h)
        pad_w = max(0, pad_w)
        
        end_h = min(pad_h + scaled_h, target_size)
        end_w = min(pad_w + scaled_w, target_size)
        actual_h = end_h - pad_h
        actual_w = end_w - pad_w
        
        final_img[pad_h:pad_h+actual_h, pad_w:pad_w+actual_w] = \
            img_content[:actual_h, :actual_w]
    else:
        final_img = img_content
    
    return final_img.astype(np.float32) / 255.0

def render_trajectory(grid, trajectory, scale=20, target_size=600, line_width=3, color_type="default"):
    # Create base terrain image
    base_image = grid_to_image(grid, scale, target_size)
    
    if not trajectory or len(trajectory) < 2:
        return base_image
    
    # Convert normalized image to uint8 for drawing
    img_uint8 = (base_image * 255).astype(np.uint8)
    
    # Calculate actual grid size in pixels
    h, w = grid.shape
    scaled_h = h * scale
    scaled_w = w * scale
    
    # Calculate padding if image was resized
    pad_h = (target_size - scaled_h) // 2
    pad_w = (target_size - scaled_w) // 2
    
    # Choose color based on type
    if color_type == "pathfinder":
        path_color = [0, 255, 255]  # Cyan for PathFinder
    else:
        path_color = [255, 255, 0]  # Yellow for RL/IL
    
    # Draw trajectory path
    for k in range(1, len(trajectory)):
        prev_pos = trajectory[k-1]
        curr_pos = trajectory[k]
        
        # Convert grid coordinates to pixel coordinates
        # Center of each cell
        prev_x = pad_w + (prev_pos[1] + 0.5) * scale
        prev_y = pad_h + (prev_pos[0] + 0.5) * scale
        curr_x = pad_w + (curr_pos[1] + 0.5) * scale
        curr_y = pad_h + (curr_pos[0] + 0.5) * scale
        
        # Draw line segment
        num_points = max(abs(int(curr_x - prev_x)), abs(int(curr_y - prev_y))) + 1
        for t in np.linspace(0, 1, num_points):
            x = int(prev_x * (1-t) + curr_x * t)
            y = int(prev_y * (1-t) + curr_y * t)
            
            # Draw thick point
            for dx in range(-line_width//2, line_width//2 + 1):
                for dy in range(-line_width//2, line_width//2 + 1):
                    px = x + dx
                    py = y + dy
                    if 0 <= px < target_size and 0 <= py < target_size:
                        img_uint8[py, px] = path_color
    
    # Mark start position
    start = trajectory[0]
    start_x = pad_w + (start[1] + 0.5) * scale
    start_y = pad_h + (start[0] + 0.5) * scale
    radius = scale // 2
    
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx*dx + dy*dy <= radius*radius:
                px = int(start_x + dx)
                py = int(start_y + dy)
                if 0 <= px < target_size and 0 <= py < target_size:
                    img_uint8[py, px] = [255, 0, 0]  # Red for start
    
    # Mark end position
    end = trajectory[-1]
    end_x = pad_w + (end[1] + 0.5) * scale
    end_y = pad_h + (end[0] + 0.5) * scale
    
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx*dx + dy*dy <= radius*radius:
                px = int(end_x + dx)
                py = int(end_y + dy)
                if 0 <= px < target_size and 0 <= py < target_size:
                    img_uint8[py, px] = [0, 255, 0]  # Green for goal
    
    # Convert back to normalized float
    return img_uint8.astype(np.float32) / 255.0

def render_pathfinder_path(grid, path, scale=20, target_size=600):
    return render_trajectory(grid, path, scale, target_size, line_width=3, color_type="pathfinder")