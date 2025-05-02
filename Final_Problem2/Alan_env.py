import numpy as np

class Envoirnment:
    def __init__(self, grid, start, goal):
        self.grid = grid                
        self.start = start              
        self.goal = goal                  
        self.state = start                
        self.actions = {
            # up
            0: (-1, 0),
            # down
            1: (1, 0),  
            # right
            2: (0, 1),
            # left
            3: (0, -1)   
        }
    
    def reset(self):
        self.state = self.start
        return self.state
    

    def get_reward(self, state):
        if state == self.goal:
            return 1  # Goal reward
        else:
            return -1  # Small negative reward to encourage shorter paths
        
    def step(self, action):
        dx = self.actions[action]
        dy = self.actions[action]
        x = self.state
        y = self.state
        delta_x = x + dx
        delta_y = y + dy
        # Check bounds and for wall
        if self.is_valid_move(delta_x, delta_y):
            self.state = (delta_x, delta_y)

        reward = self.get_reward(self.state)
        done = self.state == self.goal

        return self.state, reward, done
    
    def is_valid_move(self, x, y):
        in_bounds = 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1]
        return in_bounds and self.grid[x, y] == 1  # 1 means free
    

map_1 = Image.open("/Users/alanpaz/Documents/CS429/CS429_Assignment1-1/Final_Problem2/maps/map1_compressed(48,48).bmp") # 532 x 528
map_2 = Image.open("/Users/alanpaz/Documents/CS429/CS429_Assignment1-1/Final_Problem2/maps/map2_compressed(48,48).bmp") # 532 x 528
bw_img1 = map_1.convert('1')  # Convert to 1-bit black & white
bw_img2 = map_2.convert('1')  # Convert to 1-bit black & white

grid1 = np.array(bw_img1)
grid2 = np.array(bw_img2)

start = (0, 0)  # Pick a valid start coordinate
goal = (40, 47)  # Pick a valid goal coordinate

env1 = Envoirnment(grid=grid1, start=start, goal=goal)
env2 = Envoirnment(grid=grid2, start=start, goal=goal)