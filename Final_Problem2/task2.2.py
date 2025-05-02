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

        return self.state