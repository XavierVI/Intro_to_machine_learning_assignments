import gymnasium

import BMPtoARR as BA
import numpy as np
import random
from gymnasium import Env
from gymnasium import spaces
import os

NOTHING = 0
PLAYER = 1
WIN = 2
LOSE = 3
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3


bw_img1 = np.array(BA.Image.open("./maps/map1.bmp"))
bw_img2 = np.array(BA.Image.open("./maps/map2.bmp"))

map1 = BA.compressbmp(bw_img1,11,11)
map2 = BA.compressbmp(bw_img2,11,11)

def get_obst(inarr):
    out_arr = []
    print(inarr[0])
    print(inarr[1])
    print(inarr[0,0])
    for x in inarr[0]:
        for y in inarr[1]:
            if BA.checkscope(x,y,1,1,inarr) == False:
                out_arr.append((x,y))
            else:
                pass
        return out_arr

class BENV(Env):
    def __init__(self, inputmap):
        self.inputmap = inputmap
        self.sq = int(inputmap.shape[0] * inputmap.shape[1])
        self.state = [NOTHING] * self.sq
        self.cumul_r = 0

        self.player_pos = random.randrange(0, self.sq)
        self.win_pos = random.randrange(0, self.sq)
        self.lose_pos = get_obst(inputmap)  # pick one position

        while self.win_pos == self.player_pos:
            self.win_pos = random.randrange(0, self.sq)
        while self.lose_pos == self.win_pos or self.lose_pos == self.player_pos:
            self.player_pos = random.randrange(0, self.sq)

        self.state[self.player_pos] = PLAYER
        self.state[self.win_pos] = WIN
        for i in range(0,len(self.lose_pos)):
            self.state[self.lose_pos[i]] = LOSE

        self.state = np.array(self.state, dtype=np.int16)
        self.obser_sp = spaces.Box(0, 3, [self.sq], dtype=np.int16)
        self.action_sp = spaces.Discrete(4)

    def step(self, action):
        info = {}
        done = False
        reward = -0.01
        prev_pos = self.player_pos

        if action == UP:
            if self.player_pos[0] - int(self.inputmap.shape[0]) >= 0:
                self.player_pos -= (int(input.shape[0]))
        elif action == DOWN:
            if self.player_pos + int(self.inputmap.shape[0]) < self.sq:
                self.player_pos += (int(self.inputmap.shape[0]))
        elif action == LEFT:
            if self.player_pos % int(self.inputmap.shape[0]) != 0:
                self.player_pos -= 1
        elif action == RIGHT:
            if self.player_pos % int(self.inputmap.shape[0]) != int(self.inputmap.shape[0])-1:
                self.player_pos += 1
        else:
            raise Exception("Invalid action")

        if self.state[self.player_pos] == WIN:
            reward = 1.0
            self.cumul_r += reward
            done = True

        elif self.state[self.player_pos] == LOSE:
            reward = -1.0
            self.cumul_r += reward
            done = True
        if not done:
            self.state[self.prev_pos] = NOTHING
            self.state[self.player_pos] = PLAYER
        self.cumul_r += reward
        return self.state, reward, done, info

    def reset(self):
        self.player_pos = random.randrange(0, self.sq)
        self.win_pos = random.randrange(0, self.sq)
        self.lose_pos = random.choice(get_obst(self.inputmap))  # pick one position

        while self.win_pos == self.player_pos:
            self.win_pos = random.randrange(0, self.sq)
        while self.lose_pos == self.win_pos or self.lose_pos == self.player_pos:
            self.player_pos = random.randrange(0, self.sq)

        self.state[self.player_pos] = PLAYER
        self.state[self.win_pos] = WIN
        self.state[self.lose_pos] = LOSE

        self.state = np.array(self.state, dtype=np.int16)
        return self.state

def returnmap1():
    return BENV(inputmap=map1)

def returnmap2():
    return BENV(inputmap=map2)