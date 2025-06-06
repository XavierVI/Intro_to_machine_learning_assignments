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

def get_indice(arr, inmap):
    x = arr[0]
    y = arr[1]
    #print(x,y)
    #return ((y*inmap.shape[0]) + x)
    if inmap.shape[0] >= inmap.shape[1]:
        return (y*inmap.shape[0]) + x
    if inmap.shape[0] < inmap.shape[1]:
        return (x*inmap.shape[1]) + y

def get_cord(index, inmap):
    # print(x,y)
    # return ((y*inmap.shape[0]) + x)width = inmap.shape[1]
    width = inmap.shape[1]
    #print(f"width: {width}\n")
    #print(f"index: {index}\n")
    x = index % width
    y = index // width
    return x, y



def get_obst(inarr):
    out_arr=[]
    for x in range(0,len(inarr[0])):
        for y in range(0,len(inarr[1])):
            if BA.checkscope(x,y,1,1,inarr) == False:
                #print(x,y)
                out_arr.append((x,y))
            else:
                pass
    return out_arr

class BENV(Env):
    def __init__(self, inputmap):
        self.inputmap = inputmap
        print(f"Input map: {inputmap.shape}")
        self.height = inputmap.shape[0]
        self.width = inputmap.shape[1]
        self.sq = int(inputmap.shape[0] * inputmap.shape[1])
        self.state = [NOTHING] * self.sq
        self.cumul_r = 0

        self.player_pos = random.randrange(0, self.sq)
        self.prev_state = [NOTHING] * self.sq
        self.prev_pos = self.player_pos
        self.win_pos =  random.randrange(0, self.sq)
        self.lose_pos = get_obst(inputmap)
        #print(f"this is {self.lose_pos}")
        while self.win_pos == self.player_pos:
            self.win_pos = random.randrange(0, self.sq)
        while self.lose_pos == self.win_pos or self.lose_pos == self.player_pos:
            self.player_pos = random.randrange(0, self.sq)

        self.state[self.player_pos] = PLAYER
        self.state[self.win_pos] = WIN
        for i in range(0,len(self.lose_pos)):
            self.state[get_indice(self.lose_pos[i], self.inputmap)] = LOSE

        self.state = np.array(self.state, dtype=np.int16)
        print(self.state)
        self.obser_sp = spaces.Box(0, 3, [self.sq], dtype=np.int8)
        self.action_sp = spaces.Discrete(4)

    def step(self, action):
        print(f"action tooK: {action}")

        info = {}
        done = False
        reward = -0.01
        self.prev_pos = self.player_pos
        print(f"is action: {action} equal to {UP}")
        if action == UP:
            if self.player_pos - int(self.inputmap.shape[0]) >= 0:
                self.player_pos -= (int(self.inputmap.shape[0]))
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
            #print(action-1)
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
        self.state = [NOTHING] * self.sq
        self.player_pos = random.randrange(0, self.sq)
        self.win_pos = random.randrange(0, self.sq)
        self.lose_pos = get_obst(self.inputmap)  # pick one position

        while self.win_pos == self.player_pos:
            self.win_pos = random.randrange(0, self.sq)
        while self.lose_pos == self.win_pos or self.lose_pos == self.player_pos:
            self.player_pos = random.randrange(0, self.sq)

        self.state[self.player_pos] = PLAYER
        self.state[self.win_pos] = WIN
        for i in range(0, len(self.lose_pos)):
            self.state[get_indice(self.lose_pos[i], self.inputmap)] = LOSE

        self.state = np.array(self.state, dtype=np.int16)
        return self.state

def returnmap1():
    return BENV(inputmap=map1)

def returnmap2():
    return BENV(inputmap=map2)