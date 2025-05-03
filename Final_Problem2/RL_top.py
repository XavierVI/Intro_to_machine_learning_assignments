import Environment as E

#import Alan_env as AE
import Alex_agent as AA
import numpy as np
'''
indix = 15
arr = [[0 for _ in range(6)] for _ in range(6)]
arr = np.empty((6, 6))
x,y = E.get_cord(indix, arr)
print(f"for index: {indix} we got x: {x}, y: {y}")
'''

World1 = E.returnmap1()
World2 = E.returnmap2()

agent1 = AA.Agent(World1)
agent2 = AA.Agent(World2)

num_iter = 100

agent1.environment.reset()
agent2.environment.reset()
for i in range(0,num_iter):

   act1 = agent1.choose_action(World1.state, World1.player_pos)
   act2 = agent2.choose_action(World2.state, World2.player_pos)
   print(f"in round: {i}, agent1 took action: {act1}")

   st1, r1, done1, info1 = World1.step(act1)
   st2, r2, done2, info2 = World2.step(act2)
   print(f"in round: {i}, agen1:: the new state: {st1}, reward: {r1}, done1: {done1}, info1 {info1}")
   print(f"in round: {i}, agen1:: the new state: {st2}, reward: {r2}, done1: {done2}, info1 {info2}\n")

   agent1.update_q_table(st1, act1, r1,World1.prev_pos)
   agent2.update_q_table(st2, act2, r2,World2.prev_pos)

   agent1.environment.reset()
   agent2.environment.reset()

