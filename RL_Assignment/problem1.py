import numpy as np
from scipy.integrate import odeint
"""

"""


class Agent:
    """
    This class defines the agent for problem 1.
    """

    def __init__(self):
        self.car = CarModel()
        self.Q = np.zeros((21, 21, 13))
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01

        # state space and action space
        self.X = np.linspace(-5, 5, 21)
        self.V = np.linspace(-5, 5, 21)
        self.U = np.array([-5, -1, -0.1, -0.01, -0.001, -0.0001, 0,
                           0.0001, 0.001, 0.01, 0.1, 1, 5])

        # The indices for the Q table are retrieved from these dictionaries
        # this one can be used for either x or v
        self.s_to_index = {x_val: i for i, x_val in enumerate(self.X)}
        self.u_to_index = {u_val: i for i, u_val in enumerate(self.U)}

        self.Q = np.zeros((self.X.size, self.V.size, self.U.size))

    
    def discretize_state(self, s):
        """
        Discretizes the state s = (x, v) to the nearest values in X and V.
        """
        x = s[0]
        v = s[1]
        x_discrete = self.X[np.argmin(np.abs(self.X - x))]
        v_discrete = self.V[np.argmin(np.abs(self.V - v))]
        return (x_discrete, v_discrete)


    def get_best_action(self, s):
        """
        This function computes the best action for a given state s
        using the Q table.
        """
        # transform the state into one of our discrete states
        s = self.discretize_state(s)
        x_idx = self.s_to_index[s[0]]
        v_idx = self.s_to_index[s[1]]
        q_values = self.Q[x_idx, v_idx, :]

        best_action_idx = np.argmax(q_values)
        best_action = self.U[best_action_idx]

        return best_action


class CarModel:
    """
    This class defines the car for problem 1.
    """

    def __init__(self):
        # size of each time step
        self.delta = 0.1
        self.step = np.linspace(0, self.delta)

    def model(self, s, t, u):
        dsdt = [ s[1], u  ]
        return dsdt

    def get_next_state(self, s, u):
        """
        This function computes the next state of the car given the
        current state and the control input.
        """
        # compute the next state
        y = odeint(self.model, s, self.step, args=(u,))
        return y[-1]

    def get_reward(self, s, u):
        """
        This function goes through a sequence of steps to accumulate
        the reward, such that we punish the agent when preventing the
        state from being (0, 0) and reward it when it is at or reaching
        (0, 0).
        """
        reward = 0
        x, v = s
        # next_x, next_v = next_s

        goal_reward = 10
        velocity_reward = -0.1 * abs(v)
        position_reward = -0.5 * abs(x)
        control_reward  = -0.01 * abs(u)

        # Reward for reaching the goal
        if x == 0.0 and v == 0.0:
            reward += goal_reward

        # Punish for being very far away from the goal
        if abs(x) > 5:
            reward -= goal_reward

        # encourage moving towards the goal
        # if abs(next_x) < abs(x):
        #     reward += position_reward

        if abs(x) < 0.5 and abs(v) < 0.5:
            reward += 5

        if (abs(x) < 0.5 and abs(v) < 0.5) and abs(u) < 0.01:
            # bonus points for keeping the acceleration low
            # if the car is close to the goal
            reward += 5

        # punish for moving away from the goal
        if x != 0 and np.sign(x) == np.sign(v):
            reward += velocity_reward

        # punish the model when the position is near 0,
        # and the velocity and acceleration are high
        if abs(x) < 0.5 and abs(v) > 0.5:
            reward += velocity_reward
        
        if abs(x) < 0.5 and abs(u) > 0.5:
            reward += control_reward

        if v != 0 and np.sign(v) == np.sign(u):
            reward += control_reward

        return reward


    def generate_trajectory(self, s0, T, agent: Agent):
        """
        This function generates a trajectory of states s = (x, v), using
        a Q table to select the control input at each time step.

        s0: the initial state
        T: the final time step such that s(t) is the terminal state
        Q: the Q-table
        """
        prev_s = s0

        # compute the next control input
        prev_u = agent.get_best_action(prev_s)

        # compute the next reward outside of the loop
        R = self.get_reward(prev_s, prev_u)
        
        # let the trajectory be represented as a T x 3 array where
        # each entry is (s, u, r)
        # where s = (x, y) is the state,
        # u is the control input, and r is the reward
        trajectory = np.zeros((T,), dtype=object)
        trajectory[0] = (prev_s, prev_u, R)

        for i in range(1, T):
            # compute the next state
            s_i = self.get_next_state(prev_s, prev_u)
            u_i = agent.get_best_action(s_i)
            R = self.get_reward(s_i, u_i)

            # add the new sequence to the trajectory
            trajectory[i] = (s_i, u_i, R)

            # update the previous state and control input
            prev_s = s_i
            prev_u = u_i

        return trajectory
            
agent = Agent()
s0 = (1, -1)

env = CarModel()
T = 10
trajectory = env.generate_trajectory(s0, T, agent)
print(trajectory)
