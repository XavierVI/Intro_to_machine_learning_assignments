import numpy as np
from scipy.integrate import odeint
"""

"""

class CarModel:
    """
    This class defines the car for problem 1.
    """

    def __init__(self, X, V, U):
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
        if abs()
        return 0


    def generate_trajectory(self, s0, T, agent):
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
        trajectory = np.zeros((T, 3))
        trajectory[0] = [prev_s, prev_u, R]

        for i in range(1, T):
            # compute the next state
            s_i = self.get_next_state(prev_s, prev_u)
            u_i = agent.get_best_action(s_i)
            R = self.get_reward(s_i, u_i)

            # add the new sequence to the trajectory
            trajectory[i] = [s_i, u_i, R]

            # update the previous state and control input
            prev_s = s_i
            prev_u = u_i

        return trajectory
            


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

        Q = np.zeros((self.X.size, self.V.size, self.U.size))


    def get_best_action(self, s):
        """
        This function computes the best action for a given state s
        using the Q table.
        """
        x_idx = self.s_to_index[s[0]]
        v_idx = self.s_to_index[s[1]]
        q_values = self.Q[x_idx, v_idx, :]

        best_action_idx = np.argmax(q_values)
        best_action = self.U[best_action_idx]

        return best_action
