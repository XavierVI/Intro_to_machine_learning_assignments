import numpy as np
from scipy.integrate import odeint
"""

"""

class CarModel:
    """
    This class defines the car for problem 1.
    """

    def __init__(self):
        X = np.linspace(-5, 5, 21)
        V = np.linspace(-5, 5, 21)
        U = np.array([-5, -1, -0.1, -0.01, -0.001, -0.0001, 0,
                      0.0001, 0.001, 0.01, 0.1, 1, 5])
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
        return y

    def generate_trajectory(self, s0, u, T, Q):
        """
        This function generates a trajectory of states s = (x, v).

        s0: the initial state
        u: a vector of control inputs
        T: the final time step such that s(t) is the terminal state
        Q: the Q-table
        """
        # transforms x, y, and u into an index
        idx_transform = lambda x: 2*x + 10
        u_idx_transform = lambda u: 1.2*u + 6.0

        # let the trajectory be represented as a tuple of (s, u, r)
        # where s = (x, y) is the state, u is the control input, and r is the reward
        trajectory = np.array([(s0, u[0], 0)])

        for i in range(1, T):
            # compute the next state
            s_i, u_i, _ = trajectory[i-1]

            x_idx = idx_transform(s_i[0])
            y_idx = idx_transform(s_i[1])
            u_idx = u_idx_transform(u_i)

            # compute the next state
            s
            r = Q[x_idx, y_idx, u_idx]

    



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
