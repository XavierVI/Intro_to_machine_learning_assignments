import numpy as np
from scipy.integrate import odeint
import pyprind

import matplotlib.pyplot as plt
"""

"""


class Agent:
    """
    This class defines the agent for problem 1.
    """

    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1, state_space_size=21):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # state space and action space
        self.X = np.linspace(-5, 5, state_space_size)
        self.V = np.linspace(-5, 5, state_space_size)
        self.U = np.array([-5, -1, -0.1, -0.01, -0.001, -0.0001, 0,
                           0.0001, 0.001, 0.01, 0.1, 1, 5])

        # The indices for the Q table are retrieved from these dictionaries
        # this one can be used for either x or v
        self.s_to_index = {x_val: i for i, x_val in enumerate(self.X)}
        self.u_to_index = {u_val: i for i, u_val in enumerate(self.U)}

        self.Q = np.random.default_rng(1).uniform(
            size=(self.X.size, self.V.size, self.U.size)
        )

    
    def discretize_state(self, s):
        """
        Discretizes the state s = (x, v) to the nearest values in X and V.
        """
        # clip values to keep them within -5 and 5
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
        # for the epsilon-greedy policy, we select a random action
        # with probability epsilon
        if np.random.default_rng().uniform() < self.epsilon:
            return np.random.choice(self.U)

        # transform the state into one of our discrete states
        s = self.discretize_state(s)
        x_idx = self.s_to_index[s[0]]
        v_idx = self.s_to_index[s[1]]
        q_values = self.Q[x_idx, v_idx, :]

        best_action_idx = np.argmax(q_values)
        best_action = self.U[best_action_idx]

        return best_action


    def update_Q(self, s, u, r, next_s):
        """
        This function updates the Q table using the update method
        from the slides.

        s: the state when the action was made
        u: the action taken
        r: the reward received
        next_s: the state after the action was taken
        """
        x_idx = self.s_to_index[s[0]]
        v_idx = self.s_to_index[s[1]]
        u_idx = self.u_to_index[u]
        next_x_idx = self.s_to_index[next_s[0]]
        next_v_idx = self.s_to_index[next_s[1]]

        # get the maximum Q value for the next state
        max_next_s = np.max(
            self.Q[next_x_idx, next_v_idx, :]
        )
        
        # update the Q table
        self.Q[x_idx, v_idx, u_idx] += \
            self.alpha * (r + self.gamma * max_next_s - self.Q[x_idx, v_idx, u_idx])



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


    def get_reward(self, s, u, next_s):
        x, v = s
        next_x, next_v = next_s

        reward = 0
        goal_reward = 10
        position_penalty = -0.2 * abs(next_x)
        velocity_penalty = -0.1 * abs(next_v)
        control_penalty = -0.01 * abs(u)

        # Reward for reaching the goal (using a threshold)
        if abs(next_x) < 0.1 and abs(next_v) < 0.1:
            reward += goal_reward

        reward += position_penalty + velocity_penalty + control_penalty

        # Encourage moving towards zero
        if x * next_x < 0:  # x changed sign (crossed zero)
            reward += 1
        if v * next_v < 0:  # v changed sign (crossed zero)
            reward += 0.5

        # Out-of-bounds penalty (softer)
        if abs(next_x) > 5 or abs(next_v) > 5:
            reward -= 5

        return reward


    def generate_trajectory(self, s0, T, agent: Agent):
        """
        This function generates a trajectory of states s = (x, v), using
        a Q table to select the control input at each time step.

        s0: the initial state
        T: the final time step such that s(t) is the terminal state
        Q: the Q-table
        """
        # let the trajectory be represented as a T x 3 array where
        # each entry is (s, u, r)
        # where s = (x, y) is the state,
        # u is the control input, and r is the reward
        trajectory = np.zeros((T, 4))
        prev_s = s0
        # compute the next control input
        prev_u = agent.get_best_action(prev_s)
        
        for i in range(0, T):
            next_s = self.get_next_state(prev_s, prev_u)
            # only used for the next iteration
            next_u = agent.get_best_action(next_s)

            # compute the reward using the current state and control input
            # and the next state
            R = self.get_reward(prev_s, prev_u, next_s)

            # add the new sequence to the trajectory
            trajectory[i] = [prev_s[0], prev_s[1], prev_u, R]

            # update the previous state and control input
            # for the next iteration
            prev_s = next_s
            prev_u = next_u

        return trajectory
            

def Q_learning(agent: Agent, env: CarModel, num_episodes=50, time_steps=10):
    """
    This function is an implementation of the Q-learning algorithm.

    num_episodes: the number of episodes to run
    
    """
    history = []
    pbar = pyprind.ProgBar(num_episodes, title="Optimizing policy...", width=40)

    for episode in range(num_episodes):
        # randomly select the initial state from X and V
        random_idx1 = np.random.default_rng(1).choice(agent.X.size)
        random_idx2 = np.random.default_rng(1).choice(agent.V.size)
        s_i = (agent.X[random_idx1], agent.V[random_idx2])

        final_reward = 0

        for t in range(time_steps):
            u = agent.get_best_action(s_i)
            next_s = agent.discretize_state(env.get_next_state(s_i, u))
            reward = env.get_reward(s_i, u, next_s)

            # print(f"Episode {episode}, Time step {t}: s = {s_i}, u = {u}, r = {reward}")
            agent.update_Q(s_i, u, reward, next_s)
            
            s_i = next_s
            final_reward += reward
        
        history.append(final_reward)
        pbar.update(1)

    pbar.stop()

    return history


def evaluate_policy(agent: Agent, env: CarModel, num_episodes=10, time_steps=10):
    """
    This function evaluates the policy learned by the agent.
    It runs the agent for a number of episodes and returns the
    average reward.
    """
    # stores the average reward for each episode
    # and the number of steps taken
    # history = np.array([])
    pbar = pyprind.ProgBar(num_episodes, title="Evaluating policy...", width=40)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig2, ax = plt.subplots(1, 1, figsize=(12, 6))

    for episode in range(num_episodes):
        # randomly select the initial state from X and V
        random_idx1 = np.random.default_rng(1).choice(agent.X.size)
        random_idx2 = np.random.default_rng(1).choice(agent.V.size)
        s0 = (agent.X[random_idx1], agent.V[random_idx2])
        # generate the trajectory
        trajectory = env.generate_trajectory(s0, time_steps, agent)
        
        # plotting
        axes[0].plot(range(time_steps), trajectory[:, 0], label="episode " + str(episode))
        axes[1].plot(range(time_steps), trajectory[:, 1], label="episode " + str(episode))

        ax.plot(trajectory[:, 3], '-', label="episode " + str(episode))
        ax.legend(loc="lower left")
        
        
        # history = np.append(history, trajectory[:, 3])
        pbar.update(1)

    pbar.stop()

    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    axes[0].set_title("Position over time")
    axes[0].grid()
    axes[0].legend(loc="lower left")
    
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("v")
    axes[1].set_title("Velocity over time")
    axes[1].grid()
    axes[1].legend(loc="lower left")

    ax.set_xlabel("Time step (t)")
    ax.set_ylabel("Reward (r)")
    ax.set_title("Reward history per episode")
    ax.legend(loc="lower left")
    ax.grid()

    plt.tight_layout()
    # plt.savefig(fname="./problem1_figs/problem1-3.png", dpi=300)
    plt.show()

    return history


        
time_steps = 200
num_episodes = 100

agent = Agent(
    alpha=0.1,
    gamma=0.9,
    epsilon=0.1
)
environment = CarModel()

history = Q_learning(
    agent,
    environment,
    num_episodes=num_episodes,
    time_steps=time_steps
)

eval_history = evaluate_policy(
    agent,
    environment,
    num_episodes=3,
    time_steps=100
)
